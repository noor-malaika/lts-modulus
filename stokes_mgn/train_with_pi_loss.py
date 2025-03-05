# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import hydra
import torch
import wandb
from dgl.dataloading import GraphDataLoader
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from utils_for_train_with_pi import get_dataset
from pi_fine_tuning import PhysicsInformedFineTuner, PhysicsInformer, Stokes
import numpy as np

try:
    import apex
except:
    pass

from modulus.datapipes.gnn.stokes_dataset import StokesDataset
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from modulus.launch.logging.wandb import initialize_wandb
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.models.meshgraphnet import MeshGraphNet

from utils import relative_lp_error
from pi_fine_tuning import MdlsSymDNN

class GNNWrapper(MdlsSymDNN):
    def __init__(self, arch):
        super().__init__(arch)
        self.gnn = ()
    
    def forward(self, inputs):
        return self.gnn(inputs)

class MGNTrainer:
    def __init__(self, cfg: DictConfig, dist, rank_zero_logger):
        self.dist = dist
        self.rank_zero_logger = rank_zero_logger
        self.amp = cfg.amp

        # instantiate dataset
        dataset = StokesDataset(
            name="stokes_train",
            data_dir=to_absolute_path(cfg.data_dir),
            split="train",
            num_samples=cfg.num_training_samples,
        )
        self.train_file_paths = dataset.data_list
        self.train_file_num = 0
        # instantiate validation dataset
        validation_dataset = StokesDataset(
            name="stokes_validation",
            data_dir=to_absolute_path(cfg.data_dir),
            split="validation",
            num_samples=cfg.num_validation_samples,
        )
        self.val_file_paths = validation_dataset.data_list
        self.val_file_num = 0
        self.pred_tensor = None
        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
        )

        # instantiate validation dataloader
        self.validation_dataloader = GraphDataLoader(
            validation_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            use_ddp=False,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            cfg.input_dim_nodes,
            cfg.input_dim_edges,
            cfg.output_dim,
            aggregation=cfg.aggregation,
            hidden_dim_node_encoder=cfg.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=cfg.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=cfg.hidden_dim_node_decoder,
        )
        self.node_pde = Stokes(nu=self.nu, dim=2)
        self.phy_informer = PhysicsInformer(
            required_outputs=["continuity", "momentum_x", "momentum_y"],
            equations=self.node_pde,
            grad_method="autodiff",
            device=self.device,
        )

        if cfg.jit:
            self.model = torch.jit.script(self.model).to(dist.device)
        else:
            self.model = self.model.to(dist.device)

        # distributed data parallel for multi-node training
        if dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )

        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        self.criterion = torch.nn.MSELoss()
        try:
            self.optimizer = apex.optimizers.FusedAdam(
                self.model.parameters(), lr=cfg.lr
            )
            rank_zero_logger.info("Using FusedAdam optimizer")
        except: ##### continue from here
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        # load checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )

    def train(self, graph):
        graph = graph.to(self.dist.device)
        self.optimizer.zero_grad()
        loss = self.forward(graph) ## 
        self.backward(loss)
        self.scheduler.step()
        self.train_file_num += 1
        return loss

    def forward(self, graph):
        # forward pass
        with autocast(enabled=self.amp):
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            self.pred_tensor = pred
            loss = self.criterion(pred, graph.ndata["y"]) ## here pde loss should be added
            return loss

    def get_gnn_preds(self, pred_tensor):
        gnn_u = pred_tensor[:,0]
        gnn_v = pred_tensor[:,1]
        gnn_p = pred_tensor[:,2]
        return (gnn_u, gnn_v, gnn_p)
    
    def pi_loss(self):
        # inflow points
        x_in, y_in = self.coords_inflow[:, 0:1], self.coords_inflow[:, 1:2]
        results_inflow = self.model({"x": x_in, "y": y_in})
        pred_u_in, pred_v_in = results_inflow["u"], results_inflow["v"]

        # no-slip points
        x_no_slip, y_no_slip = self.coords_noslip[:, 0:1], self.coords_noslip[:, 1:2]
        results_noslip = self.model({"x": x_no_slip, "y": y_no_slip})
        pred_u_noslip, pred_v_noslip = results_noslip["u"], results_noslip["v"]

        # interior points
        x_int, y_int = self.coords[:, 0:1], self.coords[:, 1:2]
        model_out = self.model({"x": x_int, "y": y_int})
        results_int = self.phy_informer.forward(
            {
                "coordinates": self.coords,
                "u": model_out["u"],
                "v": model_out["v"],
                "p": model_out["p"],
            }
        )
        pred_mom_u, pred_mom_v, pred_cont = (
            results_int["momentum_x"],
            results_int["momentum_y"],
            results_int["continuity"],
        )
        pred_u, pred_v, pred_p = model_out["u"], model_out["v"], model_out["p"]

        u_in, v_in = self.parabolic_inflow(self.coords_inflow[:, 1:2])

        # Compute losses
        # data loss
        loss_u = torch.mean((self.gnn_u - pred_u) ** 2)
        loss_v = torch.mean((self.gnn_v - pred_v) ** 2)
        loss_p = torch.mean((self.gnn_p - pred_p) ** 2)

        # inflow boundary condition loss
        loss_u_in = torch.mean((u_in - pred_u_in) ** 2)
        loss_v_in = torch.mean((v_in - pred_v_in) ** 2)

        # noslip boundary condition loss
        loss_u_noslip = torch.mean(pred_u_noslip**2)
        loss_v_noslip = torch.mean(pred_v_noslip**2)

        # pde loss
        loss_mom_u = torch.mean(pred_mom_u**2)
        loss_mom_v = torch.mean(pred_mom_v**2)
        loss_cont = torch.mean(pred_cont**2)

        return (
            loss_u,
            loss_v,
            loss_p,
            loss_u_in,
            loss_v_in,
            loss_u_noslip,
            loss_v_noslip,
            loss_mom_u,
            loss_mom_v,
            loss_cont,
        )

    def pi_main(self, path):
        (
        ref_u,
        ref_v,
        ref_p,
        coords,
        coords_inflow,
        coords_outflow,
        coords_wall,
        coords_polygon,
        nu,
        ) = get_dataset(path)
        gnn_u, gnn_v, gnn_p = self.get_gnn_preds(self.pred_tensor)
        coords_noslip = np.concatenate([coords_wall, coords_polygon], axis=0)

        self.ref_u = torch.tensor(ref_u).float().to(self.device)
        self.ref_v = torch.tensor(ref_v).float().to(self.device)
        self.ref_p = torch.tensor(ref_p).float().to(self.device)

        self.gnn_u = torch.tensor(gnn_u).float().to(self.device)
        self.gnn_v = torch.tensor(gnn_v).float().to(self.device)
        self.gnn_p = torch.tensor(gnn_p).float().to(self.device)

        self.coords = torch.tensor(coords, requires_grad=True).float().to(self.device)
        self.coords_inflow = (
            torch.tensor(coords_inflow, requires_grad=True).float().to(self.device)
        )
        self.coords_noslip = (
            torch.tensor(coords_noslip, requires_grad=True).float().to(self.device)
        )

    def backward(self, loss):
        # backward pass
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        lr = self.get_lr()
        wandb.log({"lr": lr})

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    @torch.no_grad()
    def validation(self):
        error_keys = ["u", "v", "p"]
        errors = {key: 0 for key in error_keys}

        for graph in self.validation_dataloader:
            graph = graph.to(self.dist.device)
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)

            for index, key in enumerate(error_keys):
                pred_val = pred[:, index : index + 1]
                target_val = graph.ndata["y"][:, index : index + 1]
                errors[key] += relative_lp_error(pred_val, target_val)

        for key in error_keys:
            errors[key] = errors[key] / len(self.validation_dataloader)
            self.rank_zero_logger.info(f"validation error_{key} (%): {errors[key]}")

        wandb.log(
            {
                "val_u_error (%)": errors["u"],
                "val_v_error (%)": errors["v"],
                "val_p_error (%)": errors["p"],
            }
        )


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    wandb.login(key=os.environ.get("WANDB_API_KEY")) ### wandb api key

    # initialize loggers
    initialize_wandb(
        project="modulus-stokes-flow",
        entity="malaikanoor7864-mnsuam",
        name="Stokes-Training",
        group="Stokes-DDP-Group",
        mode=cfg.wandb_mode,
    )

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    rank_zero_logger.file_logging()

    trainer = MGNTrainer(cfg, dist, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        loss_agg = 0
        for graph in trainer.dataloader:
            loss = trainer.train(graph)
            loss_agg += loss.detach().cpu().numpy()
        loss_agg /= len(trainer.dataloader)
        rank_zero_logger.info(
            f"epoch: {epoch}, loss: {loss_agg:10.3e}, lr: {trainer.get_lr()}, time per epoch: {(time.time() - start):10.3e}"
        )
        wandb.log({"loss": loss_agg})

        # validation
        if dist.rank == 0:
            trainer.validation()

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            rank_zero_logger.info(f"Saved model on rank {dist.rank}")
            start = time.time()
    rank_zero_logger.info("Training completed!")


if __name__ == "__main__":
    main()
