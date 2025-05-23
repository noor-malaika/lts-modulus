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
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

try:
    import apex
except Exception:
    pass

from datapipe.shell_dataset import ShellDataset, Hdf5Dataset
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from modulus.launch.logging.wandb import initialize_wandb
from modulus.launch.utils import load_checkpoint, save_checkpoint
# from modulus.models.meshgraphnet import MeshGraphNet
from meshgraphnet import MeshGraphNet

from utils import get_datapoint_idx, get_data_splits, save_test_idx
import importlib
import utils
import sys
import re

sys.argv = [re.sub(r"^--", "", arg) for arg in sys.argv]
loss_mapping = {
    "MSELoss": "torch.nn",
    "LogCoshLoss": "custom_loss",
    "MRAELoss": "custom_loss",
    "HuberLoss": "torch.nn",
    "MultiComponentLossWithUncertainty": "custom_loss",
    "L1Loss": "torch.nn",
}

class MGNTrainer:
    def __init__(self, cfg: DictConfig, dist, rank_zero_logger, main_loss_fn, main_loss_module):
        self.dist = dist
        self.rank_zero_logger = rank_zero_logger
        self.amp = cfg.amp
        self.normalization = cfg.normalization

        # splitting dataset
        all_idx = get_datapoint_idx(cfg.data_path)
        train_idx, val_idx, test_idx = get_data_splits(all_idx, cfg.num_training_samples, cfg.num_validation_samples, cfg.num_test_samples)

        # saving test_idx for future testing
        save_test_idx(test_idx)

        # instantiate dataset
        train_hdf5 = Hdf5Dataset(cfg.data_path, train_idx, len(train_idx))
        dataset = ShellDataset(
            name="shell_train",
            dataset_split=train_hdf5,
            split="train",
            num_samples=cfg.num_training_samples,
            normalization=self.normalization,
        )

        # instantiate validation dataset
        val_hdf5 = Hdf5Dataset(cfg.data_path, val_idx, len(val_idx))
        self.validation_dataset = ShellDataset(
            name="shell_validation",
            dataset_split=val_hdf5,
            split="validation",
            num_samples=cfg.num_validation_samples,
            normalization=self.normalization,
        )

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
            self.validation_dataset,
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
            mlp_activation_fn="relu",
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
        main_loss_module = importlib.import_module(main_loss_module)
        if main_loss_fn == "MultiComponentLossWithUncertainty":
            self.criterion = getattr(main_loss_module, main_loss_fn)(
                loss_mapping[cfg.loss], cfg.loss
            )
        else:
            self.criterion = getattr(main_loss_module, main_loss_fn)()
        try:
            self.optimizer = apex.optimizers.FusedAdam(
                self.model.parameters(), lr=cfg.lr
            )
            rank_zero_logger.info("Using FusedAdam optimizer")
        except Exception:
            loss_params = (
                list(self.criterion.parameters()) if self.criterion.parameters() else []
            )
            self.optimizer = torch.optim.Adam(
                loss_params + list(self.model.parameters()), lr=cfg.lr
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=1, eta_min=1e-4
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
        self.metrics = ["mse", "rmse", "mae", "mare"]

    def train(self, graph):
        graph = graph.to(self.dist.device)
        self.optimizer.zero_grad()
        loss = self.forward(graph)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, graph):
        # forward pass
        with autocast(enabled=self.amp):
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            loss = self.criterion(pred, graph.ndata["y"])
            return loss

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
    def validation(self, metric):
        error_keys = ["disp_x", "disp_y", "disp_z"]
        errors = {key: 0 for key in error_keys}
        error_fn = getattr(utils, metric)
        for i, graph in enumerate(self.validation_dataloader):
            graph = graph.to(self.dist.device)
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)

            for index, key in enumerate(error_keys):
                pred_val = pred[:, index : index + 1]
                target_val = graph.ndata["y"][:, index : index + 1]
                if self.normalization == "max_abs":
                    pred_val = self.validation_dataset.max_abs_denorm(
                        pred_val, graph.ndata["max_abs"][key]
                    )
                    target_val = self.validation_dataset.max_abs_denorm(
                        target_val, graph.ndata["max_abs"][key]
                    )
                elif self.normalization == "z_score":
                    pred_val = self.validation_dataset.z_score_denorm(
                        pred_val,
                        self.validation_dataset.node_stats[f"{key}_mean"],
                        self.validation_dataset.node_stats[f"{key}_std"],
                    )
                    target_val = self.validation_dataset.z_score_denorm(
                        target_val,
                        self.validation_dataset.node_stats[f"{key}_mean"],
                        self.validation_dataset.node_stats[f"{key}_std"],
                    )
                elif self.normalization == "min-max":
                    pred_val = self.validation_dataset.min_max_denorm(
                        pred_val,
                        self.validation_dataset.node_stats[f"{key}_min"],
                        self.validation_dataset.node_stats[f"{key}_max"],
                    )
                    target_val = self.validation_dataset.min_max_denorm(
                        target_val,
                        self.validation_dataset.node_stats[f"{key}_min"],
                        self.validation_dataset.node_stats[f"{key}_max"],
                    )
                print("Prediction range:", pred_val.min(), pred_val.max())
                print("Target range:", target_val.min(), target_val.max())
                errors[key] += error_fn(pred_val, target_val)

        for key in error_keys:
            errors[key] = errors[key] / len(self.validation_dataloader)
            self.rank_zero_logger.info(
                f"{metric}_validation error_{key} (%): {errors[key]}"
            )

        wandb.log(
            {
                f"{metric}_val_disp_x_error (%)": errors["disp_x"],
                f"{metric}_val_disp_y_error (%)": errors["disp_y"],
                f"{metric}_val_disp_z_error (%)": errors["disp_z"],
            }
        )


@hydra.main(version_base="1.3", config_path="conf/single_run_conf", config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        main_loss_fn = cfg["main_loss"]
        main_loss_module = loss_mapping[main_loss_fn]
        run_name = f"main_loss_{main_loss_fn}_loss_{cfg.loss}"
    except:
        main_loss_fn = cfg["loss"]
        main_loss_module = loss_mapping[main_loss_fn]
        run_name = f"805_relu_l1loss_min_max_0_1_loss_{main_loss_fn}_norm_{cfg.normalization}"
    
    
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    wandb.login(key=os.environ.get("WANDB_API_KEY"))  ### wandb api key

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # initialize loggers
    initialize_wandb(
        project="shell_mgn_multi_head_805",
        entity="malaikanoor7864-mnsuam",
        name=run_name,
        group="Test_run",
        mode=cfg.wandb_mode,
        config=cfg_dict,
    )

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    rank_zero_logger.file_logging()
    torch.cuda.empty_cache()
    trainer = MGNTrainer(cfg, dist, rank_zero_logger, main_loss_fn, main_loss_module)
    wandb.watch(trainer.model, log='all', log_freq=10)
    start = time.time()
    rank_zero_logger.info("Training started...")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        loss_agg = 0
        rank_zero_logger.info(f"epoch: {epoch}")

        for i, graph in enumerate(trainer.dataloader):
            loss = trainer.train(graph)
            loss = loss.detach().cpu().numpy()
            loss_agg += loss
            wandb.log({"graph_loss_per_epoch": loss})
            rank_zero_logger.info(
                f"\tgraph_{i}_loss_per_epoch: {loss:10.3e}, lr: {trainer.get_lr()}"
            )
        loss_agg /= len(trainer.dataloader)
        rank_zero_logger.info(
            f"epoch: {epoch}, {main_loss_fn}_loss: {loss_agg:10.3e}, lr: {trainer.get_lr()}"
        )
        wandb.log({"loss": loss_agg})

        # validation
        if dist.rank == 0:
            for metric in trainer.metrics:
                trainer.validation(metric)

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
    end = time.time()
    rank_zero_logger.info("Training completed!")
    rank_zero_logger.info(f"Total time spent for {run_name}: {end-start:.4f}")
    wandb.finish()

if __name__ == "__main__":
    main()
