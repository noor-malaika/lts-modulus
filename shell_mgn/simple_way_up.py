import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from dgl.nn.pytorch.conv import GraphConv

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
import os
import time
import hydra
import torch
import wandb
from dgl.dataloading import GraphDataLoader
from hydra.utils import to_absolute_path

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

class StableGraphNetWithEdges(nn.Module):
    def __init__(
        self,
        input_dim_nodes,
        input_dim_edges,
        output_dim=3,  # dx, dy, dz
        hidden_dim=128,
        processor_size=4,
        mlp_activation_fn="silu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[mlp_activation_fn]

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim_nodes, hidden_dim),
            act_fn(),
            nn.LayerNorm(hidden_dim),
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(input_dim_edges, hidden_dim),
            act_fn(),
            nn.LayerNorm(hidden_dim),
        )

        # GNN processor (now uses edge features)
        self.processor = nn.ModuleList()
        for _ in range(processor_size):
            self.processor.append(
                GraphConv(hidden_dim, hidden_dim, activation=act_fn())
            )

        # Decoders (dx, dy, dz)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, g, node_feats, edge_feats):
        # Encode nodes and edges
        h_nodes = self.node_encoder(node_feats)
        h_edges = self.edge_encoder(edge_feats)

        # Update edge features (optional: project to same dim)
        g.edata["he"] = h_edges

        # Message passing with edge features
        for conv in self.processor:
            h_nodes_new = conv(g, h_nodes)
            h_nodes = h_nodes + h_nodes_new  # Residual connection

        # Decode to [dx, dy, dz]
        return self.decoder(h_nodes)

    def monitor_gradients(self):
        """Utility to check gradient flow."""
        grads = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.abs().mean().item()
        return grads


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
        self.model = StableGraphNetWithEdges(
            input_dim_nodes=cfg.input_dim_nodes,
            input_dim_edges=cfg.input_dim_edges,
            output_dim=cfg.output_dim,
            processor_size=4,
            mlp_activation_fn='silu',
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
        except Exception:  ##### continue from here
            loss_params = (
                list(self.criterion.parameters()) if self.criterion.parameters() else []
            )
            self.optimizer = torch.optim.Adam(
                loss_params + list(self.model.parameters()), lr=cfg.lr
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=1, eta_min=0.01
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
        self.metrics = ["mse", "rmse", "mae"]
        self.logger =  rank_zero_logger

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
            pred = self.model(graph, graph.ndata["x"], graph.edata["x"])
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
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        self.logger.info(self.model.monitor_gradients())
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
            pred = self.model(graph, graph.ndata["x"], graph.edata["x"])

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
        main_loss_fn = cfg["loss"]
        main_loss_module = loss_mapping[main_loss_fn]
        run_name = f"loss_{main_loss_fn}_norm_{cfg.normalization}"
    except:
        main_loss_fn = cfg["main_loss"]
        main_loss_module = loss_mapping[main_loss_fn]
        run_name = f"main_loss_{main_loss_fn}_loss_{cfg.loss}"
    
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    wandb.login(key=os.environ.get("WANDB_API_KEY"))  ### wandb api key

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # initialize loggers
    initialize_wandb(
        project="shell_mgn_multi_head_without_gradnorm",
        entity="malaikanoor7864-mnsuam",
        name=run_name,
        group="Watch-Gradients",
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
