import importlib

import torch
import wandb
from dgl.dataloading import GraphDataLoader
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from modulus.launch.utils import load_checkpoint
from shellmgn.dataloader.dataloader import Hdf5Dataset, ShellDataset
from shellmgn.models.meshgraphnet import MeshGraphNet
import shellmgn.utils as utils
from shellmgn.utils import get_data_splits, get_datapoint_idx, save_test_idx


loss_mapping = {
    "MSELoss": "torch.nn",
    "LogCoshLoss": "shellmgn.losses.logcosh",
    "MRAELoss": "shellmgn.losses.mrae",
    "HuberLoss": "torch.nn",
    "MultiComponentLossWithUncertainty": "shellmgn.losses.multi_comp_uncertain",
    "MultiComponentLoss": "shellmgn.losses.multi_comp",
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
        save_test_idx(test_idx, cfg.test_idx)

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
