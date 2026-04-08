import os
import time

import hydra
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from modulus.launch.logging.wandb import initialize_wandb
from modulus.launch.utils import save_checkpoint
from shellmgn.trainer.trainer import MGNTrainer, loss_mapping

@hydra.main(version_base="1.3", config_path="conf/single_run_conf", config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        main_loss_fn = cfg["main_loss"]
        main_loss_module = loss_mapping[main_loss_fn]
        run_name = f"main_loss_{main_loss_fn}_loss_{cfg.loss}"
    except Exception:
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
        project="shellmgn_multi_head_805",
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
