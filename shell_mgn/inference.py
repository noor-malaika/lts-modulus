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

import hydra
import torch
from hydra.utils import to_absolute_path
from modulus.launch.logging import PythonLogger
from modulus.launch.utils import load_checkpoint
from meshgraphnet import MeshGraphNet
from omegaconf import DictConfig

from utils import mse, load_test_idx, create_vtk_from_graph

try:
    from dgl.dataloading import GraphDataLoader
except Exception:
    raise ImportError(
        "Stokes  example requires the DGL library. Install the "
        + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )

try:
    pass
except Exception:
    raise ImportError(
        "Stokes  Dataset requires the pyvista library. Install with "
        + "pip install pyvista"
    )
from datapipe.shell_dataset import Hdf5Dataset, ShellDataset


class MGNRollout:
    def __init__(self, cfg: DictConfig, logger):
        self.logger = logger
        self.results_dir = cfg.results_dir

        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        test_idx = load_test_idx()

        test_hdf5 = Hdf5Dataset(cfg.data_path, test_idx, len(test_idx))
        self.dataset = ShellDataset(
            name="shell_test",
            dataset_split=test_hdf5,
            split="test",
            num_samples=cfg.num_test_samples,
            normalization=cfg.normalization,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
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
        self.model = self.model.to(self.device)

        # enable train mode
        self.model.eval()

        # load checkpoint
        _ = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            device=self.device,
            # epoch=48,  #### change to load ckpt of choice, or None for loading latest saved
        )

    def predict(self):
        """
        Run the prediction process.

        Parameters:
        -----------
        save_results: bool
            Whether to save the results in form of a .vtp file, by default False

        Returns:
        --------
        None
        """

        self.pred, self.graphs = [], []
        stats = {
            key: value.to(self.device) for key, value in self.dataset.node_stats.items()
        }
        for i, graph in enumerate(self.dataloader):
            graph = graph.to(self.device)
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph).detach()

            keys = ["disp_x", "disp_y", "disp_z"]
            ### read graph_data/ create polydata
            data_i = self.dataset.dataset_split[i]
            polydata = create_vtk_from_graph(data_i)
            with torch.no_grad():
                for key_index, key in enumerate(keys):
                    pred_val = pred[:, key_index : key_index + 1]
                    target_val = graph.ndata["y"][:, key_index : key_index + 1]

                    pred_val = self.dataset.min_max_denorm(
                        pred_val, stats[f"{key}_min"], stats[f"{key}_max"]
                    )
                    target_val = self.dataset.min_max_denorm(
                        target_val, stats[f"{key}_min"], stats[f"{key}_max"]
                    )

                    error = mse(pred_val, target_val)
                    self.logger.info(
                        f"Sample {i} - mse error of {key} (%): {error:.3f}"
                    )

                    polydata[f"pred_{key}"] = pred_val.detach().cpu().numpy()

            self.logger.info("-" * 50)
            os.makedirs(to_absolute_path(self.results_dir), exist_ok=True)
            polydata.save(
                os.path.join(to_absolute_path(self.results_dir), f"shell_graph_{i}.vtp")
            )


@hydra.main(version_base="1.3", config_path="conf/single_run_conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("main")  # General python logger
    logger.file_logging()

    logger.info("Rollout started...")
    rollout = MGNRollout(cfg, logger)
    rollout.predict()


if __name__ == "__main__":
    main()
