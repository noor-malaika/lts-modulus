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

import numpy as np
import torch


try:
    import pyvista as pv
except Exception:
    raise ImportError(
        "Stokes Dataset requires the pyvista library. Install with "
        + "pip install pyvista"
    )
import h5py
from collections import defaultdict
import torch.nn.functional as F


def get_datapoint_idx(data_path):
    all_idx = []
    with h5py.File(data_path, "r") as data_file:
        for variant in data_file.keys():
            for subcase in data_file[variant]:
                all_idx.append((variant, subcase))
    return all_idx


def get_data_splits(idx):
    np.random.shuffle(idx)
    train_idx = idx[: int(0.7 * (len(idx)))]
    val_idx = idx[int(0.7 * (len(idx))) : int(0.85 * len(idx))]
    test_idx = idx[int(0.85 * len(idx)) :]
    return train_idx, val_idx, test_idx


def save_test_idx(idx):
    torch.save(idx, "test_idx.pt")


def load_test_idx(file="test_idx.pt"):
    idx = torch.load(file)
    return idx

def edges_to_triangles(edge_list):
    """
    Infer triangles from an edge list assuming the mesh is made of connected trias.

    Parameters:
    -----------
    edge_list : list of tuple(int, int)
        List of edges as pairs of node indices.

    Returns:
    --------
    triangles : list of tuple(int, int, int)
        List of triangles as node index triplets.
    """
    # Build neighbor map
    neighbors = defaultdict(set)
    for n1, n2 in edge_list:
        neighbors[n1].add(n2)
        neighbors[n2].add(n1)

    triangles = set()
    for n1, adj_nodes in neighbors.items():
        for n2 in adj_nodes:
            for n3 in neighbors[n2]:
                if n3 in neighbors[n1] and n1 < n2 < n3:  # Sort to avoid duplicates
                    triangles.add((n1, n2, n3))

    return list(triangles)


def convert_egdes_to_trias(triangles):
    faces = []
    for tri in triangles:
        faces.extend([3, *tri])
    faces = np.array(faces)
    return faces


def create_vtk_from_graph(data_dict):

    points = data_dict["pos"]
    edge_list = data_dict["connectivity"]

    triangles = edges_to_triangles(edge_list)
    faces = convert_egdes_to_trias(triangles)
    polydata = pv.PolyData(points, faces)

    polydata["disp_x"], polydata["disp_y"], polydata["disp_z"] = [
        data_dict["y"][:, i] for i in range(3)
    ]
    return polydata


def mse(pred, y, p=2):
    """
    Calculate relative L2 error norm

    Parameters:
    -----------
    pred: torch.Tensor
        Prediction
    y: torch.Tensor
        Ground truth

    Returns:
    --------
    error: float
        Calculated relative L2 error norm (percentage) on cpu
    """

    error = torch.mean(torch.norm(pred - y, p=p) / torch.norm(y, p=p)).cpu().numpy()
    return error * 100


def relative_rmse(pred, y):
    error = (
        torch.sqrt(torch.mean(torch.norm(pred - y, p=2) / torch.norm(y, p=2)))
        .cpu()
        .numpy()
    )
    return error * 100


def rmse(pred, y):
    """
    Compute the Root Mean Squared Error (RMSE) between predicted and true values.

    Args:
        pred (torch.Tensor): Predicted values.
        y (torch.Tensor): Ground truth values.

    Returns:
        float: RMSE value.
    """
    error = torch.sqrt(torch.mean((pred - y) ** 2))  # Standard RMSE
    return error.item() * 100  # Convert to scalar and scale by 100


def cosine_similarity(pred, true):
    """
    Compute the Cosine Similarity between predicted and true values.

    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Cosine Similarity value (ranges from -1 to 1).
    """
    # Flatten the tensors to compute cosine similarity
    pred_flat = pred.view(pred.size(0), -1)
    true_flat = true.view(true.size(0), -1)

    # Compute cosine similarity
    return F.cosine_similarity(pred_flat, true_flat).mean()


def combined_metric(pred, true, alpha=0.5):
    """
    Compute a combined metric that balances RMSE and Cosine Similarity.

    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): Ground truth values.
        alpha (float): Weight for RMSE (default: 0.5).

    Returns:
        torch.Tensor: Combined metric value.
    """
    rmse_value = rmse(pred, true)
    cos_sim_value = cosine_similarity(pred, true)

    # Combine RMSE and Cosine Similarity
    return alpha * rmse_value + (1 - alpha) * (1 - cos_sim_value)


def relative_mae(pred, true, eps=1e-8):
    return (
        (torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true) + eps)).item()
    ) * 100


def mrae(preds, trues, eps=1e-8):
    relative_errors = torch.abs(preds - trues) / (torch.abs(trues) + eps)
    return 100 * torch.mean(relative_errors)


def log_cosh(pred, true):
    log_cosh_loss = torch.mean(torch.log(torch.cosh(pred - true)))
    return log_cosh_loss * 100


####### need to change
