# from utils import save_test_idx, load_test_idx
# from datapipe.shell_dataset import Hdf5Dataset
# import pyvista as pv
# import numpy as np
# from collections import defaultdict

# def edges_to_triangles(edge_list):
#     """
#     Infer triangles from an edge list assuming the mesh is made of connected trias.

#     Parameters:
#     -----------
#     edge_list : list of tuple(int, int)
#         List of edges as pairs of node indices.

#     Returns:
#     --------
#     triangles : list of tuple(int, int, int)
#         List of triangles as node index triplets.
#     """
#     # Build neighbor map
#     neighbors = defaultdict(set)
#     for n1, n2 in edge_list:
#         neighbors[n1].add(n2)
#         neighbors[n2].add(n1)

#     triangles = set()
#     for n1, adj_nodes in neighbors.items():
#         for n2 in adj_nodes:
#             for n3 in neighbors[n2]:
#                 if n3 in neighbors[n1] and n1 < n2 < n3:  # Sort to avoid duplicates
#                     triangles.add((n1, n2, n3))

#     return list(triangles)

# # idx = [
# #     ('var-54', '90'),
# #     ('var-78', '56')
# # ]

# # save_test_idx(idx)

# # print(load_test_idx())

# test_idx = load_test_idx("/home/sces213/Malaika/lts_modulus/shell_mgn/outputs/test_idx.pt")
# print(test_idx[0])

# test_hdf5 = Hdf5Dataset("/home/sces213/Malaika/lts_modulus/shell_mgn/dataset/dataset.hdf5", test_idx, len(test_idx))
# data_1 = test_hdf5[0]

# def create_vtk_from_graph(data_dict):

#     points = data_dict["pos"]
#     edge_list = data_dict["connectivity"]

#     triangles = edges_to_triangles(edge_list)
#     faces = convert_trias_to_faces(triangles)
#     polydata = pv.PolyData(points, faces)

#     polydata["disp_x"] ,polydata["disp_y"], polydata["disp_z"] = [data_dict["y"][:, i] for i in range(3)]
#     return polydata

# def convert_trias_to_faces(triangles):
#     # Flatten edge list to VTK's format for lines
#     faces = []
#     for tri in triangles:
#         faces.extend([3, *tri])
#     faces = np.array(faces)
#     return faces


# polydata = create_vtk_from_graph(data_1)
# polydata.save("temp1.vtp")


# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# # Small range of values for pred
# pred = torch.tensor(np.linspace(-1, 1, 100), dtype=torch.float32)
# actual = torch.zeros_like(pred)

# # Compute log(cosh(pred - actual))
# loss = torch.log(torch.cosh(pred - actual))

# # Plot the loss
# plt.figure(figsize=(8, 6))
# plt.plot(pred.numpy(), loss.numpy(), label='log(cosh(pred - actual))')
# plt.xlabel('Prediction (pred)')
# plt.ylabel('Loss')
# plt.title('log(cosh) Loss for Small Predictions vs Actuals')
# plt.legend()
# plt.grid(True)

# # Save the figure to a file
# plt.savefig('log_cosh_loss_plot.png')  # Change the filename as needed

# # Optionally, show the plot
# # plt.show()

# import torch
# import torch.nn as nn

# class LogCoshLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, preds, trues):
#         return torch.mean(torch.log(torch.cosh(preds - trues)))

# class MeanSquaredError(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, preds, trues):
#         return torch.mean((preds - trues) ** 2)

# class MeanRelativeAbsoluteError(nn.Module):
#     def __init__(self, eps=1e-8):
#         super().__init__()
#         self.eps = eps

#     def forward(self, preds, trues):
#         relative_errors = torch.abs(preds - trues) / (torch.abs(trues) + self.eps)
#         return 100 * torch.mean(relative_errors)  # Convert to percentage

# # Normalization
# def max_abs_normalize(data):
#     max_abs = torch.max(torch.abs(data))
#     return data / max_abs

# # Example usage
# y_true = torch.tensor([[-6.9, -1.0, 0.5],
#                        [-0.5, 2.0, 0.0]])
# y_pred = torch.tensor([[-7.0, 1.0, 0.6],
#                        [0.5, 3.4, 0.0]])
# '''
# mrae has better percentage than mse
# '''
# # Normalize displacements
# y_true_normalized = max_abs_normalize(y_true)
# y_pred_normalized = max_abs_normalize(y_pred)

# # Loss and metric
# loss_fn = LogCoshLoss()  # or MeanSquaredError()
# metric_fn = MeanRelativeAbsoluteError()

# loss = loss_fn(y_pred_normalized, y_true_normalized)
# metric = metric_fn(y_pred_normalized, y_true_normalized)
# print(f'true {y_true_normalized}\npred {y_pred_normalized}')
# print(f"Loss: {loss.item()}, mrae: {metric.item()}%")
# from utils import relative_lp_error
# print(relative_lp_error(y_pred_normalized, y_true_normalized))

# import wandb

# # Initialize the W&B API
# api = wandb.Api()
# print('---')
# # Specify the sweep ID
# sweep_id = "cxis0hll"  # Replace with your sweep ID
# print('---')
# # Fetch the sweep
# sweep = api.sweep(f"malaikanoor7864-mnsuam/lts_modulus-shell_mgn/cxis0hll")
# print(f'sweep {sweep}---')
# # Print all possible configurations
# for run in sweep.runs:
#     print(run.config)
# print('---')
# print("Sweep Parameter Space:")
# print(sweep.config)


# import yaml
# from itertools import product

# # Load sweep configuration
# with open("shell_mgn/conf/sweep.yaml", "r") as f:
#     sweep_config = yaml.safe_load(f)

# # Extract parameter values from the sweep configuration
# parameters = sweep_config.get("parameters", {})

# loss_values = parameters.get("loss", {}).get("values", [])
# normalization_values = parameters.get("normalization", {}).get("values", [])
# metric_values = parameters.get("metric", {}).get("values", [])

# # Generate all possible combinations
# combinations = list(product(loss_values, normalization_values, metric_values))

# # Print all combinations
# for idx, (loss, norm, metric) in enumerate(combinations, start=1):
#     print(f"Combination {idx}: Loss: {loss}, Normalization: {norm}, Metric: {metric}")

# print(f"\nTotal combinations: {len(combinations)}")

import wandb

# Initialize a W&B run
wandb.init(project="shell_mgn_sweep_v1", entity="malaikanoor7864-mnsuam")

# Log a dummy metric
wandb.log({"accuracy": 0.95})

# Finish the run
wandb.finish()
