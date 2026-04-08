# ShellMGN

A modular, package-based **MeshGraphNets** project for shell mechanics, built on top of NVIDIA Modulus, DGL, PyTorch, and Hydra, to predict structural displacements in static simulations.

This repository has been refactored from a script-first layout to a package-first layout so training, inference, losses, dataloading, and preprocessing are easier to maintain, test, and extend.

---

## What this project does

ShellMGN trains a **GNN** - (`MeshGraphNet`) to predict displacement components from mesh-based input data for static structural analysis.

This implementation is aligned with:
- the MeshGraphNets paper: **Pfaff et al., “Learning Mesh-Based Simulation with Graph Networks”** (arXiv:2010.03409) — https://arxiv.org/abs/2010.03409
- the NVIDIA Modulus MeshGraphNet implementation and ecosystem — https://docs.nvidia.com/modulus/

Core capabilities:
- **Hydra-driven training** with configurable losses and normalization.
- **Hydra-driven inference/rollout** for checkpoint-based evaluation and VTK output export.
- **Modular package structure** under `src/shellmgn`.
- **Separated concerns** for model, trainer, dataloader, losses, and utilities.
- **WandB integration** for experiment tracking.

---

## Preprocessing pipeline

The preprocessing pipeline is designed to take in the raw solver files (i.e., Optistruct/Nastran) and extract the helpful features and store them in HDF5 format. The hdf5 dataset is then transformed to a DGL graph dataset that's accepted by the MeshGraphNet model.

Each sample included keys used by `ShellDataset`:
- `connectivity`: edge list / mesh connectivity
- `pos`: node coordinates
- `ntypes`: node type markers
- `thickness`: shell thickness per node
- `spc`: support / constraint features
- `load`: applied load features
- `etypes`: edge type markers
- `y`: target displacement components, shaped like `[num_nodes, 3]` for `(disp_x, disp_y, disp_z)`

Current dataset split logic expects top-level HDF5 grouping like:
- `variant/subcase/...sample arrays...`

See:
- `src/shellmgn/dataloader/dataloader.py`
- `src/shellmgn/utils.py`

for the exact assumptions used during graph construction, normalization, and splitting.

*Note: This preprocessing pipeline is highly specific to Optistruct/Nastran format(s) but the logic can still be adapted to other data formats.*

---

## Repository layout

```text
lts-modulus/
├── pyproject.toml
├── uv.lock
├── src/
│   ├── shellmgn/
│   │   ├── main.py                    # training entrypoint (Hydra)
│   │   ├── conf/
│   │   │   ├── single_run_conf/
│   │   │   │   └── config.yaml        # default train/infer config
│   │   │   └── multi_comp/
│   │   │       ├── config.yaml        # multi-component training config
│   │   │       └── sweep.yaml         # sweep config
│   │   ├── dataloader/
│   │   │   └── dataloader.py          # HDF5 + DGL dataset pipeline
│   │   ├── models/
│   │   │   └── meshgraphnet.py        # model definition
│   │   ├── trainer/
│   │   │   └── trainer.py             # training loop and validation
│   │   ├── inference/
│   │   │   └── inference.py           # rollout/prediction entrypoint
│   │   ├── losses/
│   │   │   ├── logcosh.py
│   │   │   ├── mrae.py
│   │   │   ├── multi_comp.py
│   │   │   ├── multi_comp_uncertain.py
│   │   │   └── weighted_logcosh.py
│   │   └── utils.py                   # metrics + split/test-index helpers
│   └── preprocess/
│       ├── main.py                    # preprocessing CLI entrypoint
│       └── ...
└── .venv/
```

---

## Build and packaging

This project uses:
- **PEP 517 backend**: `hatchling`
- **Dependency manager / resolver**: `uv`
- **Package source root**: `src/`

Configured scripts in `pyproject.toml`:
- `shellmgn-train` → `shellmgn.main:main`
- `shellmgn-infer` → `shellmgn.inference.inference:main`
- `preprocess` → `preprocess.main:main`

---

## Environment setup

## 1) Create & activate virtual env

```bash
uv venv
source .venv/bin/activate
```

## 2) Install dependencies

```bash
uv sync
```

---

## CUDA / DGL / PyTorch notes

This project uses GPU-oriented dependencies (`torch`, `dgl`, `nvidia-modulus`).

`pyproject.toml` is configured with:
- a `find-links` source for DGL CUDA wheels
- a dedicated PyTorch CUDA index (`cu124`) for `torch`

If you still hit platform-specific wheel issues, install the problematic package first, then re-run sync. Example:

```bash
uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
uv sync
```

---

## Configuration

Active Hydra configs live in:
- `src/shellmgn/conf/single_run_conf/config.yaml`
- `src/shellmgn/conf/multi_comp/config.yaml`
- `src/shellmgn/conf/multi_comp/sweep.yaml`

Important paths in config:
- `data_path`: expected HDF5 dataset path (currently `./data/shellmgn/dataset.hdf5`)
- `ckpt_path`: checkpoint directory
- `results_dir`: output directory
- `test_idx`: persisted test index file path for train→infer consistency

---

## Training

Default training (uses Hydra defaults in `single_run_conf/config.yaml`):

```bash
uv run shellmgn-train
```

Run with Hydra overrides:

```bash
uv run shellmgn-train loss=MSELoss normalization=z_score epochs=100
```

Example with explicit output/checkpoint override:

```bash
uv run shellmgn-train ckpt_path=./runs/shellmgn/exp1/checkpoints results_dir=./runs/shellmgn/exp1/results
```

---

## Inference / rollout

Run inference with current defaults:

```bash
uv run shellmgn-infer
```

With overrides:

```bash
uv run shellmgn-infer ckpt_path=./runs/shellmgn/exp1/checkpoints results_dir=./runs/shellmgn/exp1/results test_idx=./runs/shellmgn/exp1/outputs/test_idx.pt
```

Inference exports `.vtp` graph files to `results_dir`.

---

## Preprocessing

The preprocessing CLI entrypoint is:

```bash
uv run preprocess generate-dataset --base-dir <raw_data_dir> --hdf5-name <output_name> --log-name <log_name>
```

Use this to generate the HDF5 dataset consumed by `data_path` in Hydra configs.

---

## Experiment tracking (WandB)

Training integrates Weights & Biases. Set your API key before run:

```bash
export WANDB_API_KEY=<your_key>
```

Then run training normally. Metrics such as training loss, LR, and validation errors are logged.

---

## License

This project is licensed under the **Apache License 2.0**.

- You are free to use, modify, and share the code.
- Keep license and attribution notices when redistributing.

See [LICENSE](LICENSE) for the full terms.

### Data note

This repository is intended to contain code and configuration. If you use private/proprietary simulation data, ensure you follow your organization’s data-sharing and confidentiality policies before distributing datasets.


