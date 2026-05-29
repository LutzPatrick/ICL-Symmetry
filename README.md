# Code and Checkpoints for Transformer Alignment Experiments

This directory contains the supplementary code and trained checkpoints for reproducing the main training and analysis workflows from the paper.

Paper link: **TODO: insert paper link**

## Contents

```text
code_paper/
├── train.ipynb          # standalone training notebook
├── analyze.ipynb        # analysis notebook for reproducing key figures
└── models/              # trained checkpoints used by analyze.ipynb
    ├── linear_classification/
    ├── semi_supervised/
    └── voronoicell/
```

Each task directory contains two model variants:

```text
models/{task}/
├── unconstrained/       # standard attention-only transformer checkpoints
└── symmetrized/         # transformers trained with the symmetrized attention update
```

Each run directory contains:

```text
config.yaml              # saved task/model/training configuration
state.pt                 # PyTorch checkpoint with model_state_dict and loss_log
```

The included checkpoints are organized as follows:

| Task | Unconstrained runs | Symmetrized runs |
|---|---:|---:|
| `linear_classification` | 5 | 5 |
| `semi_supervised` | 5 | 5 |
| `voronoicell` | 5 | 5 |

## Notebooks

### `train.ipynb`

A compact, standalone training notebook. It defines:

- the attention-only transformer architecture,
- the unconstrained and symmetrized attention variants,
- the synthetic task distributions,
- the optimizer and checkpointing loop,
- the run grid used for the paper checkpoints.

Running this notebook trains new models and writes them to `./models/{task}/{variant}/{run_id}`. The final cell keeps a record of the checkpoint paths produced by the original training run.

### `analyze.ipynb`

The analysis notebook loads the checkpoints in `./models/` and regenerates the key paper figures on transformer alignment and structure. For each task family it produces:

1. **Accuracy vs. context size:** transformer performance compared with task-appropriate classical baselines.
2. **Prediction alignment scatter:** mean true-label probability for unconstrained checkpoints vs. symmetrized checkpoints, with linear-fit \(R^2\).
3. **Weight-structure heatmaps:** representative attention weight matrices for one unconstrained and one symmetrized checkpoint.

The notebook discovers checkpoints automatically from the `models/` directory.

## How to Use

Run notebooks from inside `code_paper/` so that relative paths such as `./models` resolve correctly.

```bash
cd code_paper
jupyter lab
```

Then open either:

- `analyze.ipynb` to reproduce figures using the provided checkpoints,
- `train.ipynb` to train new checkpoints.

## Dependencies

The notebooks use standard scientific Python packages:

- Python 3
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- PyYAML
- tqdm
- Jupyter

A CUDA-capable GPU is recommended for `train.ipynb`. The analysis notebook can run on CPU, although some baseline evaluations fit a separate classical model per prompt and may take time.

## Reproducibility Notes

- Checkpoints are included, so the analysis figures can be regenerated without retraining.
- `config.yaml` files store the task, model, and training settings needed by the analysis notebook.
- `train.ipynb` writes new run directories rather than modifying existing checkpoints.
- The analysis notebook uses deterministic checkpoint discovery based on the directory structure above.
