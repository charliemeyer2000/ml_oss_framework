# Hadi ML CS Framework

Simple training framework for CS 4774 HW1/3. The purpose of this is to provide you with all of the engineering upfront _so you don't have to_. That is: you focus on improving your model—learning about training/knowledge distillation, trying new ideas, tuning hyperparameters, tracking your experiemnts, and iterating to climb the leaderboard— and everything else is handled for you.

**TL/DR**: This should lessen engineering headaches training/submitting models and help you iterate faster.

## Features

- **Config management w/hydra**: YAML configs with CLI overrides
- **Knowledge distillation**: Student-teacher training with configurable temperature and alpha
- **Training visualization**: Loss/F1 curves saved as PNG
- **Sqlite Experiment Tracking**: Database logging of all runs w/sqlite
- **Robust server cubmission + CLI**: Auto-submit with retries and rate limit handling
- **Resume training**: Continue from checkpoints after interrupts
- **Class weighting**: Handle imbalanced datasets automatically
- **Gradient accumulation**: Train with larger effective batch sizes
- **Other goated hacks**: AMP, torch.compile(), async checkpointing, and aggressive type safety

## Setup

Clone the repo and activate the virtual environment (if you don't have `uv` installed yet, please install it from [here](https://docs.astral.sh/uv/getting-started/installation/))
```bash
cd ml_oss_framework
uv venv
source .venv/bin/activate
uv sync
```

Download and extract the training dataset:

```bash
mkdir -p data
# download it from the website and drag it into this directory
# then...
unzip training_dataset.zip -d data/
```

**IMPORTANT**: Update `configs/config.yaml` with your server token and username

anything interfacing with the server (server_cli.py, checking leaderboard, submitting, etc) only works while you're on the UVA VPN/on OOD/any device on UVA eduroam.

## Training

### Basic Training

```bash
uv run train.py
```

### Override Training Parameters

```bash
# change epochs and learning rate
uv run train.py training.epochs=50 training.lr=0.0003

# use fast config for quick testing (reads from config file)
uv run train.py training=fast

# custom data dir
uv run train.py --data-dir /path/to/data

# advanced hacks like torch.compile()
uv run train.py training.use_compile=true

# resume from checkpoint
uv run train.py training.resume_from=outputs/run_xyz/latest_checkpoint.pth

# combine multiple overrides
uv run train.py training.epochs=100 training.lr=0.0001 data.batch_size=64
```

## Knowledge Distillation

### With Local Teacher

```bash
uv run train_distillation.py teacher.checkpoint=teacher.pt
```

### With HuggingFace Model

Set your HF token via one of these methods:

1. **In config** (recommended): Set `hf_token` in `configs/config_distillation.yaml`
2. **CLI override**: `hf_token=YOUR_TOKEN`
3. **env var**: `export HF_TOKEN=YOUR_TOKEN`
4. **HF cli**: `uv run huggingface-cli login`

```bash
# if token is in config or env var:
uv run train_distillation.py teacher.hf_model=google/medsiglip-448

# or pass token directly:
uv run train_distillation.py teacher.hf_model=google/medsiglip-448 hf_token=YOUR_TOKEN
```

Get your HF token at: https://huggingface.co/settings/tokens

### Distillation Parameters

```bash
uv run train_distillation.py teacher.checkpoint=t.pt distillation.temperature=6 distillation.alpha=0.1
```

- `temperature`: Softness of distributions (2-10). Higher = softer.
- `alpha`: Hard loss weight. Lower = more teacher influence.
- add more as you'd like!!!

## Server CLI

If you configured `server.token` and `server.username` in `configs/config.yaml`, you don't need to pass them on the command line.

```bash
# submit a model
uv run server_cli.py submit outputs/[run]/model.pt

# submit and wait for evaluation
uv run server_cli.py submit model.pt --wait

# check status
uv run server_cli.py status

# view leaderboard
uv run server_cli.py leaderboard
uv run server_cli.py leaderboard --top-n 50

# get your current rank
uv run server_cli.py rank

# wait for pending submission and sync to database
uv run server_cli.py wait-and-sync
```

If credentials aren't in config, pass them via CLI:

```bash
uv run server_cli.py --token YOUR_TOKEN --username YOUR_NAME leaderboard
```

## Full Pipeline (`run_experiment.py`)

Why `run_experiment.py`?
- `train.py` / `train_distillation.py`: Just trains the model locally
- `run_experiment.py`: Full pipeline - trains → submits to server → waits for eval → syncs results to DB (the whole shebang!)

```bash
# full pipeline: train + submit + sync
uv run run_experiment.py

# train only (no submission)
uv run run_experiment.py --skip-submit

# quick iteration with fast config
uv run run_experiment.py training=fast

# combine overrides example
uv run run_experiment.py training.epochs=30 training.lr=0.0005 --run-name exp_v2
```

## Outputs

```
outputs/{run_name}/
├── config.yaml          # Config used for this run
├── training.log         # Training logs
├── best_model.pth       # Best checkpoint (PyTorch)
├── model.pt             # TorchScript model (submit this)
└── training_curves.png  # Loss/F1 plots
```

Experiment data is also logged to `experiments.db` (sqlite).

## Config Structure

```
configs/
├── config.yaml (for train.py)
├── config_distillation.yaml (for train_distillation.py)
├── model/
│   └── demo_cnn.yaml
├── training/
│   ├── default.yaml
│   └── fast.yaml
└── distillation/
    ├── default.yaml         # T=4, alpha=0.3
    └── soft.yaml            # T=6, alpha=0.1
```

Override anything via CLI: `training.epochs=100 training.lr=0.0001`

## License

MIT

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)
