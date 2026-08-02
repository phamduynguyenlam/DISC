# DISC

This document provides the commands needed to install, train, and evaluate DISC.

## Setup

Python 3.11 is recommended. Run all commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the appropriate PyTorch build separately first if a specific CUDA version is required.

Check the available command-line arguments:

```powershell
python trainer.py --help
python tester.py --help
```

## Training

The following example trains DISC with ZDT1 as the held-out problem:

```powershell
python trainer.py `
  --problem ZDT1 `
  --dim 30 `
  --seed 100000 `
  --epoch 50 `
  --reward_scheme 1 `
  --num_workers 24 `
  --device cuda
```

Common training arguments:

- `--seed`: the seed for this run.
- `--epoch`: the total number of training epochs.
- `--reward_scheme`: selects one of the following training rewards:
  - `1` — **Pareto improvement**.
  - `2` — **Normalized hypervolume improvement**.

For CPU-only execution, set `--device cpu`.

## Logs and checkpoints

Each training run creates a separate directory under:

```text
training_logs/disc/trainer_<problem>_set1_<timestamp>/
```

The directory contains the training log and checkpoints.

## Evaluation

The following example evaluates a trained DISC checkpoint:

```powershell
python tester.py `
  --problem ZDT1 `
  --dim 30 `
  --seed 100000 `
  --infill disc `
  --agent_pth "<checkpoint.pth>"
```

Evaluation logs are written to `testing_logs/<PROBLEM>/`. Figures are written to `png/<PROBLEM>/` and `pdf/<PROBLEM>/`.

## Quick verification

Compile the main files:

```powershell
python -m py_compile trainer.py tester.py agents\disc.py
```

Run a small end-to-end test with a randomly initialized model:

```powershell
python tester.py `
  --problem ZDT1 `
  --dim 5 `
  --seed 7 `
  --infill disc `
  --random_model
```

`--random_model` verifies the execution path only. Its result must not be used to assess DISC performance.
