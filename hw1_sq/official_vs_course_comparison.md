# Official MuJoCo Playground notebook vs our course code

## One-line summary

The official locomotion notebook is a **broad Playground reference workflow**.
Our course code is a **Go2-focused assignment pipeline**.

## Main differences

### 1. Environment source

**Official notebook**
- Uses a built-in Playground environment such as `Go1JoystickFlatTerrain`
- Loads it directly from `registry`

**Course code**
- Registers a local `Go2JoystickFlatTerrain`
- Reuses the official Go1 joystick task structure
- Swaps in a local Go2 XML and assets

### 2. Installation style

**Official notebook**
- Installs dependencies directly in Colab
- Installs `playground` and uses built-in environments

**Course code**
- Pins MuJoCo Playground and Unitree MuJoCo commits
- Copies Go2 assets locally
- Keeps the task definition inside the course repo

### 3. Training budget

**Official tuned Go1 joystick PPO config**
- Much larger training budget
- Much larger environment count
- More suitable for research baselines

**Course code**
- Uses a smaller Colab-friendly two-stage setup
- Fits a fairness constraint for students using ordinary Colab GPUs

### 4. Notebook vs script boundary

**Official notebook**
- Training happens directly inside notebook cells
- Great for a quick orientation pass

**Course code**
- Uses explicit scripts:
  - `train.py`
  - `test_policy.py`
  - `generate_public_rollout.py`
  - `public_eval.py`
- Better for a reproducible script-first pipeline

### 5. Benchmarking

**Official notebook**
- Focuses on training and rendering policies

**Course code**
- Adds a standardized public benchmark
- Adds a deterministic rollout bundle format
- Adds a fixed evaluator for grading

### 6. Repository readability

**Original course notebook**
- Hid the local code inside a large payload blob

**Readable course version**
- Exposes the real source files
- Makes it much easier to inspect each module in a structured walkthrough

## What stayed intentionally close to the official Go1 joystick task

To keep the assignment grounded in a strong public baseline, the local Go2 task
stays close to the official Go1 joystick design in:
- observation structure
- actor / critic split
- reward families
- command-tracking objective
- domain randomization style

This is deliberate: students should learn a realistic robot-learning pipeline,
not a toy environment unrelated to current practice.
