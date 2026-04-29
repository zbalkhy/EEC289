# Go2 Course Homework (Readable Repo for Colab Download)

This repository is the readable homework package that a Google Colab notebook
can `git clone` and run.

The key idea is:
- keep the important source code visible as normal `.py` / `.json` files
- let the Colab notebook install dependencies and clone this repo on demand
- avoid the old large payload cell that hid the real homework code

## Colab usage model

If you already have a cloud notebook, it should download this repo with:

```python
COURSE_REPO_URL = "https://github.com/WeijieLai1024/EEC289A_Robotics-Homework.git"
```

## Important: Use your own GitHub repository in Colab

This assignment runs inside Google Colab, so you should **not** treat the
Colab runtime as permanent storage. Files under `/content/` can disappear when
the runtime restarts, disconnects, or times out. For that reason, the first
thing each student should do is create a personal GitHub copy of this homework:

1. either fork this repository, or create a new repository based on it
2. open the first notebook configuration cell
3. change `COURSE_REPO_URL` so it points to **your own repository**, not the
   course repository
4. keep `COURSE_REPO_BRANCH = "main"` if your repository uses `main`, or update
   that variable as well if you use a different branch name

For example:

```python
COURSE_REPO_URL = "https://github.com/<your-username>/<your-homework-repo>.git"
COURSE_REPO_BRANCH = "main"
COURSE_REPO_DIR = Path("/content/go2_course_repo")
```

From that point on, all development should happen inside the repository cloned
into `/content/go2_course_repo`, because that is the working copy used by the
notebook. After the setup cell finishes, the notebook already runs:

```python
%cd /content/go2_course_repo
```

so the files you inspect and edit there are the files that will actually be
used by `train.py`, `test_policy.py`, `generate_public_rollout.py`, and
`public_eval.py`.

### Why this matters

- If you leave `COURSE_REPO_URL` pointed at the course repository, your edits
  live only in the temporary Colab runtime and are easy to lose.
- The notebook's `ensure_course_repo(...)` helper **does not re-clone** if
  `/content/go2_course_repo` already exists. That means you should set your own
  `COURSE_REPO_URL` **before** running the setup cell.
- If you accidentally ran setup with the wrong repo URL first, the safest fix is
  to restart the Colab runtime, update `COURSE_REPO_URL`, and rerun the
  notebook from the top.

### Recommended student workflow

1. Create your own GitHub repository for the assignment.
2. In the notebook, replace `COURSE_REPO_URL` with your repository URL.
3. Run the setup cell so Colab clones **your** repository into
   `/content/go2_course_repo`.
4. Sanity-check that you are on the correct remote:

```bash
cd /content/go2_course_repo
git remote -v
```

The output should show your own GitHub repository, not the course repository.

5. Do all code changes inside `/content/go2_course_repo`.
6. Regularly save your work back to GitHub instead of relying on Colab to keep
   your files.

The basic save cycle is:

```bash
cd /content/go2_course_repo
git status
git add .
git commit -m "Describe your change"
git push origin main
```

If you prefer to work on a feature branch, replace `main` with your branch
name, but still push regularly. The important rule is:

- replace the repo URL with your own
- do all development in that cloned copy
- push your changes to GitHub often

In short: **Colab is a temporary execution environment, not your source-control
system. GitHub should be the place where your homework is stored safely and can
be reproduced later.**

The notebook then needs to:

1. clone pinned versions of `mujoco_playground` and `unitree_mujoco`
2. clone this repository into `/content/go2_course_repo`
3. install `configs/colab_requirements.txt`
4. install the cloned `mujoco_playground` checkout in editable mode
5. copy Go2 mesh assets into `go2_pg_env/xmls/assets/`
6. run `inspect_env.py`, `train.py`, `test_policy.py`, and `public_eval.py`

The included notebook template at `notebooks/go2_public_colab_template.ipynb`
has already been updated to use this public repo URL and the repo-side helper
scripts in this repository.

## Why this repo exists

The original homework notebook worked, but it hid most important files inside a
payload blob. That was fine for distribution, but it made the repository harder
to inspect and maintain. This repo keeps the same main pipeline while exposing
the code students are expected to read and modify.

## File map

```text
configs/course_config.json         Course-level knobs and benchmark settings
configs/colab_requirements.txt     Python dependencies the Colab notebook should install
course_common.py                   Shared utilities used by all scripts

go2_pg_env/
  __init__.py                      Register the local Go2 env
  constants.py                     Names and XML paths
  base.py                          Base Go2 environment wrapper
  joystick.py                      Main task logic: obs, action, reward, reset, step
  randomize.py                     Domain randomization
  xmls/                            MuJoCo XML files

scripts/copy_go2_assets.py         Copy Go2 meshes from unitree_mujoco into this repo layout

train.py                           Two-stage PPO training
test_policy.py                     Restore a checkpoint and render a deterministic demo
generate_public_rollout.py         Generate the standardized public benchmark rollout
public_eval.py                     Score a rollout bundle
inspect_env.py                     Print a compact environment summary
quick_policy_check.py              Tiny sanity-check script
benchmark_specs.py                 Deterministic command scripts
```

## Baseline scope

The default baseline in this repo is intentionally narrow:
- training samples only forward `vx` commands
- the public benchmark still probes forward, lateral, yaw-only, and combined motion

That makes the baseline easy to understand while leaving clear room for
students to extend the command space and measure the effect with the benchmark.

For the homework version, keep `stage_1` as the fixed forward-only baseline.
Students should use `stage_2` to extend command sampling and curriculum in
`go2_pg_env/joystick.py`, without changing the notebook entry points.

## Student modification boundary

Students should mostly modify:
- `go2_pg_env/joystick.py`
- `go2_pg_env/randomize.py`
- `configs/course_config.json`

Students should usually not modify:
- `public_eval.py`
- checkpoint restore logic
- rollout bundle field names

That keeps the benchmark comparable across submissions.
