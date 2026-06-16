# Fork Tactical Combat RL

A small Gymnasium-based tactical combat reinforcement learning project. The environment simulates two teams fighting on a grid map, and the training code uses a hierarchical Commander + Squad policy structure.

## Project Files

- `env.py` - Tactical combat Gymnasium environment.
- `Commander+squad.py` - Hierarchical wrapper that builds Commander and Squad observations.
- `policy.py` - Policy networks, scripted behavior-cloning teacher, evaluation helpers, and PPO-style training loop.
- `train.py` - Main training entry point.
- `test0.py` - Evaluation/test entry point with optional video recording.

## Requirements

Recommended Python version: `3.10+`.

Install the required Python packages:

```bash
pip install numpy gymnasium torch matplotlib
```

If you enable video recording, also install video dependencies:

```bash
pip install moviepy imageio imageio-ffmpeg
```

On some Linux systems, `ffmpeg` may also be needed:

```bash
sudo apt-get install ffmpeg
```

## Quick Start

Clone or enter the project directory:

```bash
cd "/path/to/fork github"
```

Check that the Python files compile:

```bash
python -m py_compile env.py policy.py train.py test0.py "Commander+squad.py"
```

## Training

Start training with:

```bash
python train.py
```

The main training settings are near the top of `train.py`, including:

- `MAP_TYPE`
- `MAP_size`
- `TeamA_n`
- `stage_team_b`
- `stage_episode_add`
- evaluation frequency and save paths

Training outputs are saved under folders like:

```text
models_suburb_20x25/
videos_suburb/
train_log.csv
```

## Testing / Evaluation

Edit the test configuration near the top of `test0.py` before running:

```python
MODEL_DIR = "/path/to/model/stage_folder"
NUM_TEAM_B_TEST = 15
TEST_EPISODES = 20
RECORD_VIDEO = True
```

Then run:

```bash
python test0.py
```

If `RECORD_VIDEO = True`, videos are saved to the folder configured by `VIDEO_FOLDER`.

## Notes

- The current action space has 10 primitive actions: `0..9`.
- Grenade/smoke throwable actions were removed, so older checkpoints trained with a 12-action policy are not compatible with this version.
- If you only want to run without videos, set `record_video=False` in training or `RECORD_VIDEO = False` in `test0.py`.
