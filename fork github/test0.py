
import os
from gymnasium.wrappers import RecordVideo
import glob
import torch
import numpy as np
from torch.distributions import Categorical

from env import TacticalCombatEnv, get_core_env
from policy import (
    CommanderPolicy,
    HierarchicalWrapper,
    OpponentControllerNN,
    SquadPolicy,
    apply_commander_logits_prior,
    apply_squad_tactical_logits,
    build_teamA_presampling_action_mask,
    set_global_seeds,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 测试配置 =====
TEST_SEED = 1212
set_global_seeds(TEST_SEED)

MODEL_DIR = "/home/yq/Desktop/fork game/models_suburb_20x25（best）/stage_03_B20"
PREFER_BEST = True              # True 优先 best.pth；否则优先 latest.pth
NUM_TEAM_B_TEST = 15           # B队人数可调整
TEST_EPISODES = 20
USE_NN_FOR_B = False

# ===== 新增：测试视频 =====
RECORD_VIDEO = True
VIDEO_FOLDER = "./test_videos_20"
VIDEO_FREQ = 2          # 每隔多少个 episode 录一次；1 = 每局都录
VIDEO_PREFIX = "rl_test_20"
# =======================

# 评估模式: "greedy" / "sample" / "temp"
EVAL_MODE = "sample"
TEMP = 0.7                      # 仅 EVAL_MODE="temp" 时生效
FIXED_SEED_BASE = TEST_SEED     # 让测试更可复现
# ===================


def pick_checkpoint(model_dir="/kaggle/working/models_suburb_32x32/stage_03_B20", prefer_best=True):
    best_path = os.path.join(model_dir, "best.pth")
    latest_path = os.path.join(model_dir, "latest.pth")

    if prefer_best and os.path.exists(best_path):
        return best_path
    if os.path.exists(latest_path):
        return latest_path

    cand = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
    if not cand:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    return cand[-1]


def select_actions_from_logits(logits: torch.Tensor, mode="greedy", temperature=0.7):
    """
    logits:
      Commander: (num_squads, num_commands)
      Squad:     (squad_size, num_actions)

    return:
      LongTensor with shape matching logits[:-1]
    """
    if mode == "greedy":
        return torch.argmax(logits, dim=-1)

    if mode == "sample":
        dist = Categorical(logits=logits)
        return dist.sample()

    if mode == "temp":
        scaled_logits = logits / max(temperature, 1e-6)
        dist = Categorical(logits=scaled_logits)
        return dist.sample()

    raise ValueError(f"Unknown EVAL_MODE: {mode}")


def select_actions_like_training(logits: torch.Tensor, mode="sample", temperature=0.7):
    if mode == "greedy":
        return torch.argmax(logits, dim=-1)

    if mode == "sample":
        dist = Categorical(logits=logits)
        return dist.sample()

    if mode == "temp":
        scaled_logits = logits / max(temperature, 1e-6)
        dist = Categorical(logits=scaled_logits)
        return dist.sample()

    raise ValueError(f"Unknown EVAL_MODE: {mode}")


CKPT_PATH = pick_checkpoint(MODEL_DIR, prefer_best=PREFER_BEST)
print(f"[TEST] Using checkpoint: {CKPT_PATH}")

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

MAP_TYPE = ckpt.get("map_type", "suburb")
GRID_SIZE = tuple(ckpt.get("grid_size", (20, 20)))
NUM_TEAM_A = ckpt.get("num_team_a", 20)
SQUAD_SIZE = ckpt.get("squad_size", 5)

base_env = TacticalCombatEnv(
    grid_size=GRID_SIZE,
    num_team_a=NUM_TEAM_A,
    num_team_b=NUM_TEAM_B_TEST,
    max_steps=450,
    map_type=MAP_TYPE,
    render_mode="rgb_array",   # ===== 改动：为了录视频 =====
    seed=TEST_SEED,
)
# ===== 新增：测试时录视频 =====
if RECORD_VIDEO:
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    base_env = RecordVideo(
        base_env,
        video_folder=VIDEO_FOLDER,
        episode_trigger=lambda ep: ((ep + 1) % VIDEO_FREQ == 0),
        name_prefix=VIDEO_PREFIX,
    )
    print(f"[RecordVideo] Test video enabled. Saving to {VIDEO_FOLDER}")
# ===============================

hier_env = HierarchicalWrapper(base_env, squad_size=SQUAD_SIZE)

num_squads = hier_env.num_squads
feat_dim_c = hier_env.feat_dim_c
feat_dim_s = hier_env.feat_dim_s
num_commands = hier_env.num_commands
map_size = hier_env.grid_h * hier_env.grid_w

commander = CommanderPolicy(num_squads, feat_dim_c, num_commands, map_size).to(DEVICE)
squad = SquadPolicy(feat_dim_s, hier_env.squad_size, 10).to(DEVICE)

commander.load_state_dict(ckpt["commander"])
squad.load_state_dict(ckpt["squad"])

commander.eval()
squad.eval()

if USE_NN_FOR_B:
    core_env = get_core_env(base_env)
    opponent_ctrl = OpponentControllerNN(
        env=core_env,
        commander_model=commander,
        squad_model=squad,
        squad_size=hier_env.squad_size,
        feat_dim_c=feat_dim_c,
        feat_dim_s=feat_dim_s,
        num_commands=num_commands,
        map_size=map_size,
        device=DEVICE,
    )
    core_env.nn_controller_B = opponent_ctrl.act

episode_returns = []
episode_results = []

with torch.no_grad():
    for ep in range(TEST_EPISODES):
        seed = FIXED_SEED_BASE + ep
        set_global_seeds(seed)
        hier_obs = hier_env.reset(seed=seed)
        done = False
        ep_return = 0.0
        steps = 0
        info = {}

        while not done:
            # Commander
            obs_C = torch.from_numpy(hier_obs.commander_obs.flatten()).float().to(DEVICE)
            map_np = hier_obs.known_map.astype(np.float32) / 2.0
            map_flat = torch.from_numpy(map_np.flatten()).float().to(DEVICE)
            commander_input = torch.cat([obs_C, map_flat], dim=0)

            logits_C_all = commander(commander_input)
            logits_C = apply_commander_logits_prior(logits_C_all[0])
            cmd_actions = select_actions_like_training(
                logits_C, mode=EVAL_MODE, temperature=TEMP
            )

            # 关键修复：把当前步 commander 命令注入 squad obs
            hier_obs_for_squad = hier_env._build_hier_obs(
                commands=cmd_actions.detach().cpu().numpy()
            )


            # Squad
            squad_action_list = []
            for s_idx in range(num_squads):
                obs_S = torch.from_numpy(hier_obs_for_squad.squad_obs_list[s_idx]).float().to(DEVICE)
                logits_S_all = squad(obs_S)
                logits_S = apply_squad_tactical_logits(logits_S_all[0], obs_S)
                action_mask_np = build_teamA_presampling_action_mask(hier_env.env, hier_env.squads[s_idx])
                action_mask_t = torch.from_numpy(action_mask_np).to(device=DEVICE, dtype=logits_S.dtype)
                logits_S = logits_S + action_mask_t
                a_squad = select_actions_like_training(
                    logits_S, mode=EVAL_MODE, temperature=TEMP
                )
                squad_action_list.append(a_squad.cpu().numpy())

            hier_obs, r, done, info = hier_env.step(
                commander_action=cmd_actions.cpu().numpy(),
                squad_actions=squad_action_list,
            )

            ep_return += float(r)
            steps += 1

        core_env = get_core_env(base_env)
        a_alive = int(core_env.alive_mask[:core_env.num_team_a].sum())
        b_alive = int(core_env.alive_mask[core_env.num_team_a:].sum())
        result = info.get("result", "")

        episode_returns.append(ep_return)
        episode_results.append(result)

        print(
            f"[TEST] Episode {ep+1:02d}/{TEST_EPISODES} | "
            f"mode={EVAL_MODE} | temp={TEMP:.2f} | "
            f"return={ep_return:.2f} | steps={steps} | "
            f"A_alive={a_alive} | B_alive={b_alive} | result={result}"
        )

avg_return = float(np.mean(episode_returns))
print(f"[TEST] avg_return = {avg_return:.2f}")

# 可选：统计胜负
win_like = {"win", "team_b_eliminated", "all_enemies_cleared"}
wins = sum(r in win_like for r in episode_results)
print(f"[TEST] win_like_rate = {wins / max(1, len(episode_results)):.3f}")

hier_env.env.close()
