import glob
import importlib.util
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordVideo
from torch.distributions import Categorical

from env import COVER, TEAM_B, TacticalCombatEnv, get_core_env

_commander_squad_path = os.path.join(os.path.dirname(__file__), "Commander+squad.py")
_commander_squad_spec = importlib.util.spec_from_file_location(
    "commander_squad", _commander_squad_path
)
_commander_squad_module = importlib.util.module_from_spec(_commander_squad_spec)
sys.modules[_commander_squad_spec.name] = _commander_squad_module
_commander_squad_spec.loader.exec_module(_commander_squad_module)
HierarchicalWrapper = _commander_squad_module.HierarchicalWrapper


def set_global_seeds(seed: int, deterministic_torch: bool = True):
    """Seed Python, NumPy and PyTorch from one project-level seed."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ======================
# 3. Policy 网络
# ======================


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor):
        return self.activation(x + self.block(x))


def init_policy_module(module: nn.Module, output_gain: float = 0.01):
    linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    for layer in linear_layers:
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(layer.bias)
    if linear_layers:
        nn.init.orthogonal_(linear_layers[-1].weight, gain=output_gain)
        nn.init.zeros_(linear_layers[-1].bias)


MEMBER_FEATURE_BASE = 15
MEMBER_SHOOT_FEAT_DIM = 6
SHOOT_LOGIT_BONUS = 0.9
VISIBLE_SHOOT_LOGIT_BONUS = 0.2
COMMAND_ADVANCE_BONUS = 0.35
COMMAND_CAUTIOUS_PENALTY = 0.15
NUM_SQUAD_ACTIONS = 10


def apply_squad_tactical_logits(logits: torch.Tensor, squad_obs: torch.Tensor) -> torch.Tensor:
    adjusted = logits.clone()
    if squad_obs.dim() == 1:
        obs = squad_obs.unsqueeze(0)
    else:
        obs = squad_obs

    if adjusted.dim() == 2:
        adjusted = adjusted.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    squad_size = adjusted.size(-2)
    feat_dim = obs.size(-1)
    required = MEMBER_FEATURE_BASE + squad_size * MEMBER_SHOOT_FEAT_DIM
    if feat_dim >= required:
        can_see = []
        can_shoot = []
        for k in range(squad_size):
            off = MEMBER_FEATURE_BASE + k * MEMBER_SHOOT_FEAT_DIM
            can_see.append(obs[..., off + 1])
            can_shoot.append(obs[..., off + 2])
        can_see_t = torch.stack(can_see, dim=-1).to(dtype=adjusted.dtype, device=adjusted.device)
        can_shoot_t = torch.stack(can_shoot, dim=-1).to(dtype=adjusted.dtype, device=adjusted.device)
        visible_not_shootable = torch.clamp(can_see_t - can_shoot_t, min=0.0, max=1.0)

        adjusted[..., 9] = (
            adjusted[..., 9]
            + SHOOT_LOGIT_BONUS * can_shoot_t
            + VISIBLE_SHOOT_LOGIT_BONUS * visible_not_shootable
        )
        adjusted[..., 9] = torch.where(
            can_shoot_t > 0.5,
            adjusted[..., 9],
            torch.full_like(adjusted[..., 9], -1e9),
        )
        adjusted[..., 0] = torch.where(
            can_shoot_t > 0.5,
            torch.full_like(adjusted[..., 0], -1e9),
            adjusted[..., 0],
        )

    if squeeze_back:
        adjusted = adjusted.squeeze(0)
    return adjusted


def build_teamA_presampling_action_mask(env: TacticalCombatEnv, squad_indices) -> np.ndarray:
    """
    只屏蔽明显错误动作，不替 A 队选择“正确动作”。
    返回 additive mask：0 保留，-1e9 表示屏蔽。
    """
    base_env = get_core_env(env)
    squad_size = len(squad_indices)
    mask = np.zeros((squad_size, NUM_SQUAD_ACTIONS), dtype=np.float32)

    support_targets_A = getattr(base_env, "current_support_target_A", None)
    support_modes_A = getattr(base_env, "current_support_mode_A", None)

    for k, agent_idx in enumerate(squad_indices):
        if agent_idx < 0 or agent_idx >= base_env.num_team_a:
            continue
        if not base_env.alive_mask[agent_idx]:
            mask[k, :] = -1e9
            mask[k, 0] = 0.0
            continue

        if getattr(base_env, "suppressed", None) is not None and base_env.suppressed[agent_idx]:
            mask[k, 3:8] = -1e9

        if (
            getattr(base_env, "agent_reload_timer", None) is not None
            and base_env.agent_reload_timer[agent_idx] > 0
        ):
            mask[k, 9] = -1e9

        if (
            getattr(base_env, "agent_ammo", None) is not None
            and base_env.agent_ammo[agent_idx] <= 0
        ):
            mask[k, 9] = -1e9

        can_see = base_env._agent_can_see_enemy(agent_idx)
        can_shoot = base_env._has_shootable_enemy(agent_idx)

        if can_shoot:
            fire_tgt = base_env._teamA_filter_target(agent_idx)
            if fire_tgt is not None:
                face_score = base_env._facing_score_towards(agent_idx, fire_tgt[0], fire_tgt[1])
                if face_score >= 0.70:
                    # 可以开火且枪口已基本对准时，先只屏蔽继续前冲，
                    # 保留横移/后退/转向这些仍可能有战术价值的动作。
                    mask[k, 3] = -1e9
                    mask[k, 7] = -1e9

        # 进入支援近区后，只屏蔽会让朝向明显变差的那个转向。
        if (
            (not can_see)
            and (not can_shoot)
            and support_targets_A is not None
            and agent_idx < len(support_targets_A)
            and support_targets_A[agent_idx] is not None
        ):
            ty, tx = support_targets_A[agent_idx]
            y, x = base_env.agents_pos[agent_idx]
            support_mode = int(support_modes_A[agent_idx]) if support_modes_A is not None else 0
            stop_dist = 2 if support_mode in (1, 3) else 3
            d_to_support = abs(y - ty) + abs(x - tx)
            if d_to_support <= stop_dist:
                face_tgt = base_env._teamA_filter_target(agent_idx)
                if face_tgt is not None:
                    ori = int(base_env.agents_orient[agent_idx])
                    cur_score = base_env._facing_score_with_ori(agent_idx, ori, face_tgt[0], face_tgt[1])
                    for turn_act, new_ori in ((1, (ori - 1) % 8), (2, (ori + 1) % 8)):
                        new_score = base_env._facing_score_with_ori(
                            agent_idx, new_ori, face_tgt[0], face_tgt[1]
                        )
                        if new_score + 0.24 < cur_score:
                            mask[k, turn_act] = -1e9

    return mask


def apply_commander_logits_prior(logits: torch.Tensor) -> torch.Tensor:
    adjusted = logits.clone()
    adjusted[..., 0] = adjusted[..., 0] + COMMAND_ADVANCE_BONUS
    adjusted[..., 5] = adjusted[..., 5] - COMMAND_CAUTIOUS_PENALTY
    return adjusted


class OpponentControllerNN:
    """
    使用同一套 CommanderPolicy + SquadPolicy 控制 B 队
    - CommanderPolicy 的结构是按 A 队的小队数 num_squads_A 固定的
    - 如果 B 队小队数 != num_squads_A，则把 B 队的小队“聚合”成 num_squads_A 个槽位
      （例如 8 个小队聚成 4 组，每组取平均中心、平均 alive_ratio 等）
    - Commander 输出 num_squads_A 条命令，我们循环复用到所有 B 队小队
    """

    def __init__(
            self,
            env: TacticalCombatEnv,
            commander_model: nn.Module,
            squad_model: nn.Module,
            squad_size: int,
            feat_dim_c: int,
            feat_dim_s: int,
            num_commands: int,
            map_size: int,
            device=None,
    ):
        self.env = env
        self.cmd_net = commander_model
        self.squad_net = squad_model
        self.squad_size = squad_size
        self.feat_dim_c = feat_dim_c
        self.feat_dim_s = feat_dim_s
        self.num_commands = num_commands
        self.map_size = map_size

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # A / B 基本信息
        self.num_team_a = env.num_team_a
        self.num_team_b = env.num_team_b
        self.n_agents_total = env.n_agents_total
        assert self.num_team_b % squad_size == 0, "B 队人数必须能整除 squad_size"

        # B 队小队划分（真实的小队数，可能 != CommanderPolicy.num_squads）
        self.squads_B = []
        start = self.num_team_a
        for s in range(0, self.num_team_b, squad_size):
            squad_indices = list(range(start + s, start + s + squad_size))
            self.squads_B.append(squad_indices)
        self.num_squads_B = len(self.squads_B)

        # 从 CommanderPolicy 读取它实际上用的小队槽位数（按 A 队定义）
        self.cmd_num_squads = getattr(self.cmd_net, "num_squads")
        self.cmd_feat_dim_c = getattr(self.cmd_net, "feat_dim_c")

        # 安全检查：输入维度要等于 cmd_num_squads * feat_dim_c + map_size
        expected_in_dim = self.cmd_num_squads * self.cmd_feat_dim_c + self.map_size
        real_in_dim = self.cmd_net.net[0].in_features
        if expected_in_dim != real_in_dim:
            print(
                f"[WARN] CommanderPolicy in_features={real_in_dim}, "
                f"but cmd_num_squads*feat_dim_c + map_size={expected_in_dim}"
            )

    def act(self, env: TacticalCombatEnv, team_id: int) -> np.ndarray:
        """
        返回一个 (num_team_b,) 的动作数组，供 _heuristic_team_actions 使用。
        注意：这里假定 team_id == TEAM_B。
        """
        assert team_id == TEAM_B, "OpponentControllerNN 目前只控制 B 队"
        base_env = get_core_env(env)

        H, W = base_env.grid_h, base_env.grid_w
        max_dist = H + W

        positions = base_env.agents_pos
        alive = base_env.alive_mask

        # 敌人对 B 队来说是 A 队 [0, num_team_a)
        enemy_pos = []
        for j in range(self.num_team_a):
            if not alive[j]:
                continue
            y, x = positions[j]
            enemy_pos.append((y, x))

        # B 队输出动作（局部索引 0..num_team_b-1）
        actions_B = np.zeros(self.num_team_b, dtype=np.int64)

        # ---------- 先为每个 B 小队算一个基础特征（不含命令 one-hot） ----------
        squad_feats_raw = []  # 每个元素：dict{cy, cx, alive_ratio, dy, dx, dist_norm}
        for s_idx, squad in enumerate(self.squads_B):
            ys = []
            xs = []
            alive_count = 0
            for idx in squad:
                if not alive[idx]:
                    continue
                alive_count += 1
                y, x = positions[idx]
                ys.append(y)
                xs.append(x)

            if alive_count == 0:
                cy = cx = 0.0
                cy_norm = cx_norm = 0.0
                alive_ratio = 0.0
                dy_norm = dx_norm = 0.0
                nearest_enemy_dist_norm = 1.0
            else:
                ys = np.array(ys, dtype=np.float32)
                xs = np.array(xs, dtype=np.float32)
                cy = float(ys.mean())
                cx = float(xs.mean())
                cy_norm = cy / max(1, H - 1)
                cx_norm = cx / max(1, W - 1)
                alive_ratio = alive_count / len(squad)

                if enemy_pos:
                    dists = [abs(ey - cy) + abs(ex - cx) for (ey, ex) in enemy_pos]
                    k = int(np.argmin(dists))
                    nearest_d = float(dists[k])
                    nearest_enemy_dist_norm = nearest_d / max_dist
                    ey, ex = enemy_pos[k]
                    dy = ey - cy
                    dx = ex - cx
                    dy_norm = dy / max_dist
                    dx_norm = dx / max_dist
                else:
                    nearest_enemy_dist_norm = 1.0
                    dy_norm = dx_norm = 0.0

            squad_feats_raw.append(
                dict(
                    cy_norm=cy_norm,
                    cx_norm=cx_norm,
                    alive_ratio=alive_ratio,
                    dy_norm=dy_norm,
                    dx_norm=dx_norm,
                    dist_norm=nearest_enemy_dist_norm,
                )
            )

        # ---------- 聚合成 CommanderPolicy 需要的 cmd_num_squads 个槽位 ----------
        commander_obs_B = np.zeros((self.cmd_num_squads, self.cmd_feat_dim_c), dtype=np.float32)

        # 简单做法：把 B 的 num_squads_B 划到 cmd_num_squads 个桶里，每桶取平均
        # 例如：num_squads_B=8, cmd_num_squads=4 -> 每 2 个小队聚成一个槽位
        if self.num_squads_B > 0:
            group_size = max(1, int(np.ceil(self.num_squads_B / self.cmd_num_squads)))
        else:
            group_size = 1

        for cmd_idx in range(self.cmd_num_squads):
            start = cmd_idx * group_size
            end = min(start + group_size, self.num_squads_B)
            if start >= end:
                # 没有对应小队，保持 0（完全空槽）
                continue

            sub_feats = squad_feats_raw[start:end]
            cy_norm = np.mean([f["cy_norm"] for f in sub_feats])
            cx_norm = np.mean([f["cx_norm"] for f in sub_feats])
            alive_ratio = np.mean([f["alive_ratio"] for f in sub_feats])
            dist_norm = np.mean([f["dist_norm"] for f in sub_feats])

            # 这里假设 Commander 的 feat_dim_c = 4 或 6，
            # 只用前面这几个基础量（你如果多了 morale/sniper，也可以在这里加平均值）
            commander_obs_B[cmd_idx, 0] = cy_norm
            commander_obs_B[cmd_idx, 1] = cx_norm
            commander_obs_B[cmd_idx, 2] = alive_ratio
            commander_obs_B[cmd_idx, 3] = dist_norm
            # 若有士气 / 狙击占比等特征，可继续写 commander_obs_B[cmd_idx,4:], 同时注意 feat_dim_c 一致

        # ---------- 拼 Commander 输入：flatten + 地图 ----------
        base_env = get_core_env(env)
        if getattr(base_env, "known_map_B", None) is not None:
            kmB = base_env.known_map_B.astype(np.float32) / 2.0
        else:
            kmB = np.zeros((H, W), dtype=np.float32)
        map_flat_B = kmB.flatten()  # len = map_size

        commander_input = np.concatenate(
            [commander_obs_B.reshape(-1), map_flat_B], axis=0
        )  # shape = (cmd_num_squads * feat_dim_c + map_size,)

        # ---------- Commander 前向：给 “槽位小队” 下命令 ----------
        with torch.no_grad():
            inp_t = torch.from_numpy(commander_input).float().to(self.device)
            logits_C_all = self.cmd_net(inp_t)  # (1, cmd_num_squads, num_commands)
            logits_C = logits_C_all[0]  # (cmd_num_squads, num_commands)
            dist_C = Categorical(logits=logits_C)
            cmd_actions_slots = dist_C.sample().cpu().numpy().astype(np.int64)  # (cmd_num_squads,)

        # ---------- 再为每个真实 B 小队构造 squad obs，并用小队网络出动作 ----------
        with torch.no_grad():
            for s_idx, squad in enumerate(self.squads_B):
                # 先拿存好的 raw feat
                f = squad_feats_raw[s_idx]
                feat = np.zeros(self.feat_dim_s, dtype=np.float32)
                feat[0] = f["cy_norm"]
                feat[1] = f["cx_norm"]
                feat[2] = f["alive_ratio"]
                feat[3] = f["dy_norm"]
                feat[4] = f["dx_norm"]

                # 地图 flatten 一样塞进来（和 A 队一致）
                feat_map_start = 5  # 注意：要和你的 SquadPolicy 里 map 的起始位置保持一致
                feat[feat_map_start: feat_map_start + self.map_size] = map_flat_B

                # 选择一个槽位命令：这里用循环复用的方式
                slot_idx = s_idx % self.cmd_num_squads
                c = int(cmd_actions_slots[slot_idx])
                cmd_onehot = np.zeros(self.num_commands, dtype=np.float32)
                if 0 <= c < self.num_commands:
                    cmd_onehot[c] = 1.0
                feat[-self.num_commands:] = cmd_onehot

                obs_t = torch.from_numpy(feat).float().to(self.device)
                logits_S = self.squad_net(obs_t)[0]  # (squad_size, num_actions)
                logits_S = mask_squad_action_logits(logits_S)
                dist_S = Categorical(logits=logits_S)
                a_squad = dist_S.sample().cpu().numpy().astype(np.int64)

                # 写入 B 队的局部动作数组
                for k, global_idx in enumerate(squad):
                    local_idx = global_idx - self.num_team_a
                    if 0 <= local_idx < self.num_team_b:
                        actions_B[local_idx] = a_squad[k]

        return actions_B


class CommanderPolicy(nn.Module):
    """
    Commander:
      输入: 拼成一维的 [所有小队特征, 全图 known_map_flat]
      输出: logits (B, num_squads, num_commands)
    """

    def __init__(self, num_squads: int, feat_dim_c: int, num_commands: int, map_size: int):
        super().__init__()
        self.num_squads = num_squads
        self.feat_dim_c = feat_dim_c
        self.num_commands = num_commands
        self.map_size = map_size

        # 输入向量 = 所有小队特征 + 全图 (H*W) 的认知值
        input_dim = num_squads * feat_dim_c + map_size

        hidden = 384
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            ResidualMLPBlock(hidden),
            ResidualMLPBlock(hidden),
            nn.Linear(hidden, num_squads * num_commands),
        )
        init_policy_module(self.net, output_gain=0.01)

    def forward(self, commander_input: torch.Tensor):
        """
        commander_input: (input_dim,) 或 (B, input_dim)
        返回:
          logits: (B, num_squads, num_commands)
        """
        if commander_input.dim() == 1:
            x = commander_input.unsqueeze(0)
        else:
            x = commander_input
        out = self.net(x)  # (B, num_squads * num_commands)
        out = out.view(-1, self.num_squads, self.num_commands)
        return out


class SquadPolicy(nn.Module):
    """
    SquadLeader 共享策略:
      输入: squad_obs: (feat_dim_s,) 或 (B, feat_dim_s)
      输出:
        logits: (B, squad_size, num_actions)
    """

    def __init__(self, feat_dim_s: int, squad_size: int, num_actions: int):
        super().__init__()
        self.feat_dim_s = feat_dim_s
        self.squad_size = squad_size
        self.num_actions = num_actions

        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(feat_dim_s, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            ResidualMLPBlock(hidden),
            ResidualMLPBlock(hidden),
            nn.Linear(hidden, squad_size * num_actions),
        )
        init_policy_module(self.net, output_gain=0.01)

    def forward(self, squad_obs: torch.Tensor):
        """
        squad_obs: (feat_dim_s,) or (B, feat_dim_s)
        返回:
          logits: (B, squad_size, num_actions)
        """
        if squad_obs.dim() == 1:
            x = squad_obs.unsqueeze(0)
        else:
            x = squad_obs
        out = self.net(x)  # (B, squad_size * num_actions)
        out = out.view(-1, self.squad_size, self.num_actions)
        return out


class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = 384
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            ResidualMLPBlock(hidden),
            ResidualMLPBlock(hidden),
            nn.Linear(hidden, 1),
        )
        init_policy_module(self.net, output_gain=1.0)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


# ======================
# 3.1 脚本小队长策略：用来做 BC 的“老师”
# ======================
def scripted_squad_policy(env: TacticalCombatEnv, squad_indices, command: int):
    """
    脚本小队长策略（行为克隆老师）- 带姿态 + 小队中心 + 2 名突击手

    command:
      0: ADVANCE         朝敌情/敌方入口推进
      1: DEFEND          优先依托当前阵位或掩体防守
      2: FLANK_LEFT      左侧迂回
      3: FLANK_RIGHT     右侧迂回
      4: RESCUE          向支援/营救目标机动
      5: CAUTIOUS        谨慎推进，优先借掩体
      6: COUNTER_ATTACK  向交火热点/敌情方向主动反击

    原子动作 (0..9):
      0: Idle
      1: TurnLeft
      2: TurnRight
      3: WalkForward
      4: WalkBackward
      5: StrafeLeft
      6: StrafeRight
      7: SprintForward
      8: ToggleCrouch
      9: Shoot

    本版增加三个核心语义：
      1) 指挥官给出“小队目标中心”，单步最多偏移 3 格（和 HierarchicalWrapper 保持一致）
      2) 队长以“小队目标中心”为主导移动，最多选出 2 名“突击手”向敌人方向拉开一点距离进攻/掩护
      3) 仍保留原有的姿态控制、掩体优先、防守/撤退/迂回等逻辑
    """
    base_env = get_core_env(env)
    squad_size = len(squad_indices)
    actions = np.zeros(squad_size, dtype=np.int64)

    positions = base_env.agents_pos
    alive = base_env.alive_mask
    H, W = base_env.grid_h, base_env.grid_w
    gun_range = base_env.gun_range

    # ---------- 小队级别信息：小队中心 ----------
    ys, xs = [], []
    for idx in squad_indices:
        if alive[idx]:
            y, x = positions[idx]
            ys.append(y)
            xs.append(x)
    if not ys:
        return actions  # 小队全灭

    cy = float(np.mean(ys))
    cx = float(np.mean(xs))

    # ---------- 只基于“感知到的敌情”，不再直接读全体敌人真实位置 ----------
    perceived_targets = []
    for idx in squad_indices:
        if alive[idx]:
            pt = base_env._perception_target(idx)
            if pt is not None:
                py, px = pt
                perceived_targets.append((int(py), int(px)))

    # 去重，避免多个成员看到同一目标重复计数
    perceived_targets = list(dict.fromkeys(perceived_targets))

    # ---------- 一些小工具函数（仅基于 perceived_targets） ----------

    def nearest_enemy_center():
        if not perceived_targets:
            return None, None
        dists = [abs(ey - cy) + abs(ex - cx) for (ey, ex) in perceived_targets]
        k = int(np.argmin(dists))
        return perceived_targets[k], dists[k]

    def nearest_cover_center():
        best = None
        best_d = None
        for y in range(H):
            for x in range(W):
                if base_env.terrain[y, x] == COVER:
                    d = abs(y - cy) + abs(x - cx)
                    if best_d is None or d < best_d:
                        best_d = d
                        best = (y, x)
        return best, best_d

    enemy_c, dist_c = nearest_enemy_center()
    cover_c, dist_cover_c = nearest_cover_center()

    # 如果当前没有感知到敌人，则退化为“按 command / doctrine / 入口方向”行动，
    # 不再回退到敌方全局真实位置。

    # A 队入口中心（RETREAT 用）
    y0, y1, x0, x1 = base_env.entrance_A
    ent_y = (y0 + y1) // 2
    ent_x = (x0 + x1) // 2
    # B 队入口中心（ADVANCE / FLANK 在无接敌时使用）
    y0_b, y1_b, x0_b, x1_b = base_env.entrance_B
    enemy_base_y = (y0_b + y1_b) / 2.0
    enemy_base_x = (x0_b + x1_b) / 2.0

    # ---------- 威胁等级 / 掩体 / 射击判断 ----------

    def can_shoot(i: int) -> bool:
        return base_env._has_shootable_enemy(i)

    def in_cover(i: int) -> bool:
        sy, sx = positions[i]
        return base_env.terrain[sy, sx] == COVER

    def enemy_threat_level():
        """
        粗略小队威胁等级:
          0: 无敌或很远
          1: 远威胁 (dist_c <= 2*gun_range)
          2: 近威胁 (dist_c <= gun_range+1)
        """
        if enemy_c is None or dist_c is None:
            return 0
        if dist_c <= gun_range + 1:
            return 2
        elif dist_c <= 2 * gun_range:
            return 1
        else:
            return 0

    threat = enemy_threat_level()

    # ---------- 小队“目标中心”（一次最多 3 格） ----------
    squad_target_y, squad_target_x = cy, cx

    if command == 0:
        # ADVANCE：朝最近敌人或敌方大致方向
        if enemy_c is not None:
            squad_target_y, squad_target_x = float(enemy_c[0]), float(enemy_c[1])
        else:
            squad_target_y, squad_target_x = float(enemy_base_y), float(enemy_base_x)

    elif command == 1:
        # DEFEND：目标中心偏向当前位置/最近掩体
        if cover_c is not None:
            squad_target_y, squad_target_x = float(cover_c[0]), float(cover_c[1])
        else:
            squad_target_y, squad_target_x = cy, cx

    elif command in (2, 3):
        # FLANK_LEFT / FLANK_RIGHT
        if enemy_c is not None:
            ey, ex = float(enemy_c[0]), float(enemy_c[1])
        else:
            ey, ex = float(enemy_base_y), float(enemy_base_x)

        if command == 2:
            fx = max(0.0, ex - 3.0)
        else:
            fx = min(float(W - 1), ex + 3.0)

        squad_target_y, squad_target_x = ey, fx

    elif command == 4:
        # RESCUE：朝敌方/任务方向推进
        if enemy_c is not None:
            squad_target_y, squad_target_x = float(enemy_c[0]), float(enemy_c[1])
        else:
            squad_target_y, squad_target_x = float(enemy_base_y), float(enemy_base_x)

    elif command == 5:
        # CAUTIOUS：优先靠近掩体；没有掩体就朝敌方方向谨慎前推
        if cover_c is not None:
            squad_target_y, squad_target_x = float(cover_c[0]), float(cover_c[1])
        else:
            squad_target_y, squad_target_x = float(enemy_base_y), float(enemy_base_x)

    elif command == 6:
        # COUNTER_ATTACK：局部反击，优先朝当前感知敌情推进
        if enemy_c is not None:
            squad_target_y, squad_target_x = float(enemy_c[0]), float(enemy_c[1])
        else:
            squad_target_y, squad_target_x = float(enemy_base_y), float(enemy_base_x)

    # 裁剪：一次最多 3 格（和 HierarchicalWrapper 一致）
    dy_c = squad_target_y - cy
    dx_c = squad_target_x - cx
    max_step = 3.0
    manhattan = abs(dy_c) + abs(dx_c)
    if manhattan > max_step and manhattan > 0:
        scale = max_step / manhattan
        squad_target_y = cy + dy_c * scale
        squad_target_x = cx + dx_c * scale

    # ---------- 选出最多 2 名“突击手” ----------
    striker_indices = []
    if enemy_c is not None:
        ey, ex = enemy_c
        dist_list = []
        for idx in squad_indices:
            if alive[idx]:
                y, x = positions[idx]
                d = abs(ey - y) + abs(ex - x)
                dist_list.append((d, idx))
        dist_list.sort(key=lambda t: t[0])
        striker_indices = [idx for (_, idx) in dist_list[:2]]

    # ---------- 一个小工具：根据“角色”（突击/普通）选择目标并移动 ----------
    def move_towards_with_role(idx: int, aggressive: bool) -> int:
        """
        aggressive=True:
          - 如果是突击手且有敌人：优先向敌人中心移动
          - 否则：向小队目标中心移动
        aggressive=False:
          - 不管是不是突击手，都按小队目标中心走（用于 RETREAT / HOLD 等）
        """
        if aggressive and (idx in striker_indices) and enemy_c is not None:
            ty, tx = enemy_c
        else:
            ty, tx = squad_target_y, squad_target_x

        # 目标可能是 float，小步修正为最近格
        ty_i = int(round(ty))
        tx_i = int(round(tx))
        ty_i = max(0, min(H - 1, ty_i))
        tx_i = max(0, min(W - 1, tx_i))
        return base_env._orient_and_move_towards(idx, ty_i, tx_i)

    # =============== 为每个成员计算动作 ===============

    for k, idx in enumerate(squad_indices):
        if not alive[idx]:
            actions[k] = 0
            continue

        post = int(base_env.agents_posture[idx])
        sy, sx = positions[idx]
        in_cov = in_cover(idx)

        # 能直接射击：优先开火
        if can_shoot(idx):
            actions[k] = 9
            continue

        # 在掩体上，先调整姿态降低被击中概率
        if in_cov and threat >= 1:
            push_cmd = command in (0, 2, 3, 4, 6)  # ADVANCE / FLANK / RESCUE / COUNTER_ATTACK
            move_aggressive = True if push_cmd else False

            if threat == 2:
                if post == 0:
                    actions[k] = 8  # 站 -> 蹲
                    continue
                elif post == 1:
                    if np.random.rand() < 0.35:
                        actions[k] = move_towards_with_role(idx, aggressive=move_aggressive)
                    else:
                        actions[k] = 0
                    continue
            else:
                if post == 0:
                    actions[k] = 8
                    continue
                elif post == 1:
                    if np.random.rand() < 0.55:
                        actions[k] = move_towards_with_role(idx, aggressive=move_aggressive)
                    else:
                        actions[k] = 0
                    continue

        # 推进类命令：到了掩体但当前打不到人，不要长期蹲坑
        if in_cov and (not can_shoot(idx)) and command in (0, 2, 3, 4, 6):
            if np.random.rand() < 0.55:
                actions[k] = move_towards_with_role(idx, aggressive=True)
                continue

        # ========== 根据 command 分支，结合“小队目标 + 突击手” ==========

        # 0) ADVANCE：突击手更激进朝敌人，其余人围着 squad_target 推进
        if command == 0:
            if threat == 2 and not in_cov:
                ey, ex = enemy_c if enemy_c is not None else (sy, sx)
                back_y = sy - (ey - sy)
                back_x = sx - (ex - sx)
                actions[k] = base_env._orient_and_move_towards(idx, back_y, back_x)
            else:
                actions[k] = move_towards_with_role(idx, aggressive=True)

        # 1) DEFEND：围绕当前阵位/掩体，少冒进，但不要长期发呆
        elif command == 1:
            if enemy_c is not None and threat == 2 and not in_cov:
                ey, ex = enemy_c
                back_y = sy - (ey - sy)
                back_x = sx - (ex - sx)
                actions[k] = base_env._orient_and_move_towards(idx, back_y, back_x)
            else:
                if np.random.rand() < 0.65:
                    actions[k] = move_towards_with_role(idx, aggressive=False)
                else:
                    actions[k] = 0

        # 2/3) FLANK_LEFT / FLANK_RIGHT
        elif command in (2, 3):
            if enemy_c is not None and threat == 2 and not in_cov:
                ey, ex = enemy_c
                back_y = sy - (ey - sy)
                back_x = sx - (ex - sx)
                actions[k] = base_env._orient_and_move_towards(idx, back_y, back_x)
            else:
                actions[k] = move_towards_with_role(idx, aggressive=True)

        # 4) RESCUE：向支援/营救目标机动，通常不比 ADVANCE 更保守
        elif command == 4:
            if threat >= 1 and post != 0:
                actions[k] = 8
            else:
                actions[k] = move_towards_with_role(idx, aggressive=False)

        # 5) CAUTIOUS：优先掩体，再小心推进
        elif command == 5:
            if cover_c is not None and (sy, sx) != cover_c:
                if threat >= 1 and post != 0:
                    actions[k] = 8
                else:
                    actions[k] = move_towards_with_role(idx, aggressive=False)
            else:
                if post == 0:
                    actions[k] = 8
                elif post == 1:
                    r = np.random.rand()
                    if r < 0.25:
                        actions[k] = 9
                    elif r < 0.60:
                        actions[k] = move_towards_with_role(idx, aggressive=False)
                    else:
                        actions[k] = 0
                else:
                    actions[k] = move_towards_with_role(idx, aggressive=False)

        # 6) COUNTER_ATTACK：朝当前敌情/交火热点主动反击
        elif command == 6:
            if threat == 2 and not in_cov:
                ey, ex = enemy_c if enemy_c is not None else (sy, sx)
                back_y = sy - (ey - sy)
                back_x = sx - (ex - sx)
                actions[k] = base_env._orient_and_move_towards(idx, back_y, back_x)
            else:
                actions[k] = move_towards_with_role(idx, aggressive=True)

        else:
            actions[k] = 0

    return actions


# ======================
# 4. 多智能体 REINFORCE 训练
# ======================

# ======================
# 4. 小队长行为克隆预训练
# ======================

def pretrain_squad_bc(
        map_type: str = "suburb",
        grid_size=(20, 20),
        num_team_a: int = 20,
        num_team_b: int = 20,
        num_episodes: int = 50,
        max_steps_per_ep: int = 80,
        batch_size: int = 64,
        bc_epochs: int = 5,
        lr: float = 1e-3,
        label_smoothing: float = 0.05,
        bc_entropy_coef: float = 0.02,
        seed: int = 42,
):
    """
    用脚本小队长策略 + 随机 Commander 命令 生成数据，
    对 SquadPolicy 做行为克隆预训练。
    返回：预训练好的 SquadPolicy 模型（可直接丢给 RL 用）
    """
    set_global_seeds(seed)

    # 创建一个小一点的环境，专门用来采 BC 数据
    base_env = TacticalCombatEnv(
        grid_size=grid_size,
        num_team_a=num_team_a if map_type != "indoor" else 10,
        num_team_b=num_team_b if map_type != "indoor" else 10,
        max_steps=max_steps_per_ep,
        map_type=map_type,
        render_mode=None,
        seed=seed,
    )
    hier_env = HierarchicalWrapper(base_env, squad_size=5)

    num_squads = hier_env.num_squads
    feat_dim_s = hier_env.feat_dim_s
    squad_size = hier_env.squad_size
    num_actions = 10
    num_commands = hier_env.num_commands

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    squad_model = SquadPolicy(feat_dim_s, squad_size, num_actions).to(device)
    optimizer = optim.Adam(squad_model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 收集数据
    X_list = []  # (feat_dim_s,)
    Y_list = []  # (squad_size,)
    rng = np.random.RandomState(seed)
    command_probs = np.asarray([0.38, 0.05, 0.12, 0.12, 0.08, 0.10, 0.15], dtype=np.float64)
    command_probs = command_probs / command_probs.sum()

    for ep in range(num_episodes):
        hier_env.env.reset(seed=seed + ep)
        done = False
        steps = 0

        while not done and steps < max_steps_per_ep:
            # BC 阶段偏向进攻命令，避免 warm-start 学成保守停顿策略。
            commands = rng.choice(num_commands, size=num_squads, p=command_probs).astype(np.int64)

            # 构造包含指令 one-hot 的 squad 观测
            hier_obs = hier_env._build_hier_obs(commands=commands)

            scripted_actions_list = []
            for s_idx, squad_indices in enumerate(hier_env.squads):
                obs_S = hier_obs.squad_obs_list[s_idx]
                # 脚本老师生成该小队动作
                teacher_actions = scripted_squad_policy(
                    hier_env.env, squad_indices, int(commands[s_idx])
                )
                for k, agent_idx in enumerate(squad_indices):
                    if hier_env.env.alive_mask[agent_idx] and hier_env.env._has_shootable_enemy(agent_idx):
                        teacher_actions[k] = 9
                scripted_actions_list.append(teacher_actions)

                # 加入数据集
                X_list.append(obs_S.astype(np.float32))
                Y_list.append(teacher_actions.astype(np.int64))

            # 环境步进（用脚本动作）
            hier_obs_next, r, done, info = hier_env.step(
                commander_action=commands,
                squad_actions=scripted_actions_list,
            )
            steps += 1

        print(f"[BC collect] episode {ep + 1}/{num_episodes}, steps={steps}")

    X = np.stack(X_list)  # (N, feat_dim_s)
    Y = np.stack(Y_list)  # (N, squad_size)
    N = X.shape[0]
    print(f"[BC] collected samples: {N}, feat_dim_s={feat_dim_s}, squad_size={squad_size}")
    action_counts = np.bincount(Y.reshape(-1), minlength=num_actions).astype(np.float64)
    action_ratio = action_counts / max(1.0, float(action_counts.sum()))
    print(
        "[BC] action ratio "
        f"idle={action_ratio[0]:.2f} "
        f"turn={(action_ratio[1] + action_ratio[2]):.2f} "
        f"move={action_ratio[3:8].sum():.2f} "
        f"shoot={action_ratio[9]:.2f} "
        f"other={(action_ratio[8] + action_ratio[10] + action_ratio[11]):.2f}"
    )

    # 训练行为克隆
    for epoch in range(bc_epochs):
        indices = rng.permutation(N)
        total_loss = 0.0
        num_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = indices[start:end]
            obs_batch = torch.from_numpy(X[batch_idx]).float().to(device)  # (B, feat_dim_s)
            act_batch = torch.from_numpy(Y[batch_idx]).long().to(device)  # (B, squad_size)

            logits = squad_model(obs_batch)  # (B, squad_size, num_actions)
            B = logits.size(0)
            logits_flat = logits.view(B * squad_size, num_actions)
            targets_flat = act_batch.view(B * squad_size)
            targets_flat = torch.where(
                targets_flat >= 10,
                torch.zeros_like(targets_flat),
                targets_flat,
            )

            # BC 的 label smoothing 只能在允许动作上计算。
            # 否则 smoothing 会给已 mask 的 10/11 分概率，和 -1e9 logits 组合导致巨大 loss。
            allowed_logits = logits_flat[:, :10]
            dist = Categorical(logits=allowed_logits)
            entropy = dist.entropy().mean()
            loss = ce_loss(allowed_logits, targets_flat) - bc_entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"[BC] epoch {epoch + 1}/{bc_epochs}, loss={avg_loss:.4f}")

    print("[BC] Squad pretraining finished.")
    return squad_model


# ======================
# 5. 多智能体 REINFORCE 训练（支持预训练小队长）
# ======================

def discount_rewards(rewards, gamma: float):
    G = 0.0
    rets = []
    for r in reversed(rewards):
        G = r + gamma * G
        rets.append(G)
    rets.reverse()
    return np.array(rets, dtype=np.float32)


def compute_gae(rewards, values, dones, gamma: float, gae_lambda: float):
    """
    rewards: (T,) np.float32
    values:  (T,) np.float32
    dones:   (T,) np.float32, 终止步=1.0
    返回:
      advantages: (T,)
      returns:    (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - dones[t]
            next_value = 0.0
        else:
            next_nonterminal = 1.0 - dones[t]
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ======================
# 5. 多智能体 PPO 训练（保持原接口，最小改动版）
# ======================

def train_multiagent(
        num_episodes: int = 350,
        max_steps_per_ep: int = 450,
        gamma: float = 0.97,
        lr_commander: float = 1e-4,
        lr_squad: float = 1e-3,
        map_type: str = "suburb",
        grid_size=(28, 28),
        num_team_a=20,
        num_team_b=15,
        squad_pretrained: SquadPolicy = None,
        use_nn_for_B: bool = False,
        record_video: bool = True,
        video_folder: str = "./videos",
        video_freq: int = 30,
        model_folder: str = "./models",
        save_last_k: int = 3,
        save_latest_every: int = 10,
        resume_path: str = None,
        squad_rl_warmup_episodes: int = 60,
        squad_rl_start_weight: float = 0.05,
        # ===== PPO 新参数（都有默认值，不影响旧调用） =====
        ppo_epochs: int = 5,
        ppo_minibatch_size: int = 128,
        ppo_clip_eps: float = 0.2,
        gae_lambda: float = 0.95,
        value_coef: float = 0.15,
        entropy_coef: float = 0.01,
        entropy_coef_c: float = None,
        entropy_coef_s: float = None,
        max_grad_norm: float = 0.5,
        seed: int = 42,
        best_eval_every: int = 50,
        best_eval_episodes: int = 4,
        best_candidate_eval_episodes: int = 3,
        best_candidate_cooldown: int = 10,
        stage=None,
        stage_start_episode: int = 0,
        train_log_path: str = "train_log.csv",
):
    set_global_seeds(seed)
    if entropy_coef_c is None:
        entropy_coef_c = entropy_coef
    if entropy_coef_s is None:
        entropy_coef_s = entropy_coef

    base_env = TacticalCombatEnv(
        grid_size=grid_size,
        num_team_a=num_team_a if map_type != "indoor" else 10,
        num_team_b=num_team_b if map_type != "indoor" else 10,
        max_steps=max_steps_per_ep,
        map_type=map_type,
        render_mode="rgb_array",
        seed=seed,
    )

    import os
    os.makedirs(model_folder, exist_ok=True)

    train_log_fields = [
        "global_episode",
        "local_episode",
        "stage",
        "num_team_a",
        "num_team_b",
        "raw_return",
        "scaled_return",
        "steps",
        "result",
        "is_win",
        "is_loss",
        "is_timeout",
    ]
    win_like_results = {"win", "win_cleared_outside", "team_b_eliminated", "all_enemies_cleared"}

    def _append_train_log(row: dict):
        if not train_log_path:
            return
        log_dir = os.path.dirname(train_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        write_header = not os.path.exists(train_log_path) or os.path.getsize(train_log_path) == 0
        with open(train_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=train_log_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ===== [改动1] 先用临时 wrapper 读取维度，不要在这里提前绑定真正训练用的 hier_env =====
    tmp_env = HierarchicalWrapper(base_env, squad_size=5)

    num_squads = tmp_env.num_squads
    feat_dim_c = tmp_env.feat_dim_c
    feat_dim_s = tmp_env.feat_dim_s
    squad_size = tmp_env.squad_size
    num_actions = 10
    num_commands = tmp_env.num_commands

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_size = tmp_env.grid_h * tmp_env.grid_w
    commander = CommanderPolicy(num_squads, feat_dim_c, num_commands, map_size).to(device)
    squad = SquadPolicy(feat_dim_s, squad_size, num_actions).to(device)
    value_input_dim = num_squads * feat_dim_c + map_size
    value_net = ValueNet(value_input_dim).to(device)

    if squad_pretrained is not None:
        print("[RL] loading pretrained squad weights into RL model")
        squad.load_state_dict(squad_pretrained.state_dict())

    if use_nn_for_B:
        base_core_env: TacticalCombatEnv = get_core_env(base_env)
        opponent_ctrl = OpponentControllerNN(
            env=base_core_env,
            commander_model=commander,
            squad_model=squad,
            squad_size=squad_size,
            feat_dim_c=feat_dim_c,
            feat_dim_s=feat_dim_s,
            num_commands=num_commands,
            map_size=map_size,
            device=device,
        )
        base_core_env.nn_controller_B = opponent_ctrl.act
        print("[RL] Team B is now controlled by neural networks (self-play opponent).")

    opt_c = optim.Adam(commander.parameters(), lr=lr_commander)
    opt_s = optim.Adam(squad.parameters(), lr=lr_squad)
    opt_v = optim.Adam(value_net.parameters(), lr=1e-4)

    start_ep = 0
    best_train_return = float("-inf")
    best_eval_metrics = {
        "win_rate": float("-inf"),
        "avg_survivors": float("-inf"),
        "avg_return": float("-inf"),
        "avg_steps": float("inf"),
    }
    last_candidate_eval_ep = -10 ** 9

    def _make_checkpoint(ep_idx: int, episode_return_value: float, extra: dict = None):
        ckpt = {
            "episode": ep_idx + 1,
            "return": episode_return_value,
            "best_train_return": max(best_train_return, episode_return_value),
            "best_eval_win_rate": float(best_eval_metrics["win_rate"]),
            "best_eval_avg_survivors": float(best_eval_metrics["avg_survivors"]),
            "best_eval_avg_return": float(best_eval_metrics["avg_return"]),
            "best_eval_avg_steps": float(best_eval_metrics["avg_steps"]),
            "map_type": map_type,
            "grid_size": grid_size,
            "num_team_a": num_team_a,
            "num_team_b_train": num_team_b,
            "squad_size": squad_size,
            "feat_dim_c": feat_dim_c,
            "feat_dim_s": feat_dim_s,
            "num_commands": num_commands,
            "map_size": map_size,
            "commander": commander.state_dict(),
            "squad": squad.state_dict(),
            "value_net": value_net.state_dict(),
            "opt_c": opt_c.state_dict(),
            "opt_s": opt_s.state_dict(),
            "opt_v": opt_v.state_dict(),
        }
        if extra:
            ckpt.update(extra)
        return ckpt

    def _is_eval_better(metrics, best_metrics):
        lhs = (
            float(metrics["win_rate"]),
            float(metrics["avg_survivors"]),
            float(metrics["avg_return"]),
            -float(metrics["avg_steps"]),
        )
        rhs = (
            float(best_metrics["win_rate"]),
            float(best_metrics["avg_survivors"]),
            float(best_metrics["avg_return"]),
            -float(best_metrics["avg_steps"]),
        )
        return lhs > rhs

    def _evaluate_current_policy(eval_episodes: int, seed_base: int):
        eval_env_core = TacticalCombatEnv(
            grid_size=grid_size,
            num_team_a=num_team_a if map_type != "indoor" else 10,
            num_team_b=num_team_b if map_type != "indoor" else 10,
            max_steps=max_steps_per_ep,
            map_type=map_type,
            render_mode=None,
            seed=seed_base,
        )

        if use_nn_for_B:
            eval_opponent_ctrl = OpponentControllerNN(
                env=eval_env_core,
                commander_model=commander,
                squad_model=squad,
                squad_size=squad_size,
                feat_dim_c=feat_dim_c,
                feat_dim_s=feat_dim_s,
                num_commands=num_commands,
                map_size=map_size,
                device=device,
            )
            eval_env_core.nn_controller_B = eval_opponent_ctrl.act

        eval_env = HierarchicalWrapper(eval_env_core, squad_size=5)
        prev_modes = (commander.training, squad.training, value_net.training)
        commander.eval()
        squad.eval()
        value_net.eval()

        returns = []
        survivors = []
        steps_list = []
        results = []
        win_like = {"win", "win_cleared_outside", "team_b_eliminated", "all_enemies_cleared"}

        with torch.no_grad():
            for k in range(eval_episodes):
                hier_obs = eval_env.reset(seed=seed_base + k)
                done = False
                ep_return = 0.0
                steps_eval = 0
                info_eval = {}

                while not done:
                    obs_C = torch.from_numpy(hier_obs.commander_obs.flatten()).float().to(device)
                    map_np = hier_obs.known_map.astype(np.float32) / 2.0
                    map_flat = torch.from_numpy(map_np.flatten()).float().to(device)
                    commander_input = torch.cat([obs_C, map_flat], dim=0)

                    logits_C_all = commander(commander_input)
                    logits_C = apply_commander_logits_prior(logits_C_all[0])
                    cmd_actions = torch.argmax(logits_C, dim=-1)
                    cmd_np_this_step = cmd_actions.detach().cpu().numpy().astype(np.int64)

                    hier_obs_for_squad = eval_env._build_hier_obs(commands=cmd_np_this_step)
                    squad_action_list = []
                    for s_idx in range(num_squads):
                        obs_S_np = hier_obs_for_squad.squad_obs_list[s_idx].astype(np.float32)
                        obs_S = torch.from_numpy(obs_S_np).float().to(device)
                        logits_S_all = squad(obs_S)
                        logits_S = apply_squad_tactical_logits(logits_S_all[0], obs_S)
                        action_mask_np = build_teamA_presampling_action_mask(eval_env.env, eval_env.squads[s_idx])
                        action_mask_t = torch.from_numpy(action_mask_np).to(device=device, dtype=logits_S.dtype)
                        logits_S = logits_S + action_mask_t
                        a_squad = torch.argmax(logits_S, dim=-1)
                        squad_action_list.append(a_squad.cpu().numpy().astype(np.int64))

                    hier_obs, r, done, info_eval = eval_env.step(
                        commander_action=cmd_np_this_step,
                        squad_actions=squad_action_list,
                    )
                    ep_return += float(r)
                    steps_eval += 1

                returns.append(ep_return)
                survivors.append(int(np.sum(get_core_env(eval_env.env).alive_mask[:num_team_a])))
                steps_list.append(steps_eval)
                results.append(info_eval.get("result", ""))

        eval_env.env.close()
        commander.train(prev_modes[0])
        squad.train(prev_modes[1])
        value_net.train(prev_modes[2])

        wins = sum(r in win_like for r in results)
        return {
            "win_rate": wins / max(1, len(results)),
            "avg_survivors": float(np.mean(survivors)) if survivors else 0.0,
            "avg_return": float(np.mean(returns)) if returns else 0.0,
            "avg_steps": float(np.mean(steps_list)) if steps_list else float("inf"),
        }

    if resume_path is not None and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        commander.load_state_dict(ckpt["commander"])
        squad.load_state_dict(ckpt["squad"])
        value_net.load_state_dict(ckpt["value_net"])
        if "opt_c" in ckpt:
            opt_c.load_state_dict(ckpt["opt_c"])
        if "opt_s" in ckpt:
            opt_s.load_state_dict(ckpt["opt_s"])
        if "opt_v" in ckpt:
            opt_v.load_state_dict(ckpt["opt_v"])
        start_ep = int(ckpt.get("episode", 0))
        best_train_return = float(
            ckpt.get("best_train_return", ckpt.get("best_return", ckpt.get("return", float("-inf"))))
        )
        if "best_eval_win_rate" in ckpt:
            best_eval_metrics = {
                "win_rate": float(ckpt.get("best_eval_win_rate", float("-inf"))),
                "avg_survivors": float(ckpt.get("best_eval_avg_survivors", float("-inf"))),
                "avg_return": float(ckpt.get("best_eval_avg_return", float("-inf"))),
                "avg_steps": float(ckpt.get("best_eval_avg_steps", float("inf"))),
            }
        print(f"[RL] Resumed from {resume_path}, start from episode {start_ep + 1}")

    # ===== [改动2] 一定要在拿到 start_ep 之后再包装 RecordVideo =====
    if record_video:
        os.makedirs(video_folder, exist_ok=True)
        base_env = RecordVideo(
            base_env,
            video_folder=video_folder,
            episode_trigger=lambda local_ep: ((start_ep + local_ep + 1) % video_freq == 0),
            name_prefix=f"map_{map_type}",
        )
        print(
            f"[RecordVideo] Enabled. Saving to {video_folder}, every {video_freq} episodes. "
            f"(resume offset = {start_ep})"
        )

    # ===== [改动3] 真正训练用的 wrapper 在 RecordVideo 之后再创建 =====
    hier_env = HierarchicalWrapper(base_env, squad_size=5)
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(int(seed) + 1000003)

    for ep in range(start_ep, num_episodes):
        hier_obs = hier_env.reset(seed=int(seed) + ep)
        done = False
        steps = 0
        info = {}

        ep_stall_penalty = 0.0
        ep_edge_penalty = 0.0
        ep_endgame_hotspot_reward = 0.0
        ep_cmd_advance = 0
        ep_cmd_cautious = 0
        ep_cmd_total = 0
        ep_act_idle = 0
        ep_act_turn = 0
        ep_act_move = 0
        ep_act_shoot = 0
        ep_act_crouch = 0
        ep_act_other = 0
        ep_act_total = 0
        ep_can_shoot_total = 0
        ep_shoot_when_can = 0
        ep_miss_shoot = 0
        ep_move_fail_ratio_sum = 0.0
        ep_move_fail_steps = 0

        rollout_commander_inputs = []
        rollout_cmd_actions = []
        rollout_old_logprob_c = []
        rollout_squad_obs = []
        rollout_squad_action_masks = []
        rollout_squad_actions = []
        rollout_old_logprob_s = []
        rollout_rewards = []
        rollout_values = []
        rollout_dones = []

        while not done:
            obs_C = torch.from_numpy(hier_obs.commander_obs.flatten()).float().to(device)
            map_np = hier_obs.known_map.astype(np.float32) / 2.0
            map_flat = torch.from_numpy(map_np.flatten()).float().to(device)
            commander_input = torch.cat([obs_C, map_flat], dim=0)

            with torch.no_grad():
                logits_C_all = commander(commander_input)
                value = value_net(commander_input)
                logits_C = apply_commander_logits_prior(logits_C_all[0])
                dist_C = Categorical(logits=logits_C)
                cmd_actions = dist_C.sample()
                logprob_C = dist_C.log_prob(cmd_actions).sum()

            cmd_np_this_step = cmd_actions.detach().cpu().numpy().astype(np.int64)
            ep_cmd_advance += int(np.sum(cmd_np_this_step == 0))
            ep_cmd_cautious += int(np.sum(cmd_np_this_step == 5))
            ep_cmd_total += int(cmd_np_this_step.size)

            hier_obs_for_squad = hier_env._build_hier_obs(commands=cmd_np_this_step)

            squad_action_list = []
            squad_obs_this_step = []
            squad_action_masks_this_step = []
            squad_actions_this_step = []
            logprob_S_total = 0.0

            with torch.no_grad():
                for s_idx in range(num_squads):
                    obs_S_np = hier_obs_for_squad.squad_obs_list[s_idx].astype(np.float32)
                    obs_S = torch.from_numpy(obs_S_np).float().to(device)
                    logits_S_all = squad(obs_S)
                    logits_S = logits_S_all[0]
                    logits_S = apply_squad_tactical_logits(logits_S, obs_S)
                    action_mask_np = build_teamA_presampling_action_mask(hier_env.env, hier_env.squads[s_idx])
                    action_mask_t = torch.from_numpy(action_mask_np).to(device=device, dtype=logits_S.dtype)
                    logits_S = logits_S + action_mask_t
                    dist_S = Categorical(logits=logits_S)
                    a_squad = dist_S.sample()
                    logprob_S = dist_S.log_prob(a_squad).mean()
                    logprob_S_total = logprob_S_total + logprob_S

                    squad_obs_this_step.append(obs_S_np)
                    squad_action_masks_this_step.append(action_mask_np)
                    a_squad_np = a_squad.cpu().numpy().astype(np.int64)
                    squad_actions_this_step.append(a_squad_np)
                    squad_action_list.append(a_squad_np)

                    ep_act_idle += int(np.sum(a_squad_np == 0))
                    ep_act_turn += int(np.sum((a_squad_np == 1) | (a_squad_np == 2)))
                    ep_act_move += int(np.sum((a_squad_np >= 3) & (a_squad_np <= 7)))
                    ep_act_shoot += int(np.sum(a_squad_np == 9))
                    ep_act_crouch += int(np.sum(a_squad_np == 8))
                    ep_act_other += int(np.sum((a_squad_np == 10) | (a_squad_np == 11)))
                    ep_act_total += int(a_squad_np.size)
                    if obs_S_np.shape[0] >= MEMBER_FEATURE_BASE + squad_size * MEMBER_SHOOT_FEAT_DIM:
                        can_shoot_np = np.zeros(squad_size, dtype=bool)
                        for k in range(squad_size):
                            off = MEMBER_FEATURE_BASE + k * MEMBER_SHOOT_FEAT_DIM
                            can_shoot_np[k] = obs_S_np[off + 2] > 0.5
                        ep_can_shoot_total += int(np.sum(can_shoot_np))
                        ep_shoot_when_can += int(np.sum((a_squad_np == 9) & can_shoot_np))
                        ep_miss_shoot += int(np.sum((a_squad_np != 9) & can_shoot_np))

                logprob_S_total = logprob_S_total / max(1, num_squads)

            hier_obs_next, r, done, info = hier_env.step(
                commander_action=cmd_np_this_step,
                squad_actions=squad_action_list,
            )
            ep_stall_penalty += float(info.get("stall_penalty", 0.0))
            ep_edge_penalty += float(info.get("edge_penalty", 0.0))
            ep_endgame_hotspot_reward += float(info.get("endgame_hotspot_reward", 0.0))
            if "move_fail_ratio_A" in info:
                ep_move_fail_ratio_sum += float(info.get("move_fail_ratio_A", 0.0))
                ep_move_fail_steps += 1

            rollout_commander_inputs.append(commander_input.detach().cpu().numpy().astype(np.float32))
            rollout_cmd_actions.append(cmd_actions.cpu().numpy().astype(np.int64))
            rollout_old_logprob_c.append(float(logprob_C.item()))
            rollout_squad_obs.append(np.stack(squad_obs_this_step, axis=0))
            rollout_squad_action_masks.append(np.stack(squad_action_masks_this_step, axis=0))
            rollout_squad_actions.append(np.stack(squad_actions_this_step, axis=0))
            rollout_old_logprob_s.append(float(logprob_S_total.item()))
            rollout_rewards.append(float(r) / 50.0)
            rollout_values.append(float(value.item()))
            rollout_dones.append(float(done))

            hier_obs = hier_obs_next
            steps += 1

        rewards_np = np.asarray(rollout_rewards, dtype=np.float32)
        values_np = np.asarray(rollout_values, dtype=np.float32)
        dones_np = np.asarray(rollout_dones, dtype=np.float32)
        adv_np, ret_np = compute_gae(rewards_np, values_np, dones_np, gamma=gamma, gae_lambda=gae_lambda)
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-6)

        commander_inputs_t = torch.from_numpy(np.asarray(rollout_commander_inputs, dtype=np.float32)).to(device)
        cmd_actions_t = torch.from_numpy(np.asarray(rollout_cmd_actions, dtype=np.int64)).to(device)
        old_logprob_c_t = torch.from_numpy(np.asarray(rollout_old_logprob_c, dtype=np.float32)).to(device)
        squad_obs_t = torch.from_numpy(np.asarray(rollout_squad_obs, dtype=np.float32)).to(device)
        squad_action_masks_t = torch.from_numpy(
            np.asarray(rollout_squad_action_masks, dtype=np.float32)
        ).to(device)
        squad_actions_t = torch.from_numpy(np.asarray(rollout_squad_actions, dtype=np.int64)).to(device)
        old_logprob_s_t = torch.from_numpy(np.asarray(rollout_old_logprob_s, dtype=np.float32)).to(device)
        returns_t = torch.from_numpy(ret_np).float().to(device)
        adv_t = torch.from_numpy(adv_np).float().to(device)

        if squad_pretrained is None:
            squad_loss_weight = 1.0
        else:
            if squad_rl_warmup_episodes <= 0:
                squad_loss_weight = 1.0
            else:
                alpha = min(1.0, ep / float(squad_rl_warmup_episodes))
                squad_loss_weight = squad_rl_start_weight + (1.0 - squad_rl_start_weight) * alpha

        N = commander_inputs_t.size(0)
        batch_size = min(ppo_minibatch_size, N)

        last_loss_c = 0.0
        last_loss_s = 0.0
        last_value_loss = 0.0
        last_entropy_c = 0.0
        last_entropy_s = 0.0

        for _ in range(ppo_epochs):
            perm = torch.randperm(N, device=device, generator=torch_rng)
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]

                mb_inputs = commander_inputs_t[idx]
                mb_cmd_actions = cmd_actions_t[idx]
                mb_old_logprob_c = old_logprob_c_t[idx]
                mb_squad_obs = squad_obs_t[idx]
                mb_squad_action_masks = squad_action_masks_t[idx]
                mb_squad_actions = squad_actions_t[idx]
                mb_old_logprob_s = old_logprob_s_t[idx]
                mb_returns = returns_t[idx]
                mb_adv = adv_t[idx]

                logits_C_all = apply_commander_logits_prior(commander(mb_inputs))
                dist_C = Categorical(logits=logits_C_all)
                new_logprob_c = dist_C.log_prob(mb_cmd_actions).sum(dim=1)
                entropy_c = dist_C.entropy().sum(dim=1).mean()

                ratio_c = torch.exp(new_logprob_c - mb_old_logprob_c)
                surr1_c = ratio_c * mb_adv
                surr2_c = torch.clamp(ratio_c, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps) * mb_adv
                loss_c = -torch.min(surr1_c, surr2_c).mean()

                B = mb_squad_obs.size(0)
                squad_obs_flat = mb_squad_obs.view(B * num_squads, feat_dim_s)
                logits_S_all = squad(squad_obs_flat)
                logits_S_all = logits_S_all.view(B, num_squads, squad_size, num_actions)
                logits_S_all = apply_squad_tactical_logits(logits_S_all, mb_squad_obs)
                logits_S_all = logits_S_all + mb_squad_action_masks
                dist_S = Categorical(logits=logits_S_all)
                new_logprob_s = dist_S.log_prob(mb_squad_actions).mean(dim=(1, 2))
                entropy_s = dist_S.entropy().mean(dim=(1, 2)).mean()

                ratio_s = torch.exp(new_logprob_s - mb_old_logprob_s)
                surr1_s = ratio_s * mb_adv
                surr2_s = torch.clamp(ratio_s, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps) * mb_adv
                loss_s = -torch.min(surr1_s, surr2_s).mean()

                values_pred = value_net(mb_inputs)
                value_loss = nn.functional.mse_loss(values_pred, mb_returns)

                total_loss = (
                        loss_c
                        + squad_loss_weight * loss_s
                        + value_coef * value_loss
                        - entropy_coef_c * entropy_c
                        - entropy_coef_s * entropy_s
                )

                opt_c.zero_grad()
                opt_s.zero_grad()
                opt_v.zero_grad()
                total_loss.backward()

                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(commander.parameters(), max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(squad.parameters(), max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)

                opt_c.step()
                opt_s.step()
                opt_v.step()

                last_loss_c = float(loss_c.item())
                last_loss_s = float(loss_s.item())
                last_value_loss = float(value_loss.item())
                last_entropy_c = float(entropy_c.item())
                last_entropy_s = float(entropy_s.item())

        raw_return = float(sum(rollout_rewards))
        scaled_return = float(raw_return * 50.0)
        episode_return = scaled_return
        result = str(info.get("result", ""))
        is_win = 1 if result in win_like_results else 0
        is_timeout = 1 if result == "timeout" else 0
        is_loss = 1 if (is_win == 0 and is_timeout == 0) else 0
        _append_train_log(
            {
                "global_episode": ep + 1,
                "local_episode": ep - int(stage_start_episode) + 1,
                "stage": "" if stage is None else stage,
                "num_team_a": num_team_a,
                "num_team_b": num_team_b,
                "raw_return": raw_return,
                "scaled_return": scaled_return,
                "steps": steps,
                "result": result,
                "is_win": is_win,
                "is_loss": is_loss,
                "is_timeout": is_timeout,
            }
        )
        is_new_best_train = episode_return > best_train_return
        if is_new_best_train:
            best_train_return = episode_return
            train_best_path = os.path.join(model_folder, "best_train_return.pth")
            torch.save(
                _make_checkpoint(ep, episode_return, extra={"checkpoint_type": "best_train_return"}),
                train_best_path,
            )

        should_eval_fixed = (
            best_eval_every is not None
            and best_eval_every > 0
            and ((ep + 1) % best_eval_every == 0)
        )
        should_eval_candidate = (
            best_candidate_eval_episodes is not None
            and best_candidate_eval_episodes > 0
            and is_new_best_train
            and (ep - last_candidate_eval_ep) >= max(1, best_candidate_cooldown)
        )

        if should_eval_fixed or should_eval_candidate:
            eval_eps = best_eval_episodes if should_eval_fixed else best_candidate_eval_episodes
            eval_seed_base = int(seed) + 200000 + (ep + 1) * 97
            eval_metrics = _evaluate_current_policy(eval_eps, eval_seed_base)
            trigger = "fixed" if should_eval_fixed else "candidate"
            print(
                f"[EVAL] Episode {ep + 1:4d} | trigger={trigger} | "
                f"episodes={eval_eps} | win={eval_metrics['win_rate']:.3f} | "
                f"survivors={eval_metrics['avg_survivors']:.2f} | "
                f"return={eval_metrics['avg_return']:.2f} | "
                f"steps={eval_metrics['avg_steps']:.1f}"
            )
            if should_eval_candidate:
                last_candidate_eval_ep = ep
            if _is_eval_better(eval_metrics, best_eval_metrics):
                best_eval_metrics = eval_metrics
                best_path = os.path.join(model_folder, "best.pth")
                torch.save(
                    _make_checkpoint(
                        ep,
                        episode_return,
                        extra={
                            "checkpoint_type": "best_eval",
                            "eval_episodes": int(eval_eps),
                            "eval_seed_base": int(eval_seed_base),
                        },
                    ),
                    best_path,
                )
                print(
                    f"[EVAL] Updated best.pth | win={best_eval_metrics['win_rate']:.3f} | "
                    f"survivors={best_eval_metrics['avg_survivors']:.2f} | "
                    f"return={best_eval_metrics['avg_return']:.2f} | "
                    f"steps={best_eval_metrics['avg_steps']:.1f}"
                )

        ckpt = _make_checkpoint(ep, episode_return)

        if save_latest_every is not None and save_latest_every > 0:
            if (ep + 1) % save_latest_every == 0:
                latest_path = os.path.join(model_folder, "latest.pth")
                torch.save(ckpt, latest_path)

        if record_video and video_freq is not None and video_freq > 0 and ((ep + 1) % video_freq == 0):
            mp4_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
            if mp4_files:
                print(f"[RecordVideo] Episode {ep + 1}: 视频文件已生成，最新文件 = {mp4_files[-1]}")
            else:
                print(f"[RecordVideo] Episode {ep + 1}: 尚未检测到视频文件，目录 = {video_folder}")

        if save_last_k is not None and save_last_k > 0:
            if (ep + 1) > (num_episodes - save_last_k):
                hist_path = os.path.join(model_folder, f"rl_ep_{ep + 1:04d}.pth")
                torch.save(ckpt, hist_path)
                latest_path = os.path.join(model_folder, "latest.pth")
                torch.save(ckpt, latest_path)

        if (ep + 1) % 5 == 0:
            cmd_adv_ratio = ep_cmd_advance / max(1, ep_cmd_total)
            cmd_caut_ratio = ep_cmd_cautious / max(1, ep_cmd_total)
            act_idle_ratio = ep_act_idle / max(1, ep_act_total)
            act_turn_ratio = ep_act_turn / max(1, ep_act_total)
            act_move_ratio = ep_act_move / max(1, ep_act_total)
            act_shoot_ratio = ep_act_shoot / max(1, ep_act_total)
            act_crouch_ratio = ep_act_crouch / max(1, ep_act_total)
            act_other_ratio = ep_act_other / max(1, ep_act_total)
            can_shoot_ratio = ep_can_shoot_total / max(1, ep_act_total)
            shoot_when_can_ratio = ep_shoot_when_can / max(1, ep_can_shoot_total)
            miss_shoot_ratio = ep_miss_shoot / max(1, ep_can_shoot_total)
            move_fail_ratio = ep_move_fail_ratio_sum / max(1, ep_move_fail_steps)
            print(
                f"[PPO] Episode {ep + 1:4d} | steps={steps:3d} "
                f"| return={episode_return:8.2f} "
                f"| stall={ep_stall_penalty:8.2f} "
                f"| edge={ep_edge_penalty:8.2f} "
                f"| failMv={move_fail_ratio:.2f} "
                f"| cmdA={cmd_adv_ratio:.2f} "
                f"| cmdCaut={cmd_caut_ratio:.2f} "
                f"| idle={act_idle_ratio:.2f} "
                f"| turn={act_turn_ratio:.2f} "
                f"| move={act_move_ratio:.2f} "
                f"| shoot={act_shoot_ratio:.2f} "
                f"| canShoot={can_shoot_ratio:.2f} "
                f"| shootCan={shoot_when_can_ratio:.2f} "
                f"| missShoot={miss_shoot_ratio:.2f} "
                f"| result={info.get('result', '')} "
                f"| entC={last_entropy_c:.3f} "
                f"| entS={last_entropy_s:.3f}"
            )

    print("[RL] Training finished.")

    hier_env.env.close()
    if record_video:
        mp4_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
        if mp4_files:
            print(f"[RecordVideo] flush 完毕，总共视频数量: {len(mp4_files)}, 最新文件: {mp4_files[-1]}")
        else:
            print(f"[RecordVideo] flush 完毕，但未检测到视频文件，目录: {video_folder}")

# All models are moved to `device`; env itself stays on CPU (normal for Gym).
