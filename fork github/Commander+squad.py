import numpy as np
from dataclasses import dataclass
from typing import List

from env import COVER, TacticalCombatEnv, get_core_env

# ======================
# 2. 分层封装：Commander + Squad
# ======================

@dataclass
class HierObs:
    commander_obs: np.ndarray  # (num_squads, feat_dim_c)
    squad_obs_list: List[np.ndarray]  # len=num_squads, each (feat_dim_s,)
    known_map: np.ndarray  # (H, W), A队的认知地图 0/1/2


class HierarchicalWrapper:
    """
    把 TacticalCombatEnv 包装成:
      - Commander: 对每个小队输出一个战术指令
      - SquadLeader: 共享策略, 对每个小队输出该队员的 primitive action
    """

    def __init__(self, env: TacticalCombatEnv, squad_size: int = 5):
        self.env = env
        self.squad_size = squad_size

        # 用底层真正的 TacticalCombatEnv 取属性
        base = get_core_env(env)

        self.num_team_a = base.num_team_a
        self.num_team_b = base.num_team_b
        self.n_agents_total = base.n_agents_total
        self.grid_h = base.grid_h
        self.grid_w = base.grid_w
        # self.entrance_B = base.entrance_B

        assert self.num_team_a % squad_size == 0, "A 队人数必须是 squad_size 的整数倍"
        self.squad_size = squad_size
        # 将 env 中的 default_squad_size 与封装的一致，方便教范策略按同样划分小队
        setattr(base, 'default_squad_size', self.squad_size)

        self.squads = []
        for start in range(0, self.num_team_a, self.squad_size):
            self.squads.append(list(range(start, start + self.squad_size)))
        self.num_squads = len(self.squads)

        # Commander 观测基础 6 维:
        # [center_y_norm, center_x_norm, alive_ratio,
        #  dist_to_target_norm, mean_morale_norm, sniper_ratio]
        # 额外再拼接:
        #   - doctrinal strategy one-hot (num_strategies)
        #   - doctrinal objective 相对于当前小队中心的 (dist_norm, dy_norm, dx_norm)
        self.num_strategies = 7
        self.strategy_list = [
            "ADVANCE",
            "DEFEND",
            "FLANK_LEFT",
            "FLANK_RIGHT",
            "RESCUE",
            "CAUTIOUS",
            "COUNTER_ATTACK",
        ]
        self.strategy2idx = {s: i for i, s in enumerate(self.strategy_list)}

        # Commander 每小队最终特征维度
        self.feat_dim_c = 6 + self.num_strategies + 3

        # Squad 观测：基础 7 维 + 战术聚合 8 维 + 队员射击特征 + 地图 + 命令 one-hot
        # 基础 7 维:
        # [center_y_norm, center_x_norm, alive_ratio,
        #  dy_norm_to_target, dx_norm_to_target, mean_morale_norm, sniper_ratio]
        #
        # 战术聚合 8 维:
        # [visible_enemy_ratio,
        #  support_needed_ratio,
        #  support_dy_norm, support_dx_norm,
        #  fight_dy_norm, fight_dx_norm,
        #  flank_dy_norm, flank_dx_norm]
        self.num_commands = self.num_strategies
        self.base_local_feat_dim = 7
        self.tactical_feat_dim = 8
        self.member_shoot_feat_dim = 6
        self.member_feat_dim = self.squad_size * self.member_shoot_feat_dim
        self.local_feat_dim = self.base_local_feat_dim + self.tactical_feat_dim + self.member_feat_dim

        # 地图大小（flatten 后长度）
        self.map_size = self.grid_h * self.grid_w  # H*W

        # 最终小队观测维度
        self.base_feat_dim_s = self.local_feat_dim + self.map_size
        self.feat_dim_s = self.base_feat_dim_s + self.num_commands

        # 神经网络对手控制（可选）：如果不为 None，B 队的动作通过这个回调给出
        self.nn_controller_B = None

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        hier_obs = self._build_hier_obs(commands=None)
        return hier_obs

    def step(self, commander_action: np.ndarray, squad_actions: List[np.ndarray]):
        # 汇总成 Team A 动作
        actions_teamA = np.zeros(self.num_team_a, dtype=np.int64)
        for s_idx, squad in enumerate(self.squads):
            a_squad = squad_actions[s_idx]
            assert a_squad.shape[0] == len(squad)
            for k, agent_idx in enumerate(squad):
                actions_teamA[agent_idx] = a_squad[k]

        non_idle = int((actions_teamA != 0).sum())
        # 你可以临时打开这一句看看：
        # print(f"[DEBUG] non-idle A agents this step: {non_idle} / {self.num_team_a}")

        obs_core, reward, terminated, truncated, info = self.env.step(actions_teamA)
        hier_obs_next = self._build_hier_obs(commands=commander_action)
        done = terminated or truncated
        return hier_obs_next, reward, done, info

    def _build_hier_obs(self, commands: np.ndarray = None) -> HierObs:
        """
        分层观测构造：
          - Commander 每个小队 feat_dim_c = 6 维：
              [center_y_norm, center_x_norm, alive_ratio,
               dist_to_target_center_norm, mean_morale_norm, sniper_ratio]

          - Squad 每个小队 feat_dim_s 维：
              [center_y_norm, center_x_norm, alive_ratio,
               dy_norm_to_target_center, dx_norm_to_target_center,
               mean_morale_norm, sniper_ratio,
               known_map_flat..., command_onehot...]

        目标中心 target_center 由 Commander 命令决定（统一 7 类语义）：
          0: ADVANCE
          1: DEFEND
          2: FLANK_LEFT
          3: FLANK_RIGHT
          4: RESCUE
          5: CAUTIOUS
          6: COUNTER_ATTACK

        然后将 (当前中心 → 目标中心) 的位移裁剪到一次最多 3 格。
        """
        H, W = self.grid_h, self.grid_w
        max_dist = H + W

        base_env = get_core_env(self.env)

        positions = base_env.agents_pos
        alive = base_env.alive_mask

        # 己方 & 敌方入口中心
        y0_a, y1_a, x0_a, x1_a = base_env.entrance_A
        y0_b, y1_b, x0_b, x1_b = base_env.entrance_B
        my_base_y = (y0_a + y1_a) / 2.0
        my_base_x = (x0_a + x1_a) / 2.0
        enemy_base_y = (y0_b + y1_b) / 2.0
        enemy_base_x = (x0_b + x1_b) / 2.0

        # 找所有掩体，用于 TAKE_COVER
        cover_cells = [
            (y, x)
            for y in range(H)
            for x in range(W)
            if base_env.terrain[y, x] == COVER
        ]

        commander_obs = np.zeros((self.num_squads, self.feat_dim_c), dtype=np.float32)
        squad_obs_list: List[np.ndarray] = []

        # 从底层 env 取 A 队认知地图（0=未知,1=墙/边界,2=可走）
        base_env = get_core_env(self.env)
        if getattr(base_env, "known_map_A", None) is not None:
            kmA = base_env.known_map_A.astype(np.float32) / 2.0  # 归一到 [0,1]
        else:
            kmA = np.zeros((H, W), dtype=np.float32)
        map_flat = kmA.flatten()  # len = self.map_size

        for s_idx, squad in enumerate(self.squads):
            alive_members = []
            main_members = []

            support_targets_A = getattr(base_env, "current_support_target_A", None)

            for idx in squad:
                if not alive[idx]:
                    continue

                alive_members.append(idx)

                # A队：被派出去支援的 agent，不参与本 squad 主体中心 / alive_ratio / 统计特征
                if support_targets_A is not None and idx < self.num_team_a:
                    if support_targets_A[idx] is not None:
                        continue

                main_members.append(idx)

            # 主体成员优先；如果整个 squad 都被支援抽空了，再退化回全部存活成员
            core_members = main_members if len(main_members) > 0 else alive_members

            if len(core_members) == 0:
                cy = cx = 0.0
                cy_norm = 0.0
                cx_norm = 0.0
                alive_ratio = 0.0
            else:
                ys = np.array([positions[idx][0] for idx in core_members], dtype=np.float32)
                xs = np.array([positions[idx][1] for idx in core_members], dtype=np.float32)
                cy = float(ys.mean())
                cx = float(xs.mean())
                cy_norm = cy / max(1, H - 1)
                cx_norm = cx / max(1, W - 1)
                alive_ratio = len(core_members) / len(squad)

            # ===== 小队平均士气 =====
            if hasattr(base_env, "agent_morale") and base_env.agent_morale is not None:
                morale_vals = [
                    float(base_env.agent_morale[idx])
                    for idx in squad
                    if alive[idx]
                ]
                if morale_vals:
                    mean_morale = float(np.mean(morale_vals))
                    morale_max = getattr(base_env, "morale_max", 1.0)
                    mean_morale_norm = mean_morale / max(1e-6, morale_max)
                else:
                    mean_morale_norm = 0.0
            else:
                mean_morale_norm = 0.0

            # ===== 狙击占比 =====
            sniper_ratio = 0.0
            if hasattr(base_env, "agent_weapon_type") and base_env.agent_weapon_type is not None:
                sniper_cnt = 0
                total_cnt = 0
                for idx in squad:
                    if not alive[idx]:
                        continue
                    total_cnt += 1
                    if int(base_env.agent_weapon_type[idx]) == 1:  # 1 = Sniper
                        sniper_cnt += 1
                sniper_ratio = (sniper_cnt / total_cnt) if total_cnt > 0 else 0.0

            # ===== 根据 command 决定“理想目标中心” =====
            # 统一命令语义（必须与 self.strategy_list / scripted_squad_policy 一致）:
            # 0 = ADVANCE
            # 1 = DEFEND
            # 2 = FLANK_LEFT
            # 3 = FLANK_RIGHT
            # 4 = RESCUE
            # 5 = CAUTIOUS
            # 6 = COUNTER_ATTACK
            if commands is not None:
                c = int(commands[s_idx])
            else:
                c = self.strategy2idx["ADVANCE"]  # 默认前推

            # ---------- 先收集当前小队附近/可感知到的敌人 ----------
            visible_enemies = []
            for j in range(self.num_team_a, self.n_agents_total):
                if not alive[j]:
                    continue
                ey, ex = positions[j]

                seen = False
                # 只要小队里任一活着成员能看到 / 感知到这个敌人，就算进来
                for idx in squad:
                    if not alive[idx]:
                        continue
                    ay, ax = positions[idx]

                    # 视野内可见
                    if base_env._within_fov(idx, ey, ex) and base_env._has_line_of_sight(ay, ax, ey, ex):
                        seen = True
                        break

                    # 或者在静态感知范围内（你 env 里本来就有 perception_range）
                    if abs(ey - ay) + abs(ex - ax) <= getattr(base_env, "perception_range", 6):
                        seen = True
                        break

                if seen:
                    visible_enemies.append((ey, ex))

            # ---------- 一些常用锚点 ----------
            # 最近掩体
            nearest_cover = None
            if len(core_members) > 0 and len(cover_cells) > 0:
                dists = [(abs(y - cy) + abs(x - cx), y, x) for (y, x) in cover_cells]
                _, cov_y, cov_x = min(dists, key=lambda t: t[0])
                nearest_cover = (float(cov_y), float(cov_x))

            # 小队已知敌情中心（有则优先用）
            if len(visible_enemies) > 0:
                enemy_mean_y = float(np.mean([p[0] for p in visible_enemies]))
                enemy_mean_x = float(np.mean([p[1] for p in visible_enemies]))
            else:
                enemy_mean_y = enemy_base_y
                enemy_mean_x = enemy_base_x

            forward_cover = None
            if len(core_members) > 0 and len(cover_cells) > 0:
                dir_y = enemy_mean_y - cy
                dir_x = enemy_mean_x - cx
                dir_len = max(1e-6, float(np.sqrt(dir_y * dir_y + dir_x * dir_x)))
                candidates = []
                for y, x in cover_cells:
                    rel_y = float(y) - cy
                    rel_x = float(x) - cx
                    forward_score = rel_y * dir_y + rel_x * dir_x
                    if forward_score <= 0:
                        continue
                    forward_progress = forward_score / dir_len
                    if forward_progress < 2.0 or forward_progress > 12.0:
                        continue
                    lateral = abs(rel_y * dir_x - rel_x * dir_y) / dir_len
                    if lateral > 5.0:
                        continue
                    d = abs(rel_y) + abs(rel_x)
                    candidates.append((forward_progress + 0.7 * lateral + 0.1 * d, y, x))
                if candidates:
                    _, cov_y, cov_x = min(candidates, key=lambda t: t[0])
                    forward_cover = (float(cov_y), float(cov_x))

            # 交火热点（若 env 有记忆）
            hotspot = getattr(base_env, "last_engagement_center", None)
            if hotspot is not None:
                hot_y, hot_x = float(hotspot[0]), float(hotspot[1])
            else:
                hot_y, hot_x = enemy_mean_y, enemy_mean_x

            # 支援目标（如果 env 正在维护）
            rescue_target = None
            if hasattr(base_env, "current_support_target_A") and base_env.current_support_target_A is not None:
                squad_targets = []
                for idx in squad:
                    if not alive[idx]:
                        continue
                    tgt = base_env.current_support_target_A[idx]
                    if tgt is not None:
                        squad_targets.append(tgt)
                if len(squad_targets) > 0:
                    rescue_target = (
                        float(np.mean([t[0] for t in squad_targets])),
                        float(np.mean([t[1] for t in squad_targets])),
                    )

            # 主推进锚点始终偏向 B 出生点，同时允许向交火热点/支援点吸附。
            push_anchor_y = enemy_base_y
            push_anchor_x = enemy_base_x
            if hotspot is not None:
                push_anchor_y = 0.35 * hot_y + 0.65 * enemy_base_y
                push_anchor_x = 0.35 * hot_x + 0.65 * enemy_base_x
            if rescue_target is not None:
                push_anchor_y = 0.55 * rescue_target[0] + 0.45 * push_anchor_y
                push_anchor_x = 0.55 * rescue_target[1] + 0.45 * push_anchor_x

            # 避免推进锚点被拉回当前战线后方，保证在左半场/上半场时也继续向右下压进。
            push_anchor_y = max(push_anchor_y, cy + 0.30 * (enemy_base_y - cy))
            push_anchor_x = max(push_anchor_x, cx + 0.30 * (enemy_base_x - cx))

            # ---------- 按统一后的 7 类命令确定目标 ----------
            # 默认：朝敌情/敌方基地方向
            ty, tx = enemy_mean_y, enemy_mean_x

            if c == self.strategy2idx["ADVANCE"]:
                # 主动推进：有敌情则压敌情，否则沿主推进锚点继续压向 B 出生点。
                if len(visible_enemies) > 0:
                    ty, tx = enemy_mean_y, enemy_mean_x
                else:
                    ty, tx = push_anchor_y, push_anchor_x

            elif c == self.strategy2idx["DEFEND"]:
                # 防守：优先最近掩体，没有就保持当前阵位
                if nearest_cover is not None:
                    ty, tx = nearest_cover
                else:
                    ty, tx = cy, cx

            elif c == self.strategy2idx["FLANK_LEFT"]:
                # 左翼包抄：仍沿主推进方向前压，但保留左侧展开。
                ty = push_anchor_y
                tx = max(0.0, push_anchor_x - W * 0.16)

            elif c == self.strategy2idx["FLANK_RIGHT"]:
                ty = push_anchor_y
                tx = min(W - 1.0, push_anchor_x + W * 0.16)

            elif c == self.strategy2idx["RESCUE"]:
                # 支援/营救：优先队友支援点，否则去能兼顾前线与纵深推进的锚点。
                if rescue_target is not None:
                    ty, tx = rescue_target
                else:
                    ty, tx = push_anchor_y, push_anchor_x

            elif c == self.strategy2idx["CAUTIOUS"]:
                # 谨慎推进：优先找前进方向上的近掩体；没有则朝敌情小步前压
                if forward_cover is not None:
                    ty, tx = forward_cover
                else:
                    ty, tx = push_anchor_y, push_anchor_x

            elif c == self.strategy2idx["COUNTER_ATTACK"]:
                # 反击：优先最近交火热点，但不允许目标被拉回后方。
                ty, tx = max(hot_y, push_anchor_y), max(hot_x, push_anchor_x)

            # ===== 将位移裁剪到“每次最多 3 格” =====
            dy = ty - cy
            dx = tx - cx
            manhattan = abs(dy) + abs(dx)
            max_step = 3.0
            if manhattan > max_step and manhattan > 0:
                scale = max_step / manhattan
                ty = cy + dy * scale
                tx = cx + dx * scale
                dy = ty - cy
                dx = tx - cx

            dist_to_target = abs(dy) + abs(dx)
            dist_target_norm = float(dist_to_target / max_dist) if max_dist > 0 else 0.0
            dy_norm = float(dy / max_dist)
            dx_norm = float(dx / max_dist)

            # ===== Commander 观测：基础 6 维 + doctrinal 附加特征 =====
            commander_obs[s_idx, 0] = cy_norm
            commander_obs[s_idx, 1] = cx_norm
            commander_obs[s_idx, 2] = alive_ratio
            commander_obs[s_idx, 3] = dist_target_norm
            commander_obs[s_idx, 4] = mean_morale_norm
            commander_obs[s_idx, 5] = sniper_ratio

            # strategy one-hot（优先读取 squad 级教范策略；没有时再退回 team 级 posture）
            strat_onehot = np.zeros(self.num_strategies, dtype=np.float32)
            base_env = get_core_env(self.env)

            strat_name = None

            squad_strats = getattr(base_env, 'current_squad_strategies_A', None)
            if isinstance(squad_strats, (list, tuple)) and len(squad_strats) == self.num_squads:
                s_name = squad_strats[s_idx]
                if isinstance(s_name, str):
                    strat_name = s_name

            if strat_name is None:
                team_strat = getattr(base_env, 'current_strategy_A', None)
                if isinstance(team_strat, str):
                    strat_name = team_strat

            if isinstance(strat_name, str) and strat_name in self.strategy2idx:
                strat_idx = self.strategy2idx[strat_name]
                strat_onehot[strat_idx] = 1.0
            # 拼接到 commander_obs 第 6..6+num_strategies-1 维
            strat_base = 6
            commander_obs[s_idx, strat_base: strat_base + self.num_strategies] = strat_onehot

            # doctrinal objective：相对当前小队中心的 (dist_norm, dy_norm, dx_norm)
            doc_base = 6 + self.num_strategies
            doc_dist_norm = 0.0
            doc_dy_norm = 0.0
            doc_dx_norm = 0.0

            # 从 env.current_squad_objectives_A 中取出对应小队目标（按相同的 squad 划分）
            doc_objs = getattr(base_env, 'current_squad_objectives_A', None)
            if isinstance(doc_objs, (list, tuple)) and len(doc_objs) == self.num_squads:
                obj = doc_objs[s_idx]
                if obj is not None:
                    ty_doc, tx_doc = obj
                    dy_doc = float(ty_doc - cy)
                    dx_doc = float(tx_doc - cx)
                    doc_dist = abs(dy_doc) + abs(dx_doc)
                    max_dist = self.grid_h + self.grid_w
                    if max_dist > 0:
                        doc_dist_norm = doc_dist / max_dist
                        doc_dy_norm = dy_doc / max_dist
                        doc_dx_norm = dx_doc / max_dist

            commander_obs[s_idx, doc_base + 0] = doc_dist_norm
            commander_obs[s_idx, doc_base + 1] = doc_dy_norm
            commander_obs[s_idx, doc_base + 2] = doc_dx_norm

            # ===== Squad 观测：基础 7 维 + 战术聚合 8 维 + 地图 + 命令 one-hot =====
            feat = np.zeros(self.feat_dim_s, dtype=np.float32)

            # ------------------------------------------------------------------
            # 1) 基础 7 维
            # ------------------------------------------------------------------
            feat[0] = cy_norm
            feat[1] = cx_norm
            feat[2] = alive_ratio
            feat[3] = dy_norm  # 对目标中心的偏移
            feat[4] = dx_norm
            feat[5] = mean_morale_norm
            feat[6] = sniper_ratio

            # ------------------------------------------------------------------
            # 2) 从底层 env 的 20 维单兵 obs 聚合战术维度
            #    env._get_obs() 里 A 队已经有 visible enemy / support / fight / flank 等信息
            # ------------------------------------------------------------------
            obs_all = base_env._get_obs()  # (n_agents_total, obs_dim=20)

            obs_members = core_members if len(core_members) > 0 else alive_members

            # ===== [可选改动] squad 观测里排除已被派出去支援的 agent，避免轻微带偏 visible_enemy_ratio =====
            alive_main_members = []
            for idx in alive_members:
                if getattr(base_env, "current_support_target_A", None) is not None:
                    if idx < self.num_team_a and base_env.current_support_target_A[idx] is not None:
                        continue
                alive_main_members.append(idx)

            if len(obs_members) > 0:
                squad_obs_agents = obs_all[obs_members]  # (n_alive_main, 20)

                # [9] 自己是否看到敌人
                visible_enemy_ratio = float(np.mean(squad_obs_agents[:, 9]))

                # [15] support_needed
                support_needed_ratio = float(np.mean(squad_obs_agents[:, 15]))

                # [13],[14] 当前支援/目标相对坐标（你 env 里这里实际是 squad objective）
                support_dy_norm = float(np.mean(squad_obs_agents[:, 13]))
                support_dx_norm = float(np.mean(squad_obs_agents[:, 14]))

                # [16],[17] 最近交火中心相对坐标
                fight_dy_norm = float(np.mean(squad_obs_agents[:, 16]))
                fight_dx_norm = float(np.mean(squad_obs_agents[:, 17]))

                # [18],[19] 包抄参考点相对坐标
                flank_dy_norm = float(np.mean(squad_obs_agents[:, 18]))
                flank_dx_norm = float(np.mean(squad_obs_agents[:, 19]))
            else:
                visible_enemy_ratio = 0.0
                support_needed_ratio = 0.0
                support_dy_norm = 0.0
                support_dx_norm = 0.0
                fight_dy_norm = 0.0
                fight_dx_norm = 0.0
                flank_dy_norm = 0.0
                flank_dx_norm = 0.0

            # 写入新增 8 维
            feat[7] = visible_enemy_ratio
            feat[8] = support_needed_ratio
            feat[9] = support_dy_norm
            feat[10] = support_dx_norm
            feat[11] = fight_dy_norm
            feat[12] = fight_dx_norm
            feat[13] = flank_dy_norm
            feat[14] = flank_dx_norm

            # ------------------------------------------------------------------
            # 3) 每个队员的射击局部特征，按 squad 内固定顺序写入。
            #    [alive, can_see, can_shoot, rel_enemy_y, rel_enemy_x, ammo_ready]
            # ------------------------------------------------------------------
            member_base = self.base_local_feat_dim + self.tactical_feat_dim
            for k, idx in enumerate(squad):
                if k >= self.squad_size:
                    break
                off = member_base + k * self.member_shoot_feat_dim
                if not alive[idx]:
                    continue

                can_see = 1.0 if base_env._agent_can_see_enemy(idx) else 0.0
                can_shoot = 1.0 if base_env._has_shootable_enemy(idx) else 0.0

                rel_enemy_y = 0.0
                rel_enemy_x = 0.0
                nearest_enemy_idx = base_env._get_nearest_visible_enemy_for_agent(idx)
                if nearest_enemy_idx is not None:
                    ey, ex = positions[nearest_enemy_idx]
                    ay, ax = positions[idx]
                    rel_enemy_y = float(np.clip((ey - ay) / max(1, H - 1), -1.0, 1.0))
                    rel_enemy_x = float(np.clip((ex - ax) / max(1, W - 1), -1.0, 1.0))

                ammo_ready = 1.0
                if getattr(base_env, "agent_reload_timer", None) is not None:
                    if int(base_env.agent_reload_timer[idx]) > 0:
                        ammo_ready = 0.0
                if getattr(base_env, "agent_ammo", None) is not None:
                    if int(base_env.agent_ammo[idx]) <= 0:
                        ammo_ready = 0.0

                feat[off + 0] = 1.0
                feat[off + 1] = can_see
                feat[off + 2] = can_shoot
                feat[off + 3] = rel_enemy_y
                feat[off + 4] = rel_enemy_x
                feat[off + 5] = ammo_ready

            # ------------------------------------------------------------------
            # 4) flatten map
            # ------------------------------------------------------------------
            start_map = self.local_feat_dim
            end_map = start_map + self.map_size
            feat[start_map:end_map] = map_flat

            # base_feat_dim_s = local_feat_dim + map_size
            # [base_feat_dim_s:] 存 command one-hot
            cmd_onehot = np.zeros(self.num_commands, dtype=np.float32)
            if commands is not None and 0 <= c < self.num_commands:
                cmd_onehot[c] = 1.0
            feat[self.base_feat_dim_s: self.base_feat_dim_s + self.num_commands] = cmd_onehot

            squad_obs_list.append(feat)

        # 返回 HierObs（带上当前 A 队认知地图）
        known_map = base_env.known_map_A.copy() if getattr(base_env, "known_map_A", None) is not None else None

        return HierObs(
            commander_obs=commander_obs,
            squad_obs_list=squad_obs_list,
            known_map=known_map,
        )
