import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Wedge, Rectangle, Circle, FancyArrowPatch

def get_core_env(env):
    """Unwrap nested gym / gymnasium wrappers without using .unwrapped."""
    while hasattr(env, "env"):
        env = env.env
    return env

# ======================
# 1. TacticalCombatEnv：真实战术环境
# ======================

PLAIN = 0
WALL = 1
COVER = 2

TEAM_A = 0
TEAM_B = 1


class TacticalCombatEnv(gym.Env):
    """
    强化版战术环境（V2）:
      - 三类地图: 郊外(suburb) / 城市(city) / 室内(indoor)
      - 双入口: entrance_A / entrance_B
      - 城市: 平民 (civilians)，误杀 = 重罚 + 终止
      - 室内: 人质房间 (hostage room)，人质死 = 重罚 + 终止
      - 朝向: 0~7 (8 个方向)
      - 视野: FOV 锥形，只允许在 FOV + 射程 + LOS 内射击
      - 姿态: 0=站, 1=蹲
      - 动作空间（敌我完全相同，10 个离散动作）:
          0: Idle            不动
          1: TurnLeft        朝向左转 45°
          2: TurnRight       朝向右转 45°
          3: WalkForward     朝当前朝向走 1 格
          4: WalkBackward    反向走 1 格
          5: StrafeLeft      左平移 1 格（侧移）
          6: StrafeRight     右平移 1 格
          7: SprintForward   朝向方向冲刺最多 2 格（速度快, 声音大）
          8: ToggleCrouch    站<->蹲
          9: Shoot          射击（仅 FOV + 射程 + LOS 内目标）
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
            self,
            grid_size=(20, 20),
            num_team_a=20,
            num_team_b=20,
            max_steps=350,
            max_hp=3,
            map_type="suburb",  # "suburb" / "city" / "indoor"
            render_mode=None,
            seed=None,
    ):
        super().__init__()

        self.grid_h, self.grid_w = grid_size
        self.grid_size = grid_size
        self.num_team_a = num_team_a
        self.num_team_b = num_team_b
        self.n_agents_total = num_team_a + num_team_b
        self.max_steps = max_steps
        self.max_hp = max_hp
        self.map_type = map_type
        self.render_mode = render_mode

        assert self.n_agents_total <= self.grid_h * self.grid_w, "Too many agents for grid"

        self.np_random = np.random.RandomState(seed)

        # ----- 观测空间 -----
        # 每个 agent 观测 9/16 维:
        # [team_id, hp_norm, y_norm, x_norm,
        #  terrain_norm, sound_norm, orient_norm, posture_norm, alive_flag]
        obs_dim = 20
        obs_low = -np.ones((self.n_agents_total, obs_dim), dtype=np.float32)
        obs_high = np.ones((self.n_agents_total, obs_dim), dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ----- 动作空间（敌我同构） -----
        self.num_actions = 10
        # 只对 Team A 暴露动作空间；Team B 由启发式策略产生同样的动作编码
        self.action_space = spaces.MultiDiscrete([self.num_actions] * self.num_team_a)

        # ----- 战斗 & 视野参数 -----
        self.gun_range = 6  # 射程（曼哈顿距离）
        self.sound_decay = 0.8  # 脚步声每步衰减
        self.sound_strength = 1.0  # 站立普通步伐声音基准
        self.hearing_threshold = 0.6  # 敌人听脚步声的阈值
        # 新增：静态感知半径（不靠脚步声，只要无遮挡 + 敌人不在掩体，就能“感觉到”有敌人）
        self.perception_range = 6  # 可以按需要调大/调小
        # 矩形感知范围（在 agent 局部坐标系中）
        # 前方：感知更远，左右：中等，后方：很近
        self.perc_front = 6  # 朝向正前方最大感知距离
        self.perc_side = 4  # 左右最大感知宽度
        self.perc_back = 2  # 背后感知距离

        self.suppressed = None  # (N,) 是否被压制（掩护火力限制移动）

        # FOV 相关: 视野锥半角（度） & cos 值阈值
        self.fov_half_angle_deg = 60.0
        self.fov_cos = np.cos(np.deg2rad(self.fov_half_angle_deg))

        # ----- 受击后本能转向 -----
        self.hit_reorient_prob = 0.78  # 被击中后，更积极地朝来袭方向转身，减轻背身挨打
        self.hit_reorient_keep_prob = 0.22  # 降低“继续保持原朝向”的概率，让A队反应更快

        # 脚步声系数（按速度 & 姿态）
        self.sound_walk = 1.0
        self.sound_sprint = 2.0
        self.sound_crouch_factor = 0.5
        self.sound_prone_factor = 0.25
        self.sound_turn = 0.2  # 转身/调姿轻微声音

        # ----- 进攻推进奖励（potential-based shaping） -----
        # 只在每个 episode 的前 progress_steps 步生效，用于鼓励 A 队离开出生点向目标推进
        self.progress_steps = int(self.max_steps * 0.85)
        self.progress_coef = 6.0
        # 每一步会维护 A队推进分数（平均推进 + 尾部约束）
        self.prev_teamA_progress = None
        self.prev_teamA_route_progress = None
        # ===== [新增] A队每个 squad 的推进缓存 =====
        self.prev_squadA_progress = None

        # ----- 战斗 & 武器系统 -----
        # 0=SMG(冲锋), 1=Sniper(狙击), 2=Shotgun(霰弹)
        self.num_weapon_types = 3

        # 每种武器的参数（可以之后慢慢调）
        self.weapon_names = {0: "SMG", 1: "Sniper", 2: "Shotgun"}

        # 射程修正（叠加在 self.gun_range 的可视/射击范围上）
        self.weapon_range_bonus = {
            0: 0,  # SMG：中距离
            1: 3,  # Sniper：远距离
            2: -1,  # Shotgun：近距离
        }

        # 基础命中率
        self.weapon_base_hit = {
            0: 0.60,  # SMG
            1: 0.85,  # Sniper
            2: 0.75,  # Shotgun（近距离准）
        }

        # 单发伤害（HP 点数）
        self.weapon_damage = {
            0: 1,  # SMG
            1: 2,  # Sniper
            2: 1,  # Shotgun：多弹丸
        }

        # Shotgun 的弹丸数量 & 近距离加成
        self.shotgun_pellets = 3
        self.shotgun_optimal_range = 3  # 曼哈顿距离 <=3 时收益最大

        # 弹匣大小 & 装弹时间（step 数）
        self.weapon_mag_size = {
            0: 30,  # SMG
            1: 5,  # Sniper
            2: 8,  # Shotgun
        }
        self.weapon_reload_steps = {
            0: 2,
            1: 3,
            2: 2,
        }

        # 每个 agent 的武器/弹药状态（重置时初始化）
        self.agent_weapon_type = None  # (N,) int in [0,1,2]
        self.agent_ammo = None  # (N,)
        self.agent_max_ammo = None  # (N,)
        self.agent_reload_timer = None  # (N,) >0 表示还在装弹，不能射击

        # ----- 命中率 & 压制（上一条回答的东西稍微整理一下） -----
        self.base_hit_prob = 0.95  # 基础命中率（如果没指定武器则兜底）
        self.moving_shoot_penalty = 0.25
        self.suppressed_penalty = 0.20
        self.dist_decay_per_tile = 0.08
        self.dist_min_factor = 0.4

        self.multi_shot_min_damage = 1
        self.multi_shot_friendly_hp_loss = 1

        self.suppression_radius = 2
        self.suppression_steps = 1

        self.agent_armor = None  # (N,) float
        self.moved_this_step = None  # (N,) bool
        self.suppressed_timer = None  # (N,) int
        # 新增每步不动惩罚配套的工具
        self.shot_this_step = np.zeros(self.n_agents_total, dtype=bool)
        # ===== [改动2-1] 反“小动骗reward”缓存 =====
        self.prev_agent_pos = None
        self.prev_agent_target_dist = None

        self.last_combat_stats = {}

        # ----- 伤害-移动耦合（受伤减速） -----
        # hp_ratio > 0.7: 正常
        # 0.4~0.7: 冲刺变1格
        # <0.4: 冲刺禁止，只能走1格
        self.wounded_threshold = 0.4
        self.critical_threshold = 0.15

        # ----- 士气系统（0.2 ~ 1.5）-----
        self.agent_morale = None
        self.base_morale = 1.0
        self.morale_min = 0.2
        self.morale_max = 1.5
        self.morale_loss_per_friend_death = 0.08
        self.morale_loss_when_critical = 0.1
        self.morale_gain_on_kill = 0.25

        # ----- 状态变量 -----
        self.terrain = None  # (H,W) -> 0=PLAIN,1=WALL,2=COVER
        self.sound_map = None  # (H,W) -> float

        self.agents_hp = None  # (N,)
        self.agents_pos = None  # (N,2)
        self.alive_mask = None  # (N,)
        self.agents_orient = None  # (N,) 0..7
        self.agents_posture = None  # (N,) 0=站,1=蹲
        # 投掷武器 & 眩晕状态

        # 队伍级“战场认知地图”：0=未知, 1=障碍/边界, 2=可走
        self.known_map_A = None  # (H,W)
        self.known_map_B = None  # (H,W)

        # 每步破坏地形计数，用于奖励惩罚
        self.last_terrain_destroyed = 0
        self.current_step = 0

        # civilians / hostage
        self.civilians_pos = []
        self.civilians_alive = []
        self.hostage_pos = None
        self.hostage_alive = True
        self.hostage_room_mask = None

        # 入口: (y0,y1,x0,x1)
        self.entrance_A = None
        self.entrance_B = None

        # render
        self.last_shots_A = []
        self.last_shots_B = []

        # 当前步的“教范策略”信息（供分层 RL 观察）
        self.current_strategy_A = None  # team 级 posture / 背景策略
        self.current_strategy_B = None
        self.current_squad_strategies_A = None  # 每个 squad 的局部 doctrinal mode
        self.current_squad_strategies_B = None
        # 每队的小队 doctrinal objective: List[(ty, tx)]，按 default_squad_size 划分
        self.current_squad_objectives_A = None
        self.current_squad_objectives_B = None

        # A队支援边界状态
        self.current_support_target_A = None
        self.current_support_mode_A = None
        self.support_persist_timer_A = None
        self.edge_turn_margin_A = 1
        # 最近一次交火信息（给 A 队做支援参考）
        self.last_engagement_center = None  # (y, x) or None
        self.last_engagement_step = -9999
        self.last_engagement_side_hint = None  # (y, x) or None
        self.engagement_memory_steps = 15  # 保留几步，别太久

        # ===== A队残局热点收拢参数（最小增量补丁）=====
        self.endgame_hotspot_enabled = True
        self.endgame_hotspot_start_step = max(0, self.max_steps - 80)  # 只在最后80步启用
        self.endgame_hotspot_enemy_threshold = 5  # B队剩余 <= 5 才启用
        self.endgame_hotspot_group_radius = 4  # 离热点多近算“已在热点组”
        self.endgame_hotspot_assign_radius = 5  # 给目标点找可走格时的半径
        self.teamA_support_persist_steps = 3  # 支援目标至少保持几步，减少每步重算抖动
        self.teamA_same_squad_support_count = 2  # 一个接敌 squad 默认派多少人补位
        self.teamA_cross_squad_support_count = 1  # 安全 squad 默认额外派多少人增援
        self.teamA_follow_objective_blend = 0.65  # follower 目标更偏向 objective，减少后排缩在 leader 身后
        self.teamA_follow_sprint_gap = 3  # 明显掉队时允许更积极地 sprint 追上前线

        # B队巡逻
        self.patrol_routes_B = None  # 后面 reset 里初始化

    # ======================================================================
    # Gym API
    # ======================================================================

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        self.current_step = 0

        # 地图 + 入口 + 房间
        self.terrain = self._generate_terrain_and_entrances(self.map_type)
        self.sound_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        # 防止在掩体里僵持的计算僵持步数定义
        self.passive_counter = np.zeros(self.num_team_a)

        # 初始化两队的“认知地图”
        self._init_known_maps()

        # Agent 基础状态
        self.agents_hp = np.full(self.n_agents_total, self.max_hp, dtype=np.int32)
        self.alive_mask = np.ones(self.n_agents_total, dtype=bool)

        # 护甲（目前全部 1.0，可以后面按职业区分）
        self.agent_armor = np.ones(self.n_agents_total, dtype=np.float32)

        # 每步状态
        self.moved_this_step = np.zeros(self.n_agents_total, dtype=bool)
        self.suppressed_timer = np.zeros(self.n_agents_total, dtype=np.int32)

        # 士气（初始=1.0）
        self.agent_morale = np.full(self.n_agents_total, self.base_morale, dtype=np.float32)

        # 武器分配：这里简单给 A 队三种均匀分布，B 队也一样（你可以按需要改）
        # 全员统一使用 SMG
        self.agent_weapon_type = np.zeros(self.n_agents_total, dtype=np.int32)  # 0 = SMG

        # 弹药 / 装弹计时
        self.agent_max_ammo = np.zeros(self.n_agents_total, dtype=np.int32)
        self.agent_ammo = np.zeros(self.n_agents_total, dtype=np.int32)
        self.agent_reload_timer = np.zeros(self.n_agents_total, dtype=np.int32)
        for i in range(self.n_agents_total):
            wt = int(self.agent_weapon_type[i])
            mag = self.weapon_mag_size[wt]
            self.agent_max_ammo[i] = mag
            self.agent_ammo[i] = mag  # 初始满弹
            self.agent_reload_timer[i] = 0

        # 清空上一回合战斗统计
        self.last_combat_stats = {}
        # 清空迎敌记忆
        self.last_engagement_center = None
        self.last_engagement_step = -9999
        self.last_engagement_side_hint = None
        self.current_strategy_A = None
        self.current_strategy_B = None
        self.current_squad_strategies_A = None
        self.current_squad_strategies_B = None
        self.current_squad_objectives_A = None
        self.current_squad_objectives_B = None

        # 增加A队支援队友奖励
        self.prev_support_dist = np.full(self.num_team_a, -1.0, dtype=np.float32)
        self.prev_facing_score = np.zeros(self.num_team_a, dtype=np.float32)
        self.current_support_target_A = [None] * self.num_team_a
        self.current_support_mode_A = np.zeros(self.num_team_a, dtype=np.int8)
        self.support_persist_timer_A = np.zeros(self.num_team_a, dtype=np.int32)
        # ===== 训练日志用：每局累计的 reward 分解 =====

        # ===== [改动1] 记录“上一时刻是否处于贴边且朝外看” =====
        self.prev_outward_edge_A = np.zeros(self.num_team_a, dtype=np.int8)
        # ===== [新增] 记录本步 A队原始动作，供 edge 行为惩罚使用 =====
        self.last_action_A = np.zeros(self.num_team_a, dtype=np.int64)

        self.ep_stall_penalty = 0.0
        self.ep_edge_penalty = 0.0

        self.suppressed = np.zeros(self.n_agents_total, dtype=bool)

        # 每步破坏计数
        self.last_terrain_destroyed = 0

        # 双入口出生
        spawn_A = self._get_walkable_cells_in_rect(*self.entrance_A)
        spawn_B = self._get_walkable_cells_in_rect(*self.entrance_B)
        self.np_random.shuffle(spawn_A)
        self.np_random.shuffle(spawn_B)
        assert len(spawn_A) >= self.num_team_a
        assert len(spawn_B) >= self.num_team_b

        pos_A = self._sample_spread_positions(spawn_A, self.num_team_a, min_l1=2)
        pos_B = self._sample_spread_positions(spawn_B, self.num_team_b, min_l1=3)
        self.agents_pos = np.array(pos_A + pos_B, dtype=np.int32)

        # 朝向 & 姿态初始化
        self.agents_posture = np.zeros(self.n_agents_total, dtype=np.int32)  # 全部站立

        # 计算 A 队入口中心（目标点 for B队）
        ay0, ay1, ax0, ax1 = self.entrance_A
        a_center_y = (ay0 + ay1) / 2.0
        a_center_x = (ax0 + ax1) / 2.0

        # 计算 B 队入口中心（目标点 for A队）
        by0, by1, bx0, bx1 = self.entrance_B
        b_center_y = (by0 + by1) / 2.0
        b_center_x = (bx0 + bx1) / 2.0

        self.agents_orient = np.zeros(self.n_agents_total, dtype=np.int32)

        for i in range(self.n_agents_total):
            y, x = self.agents_pos[i]

            # A队：仍然默认朝向 B 队入口中心
            if i < self.num_team_a:
                target_y, target_x = b_center_y, b_center_x
                dy = target_y - y
                dx = target_x - x

                if abs(dy) < 1 and abs(dx) < 1:
                    ori = self.np_random.randint(0, 8)
                else:
                    angle = np.arctan2(dy, dx)
                    angle = (angle + np.pi / 2) % (2 * np.pi)
                    ori = int(np.round(angle / (np.pi / 4))) % 8
                    offset = self.np_random.choice([-1, 0, 1], p=[0.20, 0.60, 0.20])
                    ori = (ori + offset) % 8

                self.agents_orient[i] = ori
                continue

            # B队：降低“开局就正对A队”的比例，保留更多随机朝向，避免显得过强/过聪明
            if self.np_random.rand() < 0.35:
                target_y, target_x = self._get_b_spawn_target(y, x, a_center_y, a_center_x)
                dy = target_y - y
                dx = target_x - x

                if abs(dy) < 1 and abs(dx) < 1:
                    ori = self.np_random.randint(0, 8)
                else:
                    angle = np.arctan2(dy, dx)
                    angle = (angle + np.pi / 2) % (2 * np.pi)
                    ori = int(np.round(angle / (np.pi / 4))) % 8
                    offset = self.np_random.choice(
                        [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                        p=[0.08, 0.12, 0.15, 0.10, 0.10, 0.10, 0.15, 0.12, 0.08],
                    )
                    ori = (ori + offset) % 8

                self.agents_orient[i] = ori
            else:
                self.agents_orient[i] = self.np_random.randint(0, 8)

        # 平民 & 人质
        self._spawn_civilians_and_hostage()

        # B队巡逻
        self._init_patrol_routes()

        # B队分批出发：在小图上重新拉大错峰间隔，避免多个 squad
        # 过快同时进入前沿而重新拼成火力网。
        squad_size = self._get_effective_squad_size(TEAM_B)
        num_b_squads = max(1, int(np.ceil(self.num_team_b / squad_size)))

        self.b_squad_launch_step = []
        base_gap = max(8, int(round(min(self.grid_h, self.grid_w) * 0.42)))
        jitter = 3

        for s_idx in range(num_b_squads):
            launch = 2 + s_idx * base_gap + self.np_random.randint(0, jitter + 1)
            self.b_squad_launch_step.append(int(launch))

        # 初始位置的信息更新到认知地图（开局就知道自己脚下有哪些格子）
        self._update_team_knowledge(TEAM_A)
        self._update_team_knowledge(TEAM_B)

        # 初始化推进奖励的距离基准（平均推进 + 尾部约束）
        self.prev_teamA_progress = self._teamA_progress_score()
        self.prev_teamA_route_progress = self._teamA_route_progress_score()
        # ===== [新增] 初始化每个 squad 的推进基准 =====
        self.prev_squadA_progress = self._teamA_squad_progress_scores()
        # 发现敌人奖励
        self.teamA_first_contact = False
        # ===== [改动2-2] 初始化单兵推进缓存，用于识别“小动骗reward” =====
        self.prev_agent_pos = self.agents_pos[: self.num_team_a].copy()
        self.prev_agent_target_dist = np.full(self.num_team_a, -1.0, dtype=np.float32)

        target_y, target_x = self._get_teamA_progress_target()
        for i in range(self.num_team_a):
            if not self.alive_mask[i]:
                continue
            y, x = self.agents_pos[i]
            self.prev_agent_target_dist[i] = abs(y - target_y) + abs(x - target_x)

        # 新增：中心偏好初始化
        H, W = self.grid_h, self.grid_w
        center_y = (H - 1) / 2.0
        center_x = (W - 1) / 2.0
        alive_A_pos = [self.agents_pos[i] for i in range(self.num_team_a) if self.alive_mask[i]]
        if alive_A_pos:
            a_mean_y = np.mean([p[0] for p in alive_A_pos])
            a_mean_x = np.mean([p[1] for p in alive_A_pos])
            self.prev_dist_to_center = abs(a_mean_y - center_y) + abs(a_mean_x - center_x)
        else:
            self.prev_dist_to_center = 0.0
        # 中心结束

        # 先为 A / B 预计算一遍教范策略，保证 reset 后第一帧 obs 就有 squad objective 信息
        try:
            _ = self._heuristic_team_actions(TEAM_A)
        except Exception:
            pass

        try:
            _ = self._heuristic_team_actions(TEAM_B)
        except Exception:
            pass

        obs = self._get_obs()
        info = {"map_type": self.map_type}
        return obs, info

    def _has_shootable_enemy(self, i: int) -> bool:
        if not self.alive_mask[i]:
            return False
        sy, sx = self.agents_pos[i]
        team_i = TEAM_A if i < self.num_team_a else TEAM_B

        if self.agent_weapon_type is not None:
            wtype = int(self.agent_weapon_type[i])
        else:
            wtype = 0
        max_range = self.gun_range + self.weapon_range_bonus.get(wtype, 0)  # <<< 修改

        for j in range(self.n_agents_total):
            if not self.alive_mask[j]:
                continue
            team_j = TEAM_A if j < self.num_team_a else TEAM_B
            if team_j == team_i:
                continue
            jy, jx = self.agents_pos[j]
            dist = abs(jy - sy) + abs(jx - sx)
            if (
                    dist <= max_range
                    and self._within_fov(i, jy, jx)
                    and self._has_line_of_sight(sy, sx, jy, jx)
            ):
                return True
        return False

    def step(self, action):
        """
        action: (num_team_a,) MultiDiscrete，每个元素 0..9
        """
        self.current_step += 1
        suppress_opening_fire = self.current_step < 2

        # 每步开始：重置移动标记
        if self.moved_this_step is not None:
            self.moved_this_step[:] = False
        # 新增：每步开始重置射击标记（和上面完全一样风格）
        if self.shot_this_step is not None:
            self.shot_this_step[:] = False  # 或写成 self.shot_this_step.fill(False)

        self.last_terrain_destroyed = 0  # 每步重置地形破坏计数
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (self.num_team_a,):
            raise ValueError(f"Expected action shape {(self.num_team_a,)}, got {action.shape}")
        if suppress_opening_fire:
            action = action.copy()
            action[action == 9] = 0

        # ===== [新增] 缓存本步真正执行的 A队动作，供 reward 端判断“外向坏动作” =====
        self.last_action_A = action.copy()
        # A队：靠近边界且朝外看时，做保护性强制转向（比B队更灵活）
        # for i in range(self.num_team_a):
        #   if not self.alive_mask[i]:
        #      continue
        # action[i] = self._force_turn_from_edge_for_A(i, int(action[i]))
        prev_hp = self.agents_hp.copy()

        # 先根据当前状态计算“掩护火力压制”
        self._compute_suppression()

        # 声音衰减
        self.sound_map *= self.sound_decay

        # 先用启发式为 A 队计算一份“教范策略”（只用于暴露给 RL，不执行动作）
        try:
            _ = self._heuristic_team_actions(TEAM_A)
        except Exception:
            # 如果由于某些原因失败，不影响主流程
            pass

        support_targets_A = getattr(self, "current_support_target_A", None)

        for i in range(self.num_team_a):
            if not self.alive_mask[i]:
                self.prev_support_dist[i] = -1.0
                self.prev_facing_score[i] = 0.0
                continue

            if self._agent_can_see_enemy(i):
                self.prev_support_dist[i] = -1.0
                self.prev_facing_score[i] = 0.0
                continue

            tgt = None
            # 先沿用原有支援目标
            if support_targets_A is not None and i < len(support_targets_A):
                tgt = support_targets_A[i]

            # 如果原支援目标没有，再用残局热点目标
            if tgt is None:
                tgt = self._get_endgame_hotspot_target_for_agent_A(i)

            if tgt is None:
                self.prev_support_dist[i] = -1.0
                self.prev_facing_score[i] = 0.0
                continue

            y, x = self.agents_pos[i]
            ty, tx = tgt
            self.prev_support_dist[i] = abs(y - ty) + abs(x - tx)
            face_tgt = tgt
            fight_center_hint, _ = self._get_recent_engagement_hint_for_A()
            if fight_center_hint is not None:
                face_tgt = fight_center_hint
            else:
                fallback = self._get_teamB_spawn_fallback_target_for_agent_A(i)
                if fallback is not None:
                    face_tgt = fallback

            self.prev_facing_score[i] = self._facing_score_towards(i, face_tgt[0], face_tgt[1])

        # 缓存 A 队执行动作前的射击态势，reward 使用这个快照做归因。
        # 结算后敌人可能死亡/位移，若事后再判定 shootable 会让射击奖励失真。
        self.pre_visible_enemy_A = np.zeros(self.num_team_a, dtype=bool)
        self.pre_shootable_enemy_A = np.zeros(self.num_team_a, dtype=bool)
        for i in range(self.num_team_a):
            if not self.alive_mask[i]:
                continue
            self.pre_visible_enemy_A[i] = self._agent_can_see_enemy(i)
            self.pre_shootable_enemy_A[i] = self._has_shootable_enemy(i)

        # Team A 行动（由 RL 或外部控制给出动作）
        shots_A, shoot_good_A, shoot_bad_A = self._apply_team_actions(TEAM_A, action)

        # Team B 启发式，产生相同动作空间的动作编码（对称能力）
        actions_B = self._heuristic_team_actions(TEAM_B)
        if suppress_opening_fire:
            actions_B = np.asarray(actions_B, dtype=np.int64).copy()
            actions_B[actions_B == 9] = 0

        # B队推进过程中保留一点迟钝感，但不再高频打断移动，
        # 否则容易在出生区和狭口堆成固定火力点。
        for local_i in range(len(actions_B)):
            gi = self.num_team_a + local_i
            if not self.alive_mask[gi]:
                continue

            # 只在没看到敌人时改
            if self._agent_can_see_enemy(gi):
                continue

            # 只对移动类动作处理
            if actions_B[local_i] in (3, 4, 5, 6, 7):
                if self.current_step < 120:
                    scan_prob = 0.12
                else:
                    scan_prob = 0.20
                if self.np_random.rand() < scan_prob:
                    actions_B[local_i] = self._random_scan_action_B()

        shots_B, _, _ = self._apply_team_actions(TEAM_B, actions_B)

        self.last_shots_A = list(shots_A)
        self.last_shots_B = list(shots_B)

        # 射击结算（含误伤 / 人质）
        self._resolve_shooting(shots_A, shots_B)
        # 更新交火锚点
        # self._resolve_shooting(shots_A, shots_B)
        self._update_last_engagement_for_A(shots_A, shots_B)

        # ===== 在这里插入参与度更新 =====
        # ===== [改动3] 更严格的参与度更新：禁止靠“小动”清空 passive_counter =====
        target_y, target_x = self._get_teamA_progress_target()

        for i in range(self.num_team_a):
            if not self.alive_mask[i]:
                continue

            engaged = False
            y, x = self.agents_pos[i]
            py, px = self.prev_agent_pos[i]

            # 1) 真接敌：附近有敌人
            near_enemy = False
            for j in range(self.num_team_a, self.n_agents_total):
                if not self.alive_mask[j]:
                    continue
                ey, ex = self.agents_pos[j]
                if abs(y - ey) + abs(x - ex) <= 5:
                    near_enemy = True
                    break
            if near_enemy:
                engaged = True

            # 2) 真开火：有可打目标时开火
            if self.shot_this_step[i] and self._has_shootable_enemy(i):
                engaged = True

            # 3) 真推进：明显朝总目标缩短距离（至少2格）
            old_prog_d = self.prev_agent_target_dist[i]
            new_prog_d = abs(y - target_y) + abs(x - target_x)
            prog_delta = old_prog_d - new_prog_d
            if prog_delta >= 1:
                engaged = True

            # 4) 真支援：明显朝支援目标缩短距离（至少2格）
            support_tgt = None
            if getattr(self, "current_support_target_A", None) is not None:
                support_tgt = self.current_support_target_A[i]

            if support_tgt is not None:
                ty, tx = support_tgt
                old_sup_d = abs(py - ty) + abs(px - tx)
                new_sup_d = abs(y - ty) + abs(x - tx)
                if old_sup_d - new_sup_d >= 1:
                    engaged = True

            # 5) 小动识别：位置变了，但没有真推进/真支援/真接敌
            moved_l1 = abs(y - py) + abs(x - px)
            fake_move = (moved_l1 > 0 and not engaged)

            # 6) 放宽：只要有较明显机动，就不要视为完全消极
            #    这样前期换路/补位、残局扫图时，不会因为“没立刻缩短目标距离”被压死
            if (not engaged) and moved_l1 >= 2:
                engaged = True

            if engaged:
                self.passive_counter[i] = 0
            else:
                # 放软 fake_move 惩罚，避免搜索和找路被过度打压
                self.passive_counter[i] += 1

            # 更新缓存
            self.prev_agent_pos[i] = np.array([y, x], dtype=np.int32)
            self.prev_agent_target_dist[i] = new_prog_d

        # 平民随机移动
        self._step_civilians()

        # 奖励 + 终止
        reward, terminated, info = self._compute_reward_and_terminated(prev_hp)
        info["shoot_good_A"] = int(shoot_good_A)
        info["shoot_bad_A"] = int(shoot_bad_A)
        move_attempt_mask = (self.last_action_A >= 3) & (self.last_action_A <= 7)
        move_attempts = int(np.sum(move_attempt_mask))
        move_successes = int(np.sum(self.moved_this_step[: self.num_team_a] & move_attempt_mask))
        move_failures = max(0, move_attempts - move_successes)
        info["move_attempt_ratio_A"] = float(move_attempts / max(1, self.num_team_a))
        info["move_fail_ratio_A"] = float(move_failures / max(1, move_attempts))

        truncated = False
        if self.current_step >= self.max_steps and not terminated:
            truncated = True
            info.setdefault("result", "timeout")
            # ===== timeout 消极惩罚（只在超时且尚未终局时触发）=====
            # 这里只补“超时失败成本”，不改前面的 reward 主体结构
            survivors = int(np.sum(self.alive_mask[: self.num_team_a]))
            enemies_left = int(np.sum(self.alive_mask[self.num_team_a:]))

            timeout_penalty = 80.0 + 5.0 * enemies_left + 2.0 * survivors

            reward -= timeout_penalty
            info["timeout_penalty"] = float(timeout_penalty)

        # 每一步用最新位置和矩形感知更新两队的认知地图
        self._update_team_knowledge(TEAM_A)
        self._update_team_knowledge(TEAM_B)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # ======================================================================
    # 地图 & 入口 & 房间
    # ======================================================================

    def _generate_terrain_and_entrances(self, map_type: str):
        H, W = self.grid_h, self.grid_w
        terrain = np.full((H, W), PLAIN, dtype=np.int32)

        # 左右入口区域
        entrance_h = max(3, H // 4)
        entrance_w = max(2, W // 5)

        # A队出生区给出富余面积，避免入口格子数几乎等于人数，导致开局无法横向展开。
        target_a_cells = min(H * W, max(self.num_team_a + 12, 30))
        entrance_a_h = min(H, max(entrance_h + 1, H // 3))
        entrance_a_w = min(W, max(entrance_w + 2, int(np.ceil(target_a_cells / max(1, entrance_a_h)))))
        self.entrance_A = (0, entrance_a_h, 0, entrance_a_w)

        # B队出生区：保证能稳定容纳 30 人即可，不再额外放太大
        target_b_cells = min(H * W, max(30, self.num_team_b))
        entrance_b_h = min(H, 8)
        entrance_b_w = min(W, int(np.ceil(target_b_cells / entrance_b_h)))

        self.entrance_B = (H - entrance_b_h, H, W - entrance_b_w, W)

        self.hostage_room_mask = np.zeros((H, W), dtype=bool)
        self.hostage_pos = None
        self.hostage_alive = True

        if map_type == "suburb":
            # 空旷 + 少量障碍
            num_blocks = max(1, (H * W) // 80)
            for _ in range(num_blocks):
                bh = self.np_random.randint(2, max(3, H // 5))
                bw = self.np_random.randint(2, max(3, W // 5))
                y0 = self.np_random.randint(0, max(1, H - bh))
                x0 = self.np_random.randint(0, max(1, W - bw))
                terrain[y0:y0 + bh, x0:x0 + bw] = WALL

            num_covers = (H * W) // 25
            for _ in range(num_covers):
                y = self.np_random.randint(0, H)
                x = self.np_random.randint(0, W)
                if terrain[y, x] == PLAIN:
                    terrain[y, x] = COVER

        elif map_type == "city":
            # 密集建筑 + 道路 + 掩体
            num_blocks = max(5, (H * W) // 40)
            for _ in range(num_blocks):
                bh = self.np_random.randint(3, max(4, H // 4))
                bw = self.np_random.randint(3, max(4, W // 4))
                y0 = self.np_random.randint(0, max(1, H - bh))
                x0 = self.np_random.randint(0, max(1, W - bw))
                terrain[y0:y0 + bh, x0:x0 + bw] = WALL

            # 十字路口
            for _ in range(2):
                y = self.np_random.randint(0, H)
                terrain[y, :] = PLAIN
            for _ in range(2):
                x = self.np_random.randint(0, W)
                terrain[:, x] = PLAIN

            num_covers = (H * W) // 15
            for _ in range(num_covers):
                y = self.np_random.randint(0, H)
                x = self.np_random.randint(0, W)
                if terrain[y, x] == PLAIN:
                    terrain[y, x] = COVER

        elif map_type == "indoor":
            # 九宫格房间 + 中央走廊 + 人质房间
            terrain[:, :] = WALL
            room_h = H // 3
            room_w = W // 3

            for ry in range(3):
                for rx in range(3):
                    y0 = ry * room_h + 1
                    x0 = rx * room_w + 1
                    y1 = min(H - 2, y0 + room_h - 2)
                    x1 = min(W - 2, x0 + room_w - 2)
                    terrain[y0:y1, x0:x1] = PLAIN

            mid_y = H // 2
            mid_x = W // 2
            terrain[mid_y, :] = PLAIN
            terrain[:, mid_x] = PLAIN

            # 人质房间
            hr_ry = self.np_random.randint(0, 3)
            hr_rx = self.np_random.randint(0, 3)
            hr_y0 = hr_ry * room_h + 1
            hr_x0 = hr_rx * room_w + 1
            hr_y1 = min(H - 2, hr_y0 + room_h - 2)
            hr_x1 = min(W - 2, hr_x0 + room_w - 2)
            self.hostage_room_mask[hr_y0:hr_y1, hr_x0:hr_x1] = True

        # 确保入口区域可通行。A队额外清出展开区，避免出生点附近密集掩体/墙体堵住前期推进。
        self._clear_entrance_area(terrain, self.entrance_A, margin=3)
        self._clear_entrance_area(terrain, self.entrance_B, margin=1)
        return terrain

    def _clear_entrance_area(self, terrain, rect, margin: int = 0):
        y0, y1, x0, x1 = rect
        yy0 = max(0, y0 - margin)
        yy1 = min(self.grid_h, y1 + margin)
        xx0 = max(0, x0 - margin)
        xx1 = min(self.grid_w, x1 + margin)
        terrain[yy0:yy1, xx0:xx1] = PLAIN

    def _get_walkable_cells_in_rect(self, y0, y1, x0, x1):
        cells = []
        for y in range(y0, y1):
            for x in range(x0, x1):
                if 0 <= y < self.grid_h and 0 <= x < self.grid_w:
                    if self.terrain[y, x] != WALL:  # if terrain := self.terrain[y, x] != WALL:
                        cells.append((y, x))
        return cells

    def _sample_spread_positions(self, cells, n, min_l1=3):
        """从候选格中采样，尽量保证出生点不要过于密集。"""
        cells = list(cells)
        self.np_random.shuffle(cells)

        chosen = []
        for y, x in cells:
            if all(abs(y - cy) + abs(x - cx) >= min_l1 for cy, cx in chosen):
                chosen.append((y, x))
                if len(chosen) == n:
                    return chosen

        for relax in (2, 1, 0):
            for y, x in cells:
                if (y, x) in chosen:
                    continue
                if all(abs(y - cy) + abs(x - cx) >= relax for cy, cx in chosen):
                    chosen.append((y, x))
                    if len(chosen) == n:
                        return chosen

        return chosen[:n]

    def _get_b_spawn_target(self, y, x, a_center_y, a_center_x):
        """B 队出生朝向只保留很弱的“向战区内侧”偏好，其余尽量随机。"""
        by0, by1, bx0, bx1 = self.entrance_B
        cy = 0.5 * (by0 + by1)
        cx = 0.5 * (bx0 + bx1)

        inward_y = cy - self.np_random.uniform(0.05 * self.grid_h, 0.30 * self.grid_h)
        side_bias = self.np_random.uniform(-0.40 * self.grid_w, 0.40 * self.grid_w)
        tx = cx + side_bias
        ty = inward_y + self.np_random.uniform(-0.20 * self.grid_h, 0.20 * self.grid_h)

        ty = float(np.clip(ty, 1, self.grid_h - 2))
        tx = float(np.clip(tx, 1, self.grid_w - 2))
        return ty, tx

    def _spawn_civilians_and_hostage(self):
        H, W = self.grid_h, self.grid_w
        self.civilians_pos = []
        self.civilians_alive = []

        if self.map_type == "city":
            num_civ = (H * W) // 30
            cells = [(y, x) for y in range(H) for x in range(W) if self.terrain[y, x] == PLAIN]
            self.np_random.shuffle(cells)
            for pos in cells[:num_civ]:
                self.civilians_pos.append(pos)
                self.civilians_alive.append(True)

        if self.map_type == "indoor":
            ys, xs = np.where(self.hostage_room_mask & (self.terrain != WALL))
            assert len(ys) > 0, "Hostage room has no free cell"
            k = self.np_random.randint(0, len(ys))
            self.hostage_pos = (int(ys[k]), int(xs[k]))
            self.hostage_alive = True

    def _init_patrol_routes(self):
        """
        B队巡逻路线：
        - 从右下出生区外的浅层警戒带开始
        - 前两个点只在B半场前沿巡逻，避免在出生区抱团，也避免过早压到中场
        - 每条路线先做 clip，再找最近可走点
        """
        H, W = self.grid_h, self.grid_w
        by0, by1, bx0, bx1 = self.entrance_B
        cy = (by0 + by1) // 2
        cx = (bx0 + bx1) // 2

        spawn_front = max(3, H // 6)
        depth1 = max(spawn_front + 1, H // 4)
        depth2 = max(depth1 + 2, int(H * 0.38))
        depth3 = max(depth2 + 2, int(H * 0.48))

        side1 = max(5, W // 5)
        side2 = max(8, W // 3)
        side3 = max(10, int(W * 0.42))

        early_band_y = max(2, cy - spawn_front)
        mid_band_y = max(2, cy - depth1)
        late_band_y = max(2, cy - depth2)
        deep_band_y = max(2, cy - depth3)

        raw_routes = [
            [
                (early_band_y, cx - max(2, side1 // 2)),
                (mid_band_y, cx - side1),
                (late_band_y, cx - side2),
                (deep_band_y, cx - side3),
            ],
            [
                (early_band_y, cx),
                (mid_band_y, cx - 1),
                (late_band_y, cx + 1),
                (deep_band_y, cx - max(2, side1 // 2)),
            ],
            [
                (early_band_y, cx + max(2, side1 // 2)),
                (mid_band_y, cx + side1),
                (late_band_y, cx + side2),
                (deep_band_y, cx + side3),
            ],
        ]

        self.patrol_routes_B = []
        for route in raw_routes:
            fixed = []
            for y, x in route:
                y = int(np.clip(y, 1, H - 2))
                x = int(np.clip(x, 1, W - 2))
                y, x = self._find_nearest_free_cell_for_team(
                    y, x, TEAM_B, ignore_idx=None, max_r=8
                )
                fixed.append((y, x))
            self.patrol_routes_B.append(fixed)

    def _get_team_b_early_limit_y(self) -> int:
        """B队前期只允许活动到B半场前沿，不早压到中场。"""
        by0, by1, _, _ = self.entrance_B
        mid_y = self.grid_h // 2
        spawn_cy = (by0 + by1) // 2
        limit_y = max(2, int(round(0.65 * mid_y + 0.35 * spawn_cy)))
        return min(self.grid_h - 2, limit_y)

    def _get_team_b_forward_hold_point(self, s_idx: int, leader_idx: int = None):
        """
        给B队一个出生区外的浅层警戒点。
        目标是尽快离开出生区，展开成松散前沿，而不是守门抱团。
        """
        route = self.patrol_routes_B[s_idx % len(self.patrol_routes_B)]
        target_y, target_x = route[0]
        target_y = min(target_y, self._get_team_b_early_limit_y())
        target_y = int(np.clip(target_y + self.np_random.randint(-1, 2), 1, self.grid_h - 2))
        target_x = int(np.clip(target_x + self.np_random.randint(-1, 2), 1, self.grid_w - 2))
        return self._find_low_density_cell_for_team(
            target_y, target_x, TEAM_B, ignore_idx=leader_idx, max_r=7, density_radius=2
        )

    # ======================================================================
    # 观测
    # ======================================================================

    # 新增维度的工具函数
    def _update_last_engagement_for_A(self, shots_A, shots_B):
        """
    基于“真实交火”或“A 队真实可见敌人”更新 A 队共享战场锚点。
    目标：
      - 让 A 知道交火大概发生在哪
      - 给一个可选侧翼点
      - 看见敌人即可形成热点，不强制要求残局阶段必须先命中
        """
        points = []

        # A队开火命中的目标位置
        for shooter_idx, ttype, tidx in shots_A:
            if ttype == "enemy" and tidx is not None and 0 <= tidx < self.n_agents_total:
                ty, tx = self.agents_pos[tidx]
                points.append((float(ty), float(tx)))

        # B队开火时，记录射手位置，避免热点落在 A 队受击位置后方
        for shooter_idx, ttype, tidx in shots_B:
            if 0 <= shooter_idx < self.n_agents_total and self.alive_mask[shooter_idx]:
                ty, tx = self.agents_pos[shooter_idx]
                points.append((float(ty), float(tx)))

        # 如果本步没有命中/被命中，但 A 队已经真实看见敌人，也更新热点。
        # 这里不使用全图敌人，只使用 A 队当前可见敌人，避免普通阶段开图。
        if not points:
            visible_enemies = self._get_team_visible_enemies(TEAM_A)
            for j in visible_enemies:
                if self.alive_mask[j]:
                    ty, tx = self.agents_pos[j]
                    points.append((float(ty), float(tx)))

        if not points:
            return

        cy = float(np.mean([p[0] for p in points]))
        cx = float(np.mean([p[1] for p in points]))
        self.last_engagement_center = (cy, cx)
        self.last_engagement_step = self.current_step

        # 用 B 队侧的几何信息构造侧翼点，避免把提示点拉回 A 队后方。
        if self.entrance_B is not None:
            by0, by1, bx0, bx1 = self.entrance_B
            base_bx = float((bx0 + bx1) / 2.0)
        else:
            self.last_engagement_side_hint = (cy, cx)
            return

        lateral_shift = float(np.clip(cx - base_bx, -3.0, 3.0))
        if abs(lateral_shift) < 1.0:
            lateral_shift = 2.0 if cx <= base_bx else -2.0

        fy = int(np.clip(round(cy), 0, self.grid_h - 1))
        fx = int(np.clip(round(cx + lateral_shift), 0, self.grid_w - 1))

        # 如果落在墙上，退回交火中心
        if self.terrain[fy, fx] == WALL:
            fy, fx = int(round(cy)), int(round(cx))

        self.last_engagement_side_hint = (float(fy), float(fx))

    def _get_recent_engagement_hint_for_A(self):
        """
    返回最近交火中心和侧翼点；过期则返回 None。
        """
        if self.last_engagement_center is None:
            return None, None
        if (self.current_step - self.last_engagement_step) > self.engagement_memory_steps:
            return None, None
        return self.last_engagement_center, self.last_engagement_side_hint

    def _get_teamB_spawn_fallback_target_for_agent_A(self, i: int = None):
        """
        A队残局没有交火热点时的兜底目标：B队出生点中心。
        不读取真实 B 队当前位置，只使用地图入口信息，避免“没热点 -> 没目标”的死循环。
        """
        if self.entrance_B is None:
            return None

        y0, y1, x0, x1 = self.entrance_B
        ty = int(np.clip(round((y0 + y1) / 2.0), 1, self.grid_h - 2))
        tx = int(np.clip(round((x0 + x1) / 2.0), 1, self.grid_w - 2))
        ignore_idx = i if i is not None and 0 <= i < self.num_team_a else None
        ty, tx = self._find_nearest_free_cell_for_team(
            ty,
            tx,
            TEAM_A,
            ignore_idx=ignore_idx,
            max_r=int(getattr(self, "endgame_hotspot_assign_radius", 7)),
        )
        return (ty, tx)

    def _get_nearest_visible_enemy_for_agent(self, i: int):
        if not self.alive_mask[i]:
            return None

        team_i = TEAM_A if i < self.num_team_a else TEAM_B
        sy, sx = self.agents_pos[i]

        wt = int(self.agent_weapon_type[i]) if self.agent_weapon_type is not None else 0
        eff_range = self.gun_range + self.weapon_range_bonus.get(wt, 0)

        best_j = None
        best_d = None

        for j in range(self.n_agents_total):
            if not self.alive_mask[j]:
                continue
            team_j = TEAM_A if j < self.num_team_a else TEAM_B
            if team_j == team_i:
                continue

            ty, tx = self.agents_pos[j]
            dist = abs(ty - sy) + abs(tx - sx)

            if (dist <= eff_range and
                    self._within_fov(i, ty, tx) and
                    self._has_line_of_sight(sy, sx, ty, tx)):
                if best_d is None or dist < best_d:
                    best_d = dist
                    best_j = j

        return best_j

    def _get_squad_index_for_agent_A(self, i: int):
        if i < 0 or i >= self.num_team_a:
            return 0

        squad_size = getattr(self, "default_squad_size", 5)
        if squad_size <= 0:
            squad_size = 5

        return i // squad_size

    def _get_effective_squad_size(self, team_id: int) -> int:
        """
        当前保持 A/B 使用同一套基础编组大小。
        B队的分散主要通过 follower 逻辑控制，而不是改 squad 人数。
        """
        squad_size = getattr(self, "default_squad_size", 5)
        if squad_size <= 0:
            squad_size = 5
        return squad_size

    def _endgame_hotspot_mode_on(self) -> bool:
        """
        只在残局启用“热点收拢推进”：
          1) 开关打开
          2) 到了最后一段步数
          3) B队剩余人数足够少
        注意：这里不要求已经有交火热点；没有热点时，目标层会 fallback 到 B 队出生点。
        """
        if not getattr(self, "endgame_hotspot_enabled", True):
            return False

        if self.current_step < getattr(self, "endgame_hotspot_start_step", max(0, self.max_steps - 80)):
            return False

        enemies_left = int(np.sum(self.alive_mask[self.num_team_a:]))
        if enemies_left > getattr(self, "endgame_hotspot_enemy_threshold", 6):
            return False

        return enemies_left > 0

    def _get_endgame_hotspot_target_for_agent_A(self, i: int):
        """
        给 A队单个 agent 返回残局热点推进目标。
        只用于 RL 观察 / reward shaping，不直接改动作。
        返回:
          None 或 (ty, tx)
        """
        if i < 0 or i >= self.num_team_a:
            return None
        if not self.alive_mask[i]:
            return None
        if not self._endgame_hotspot_mode_on():
            return None
        if self._agent_can_see_enemy(i):
            return None

        fight_center, flank_hint = self._get_recent_engagement_hint_for_A()

        if fight_center is None:
            fallback = self._get_teamB_spawn_fallback_target_for_agent_A(i)
            if fallback is None:
                return None
            fight_center = (float(fallback[0]), float(fallback[1]))

        if flank_hint is None:
            flank_hint = fight_center

        cy, cx = fight_center
        fy, fx = flank_hint

        # 中路推进点
        my = int(round(0.5 * cy + 0.5 * fy))
        mx = int(round(0.5 * cx + 0.5 * fx))

        candidates = [
            (int(round(cy)), int(round(cx))),  # 正面
            (int(round(fy)), int(round(fx))),  # 侧翼
            (int(round(my)), int(round(mx))),  # 中路
        ]

        y, x = self.agents_pos[i]

        # 选离自己最近的那个热点推进点
        best_tgt = None
        best_d = None
        for ty, tx in candidates:
            d = abs(y - ty) + abs(x - tx)
            if best_d is None or d < best_d:
                best_d = d
                best_tgt = (ty, tx)

        ty, tx = best_tgt
        ty = int(np.clip(ty, 1, self.grid_h - 2))
        tx = int(np.clip(tx, 1, self.grid_w - 2))
        ty, tx = self._find_nearest_free_cell_for_team(
            ty,
            tx,
            TEAM_A,
            ignore_idx=i,
            max_r=int(getattr(self, "endgame_hotspot_assign_radius", 7)),
        )
        return (ty, tx)

    def _apply_endgame_hotspot_support_A(
            self,
            squads,
            idx_start,
            can_see_mask,
            squad_centers,
            squad_fire_anchors,
            engaged_squads,
    ):
        """
        只在残局启用：
        - 保留前面原有的 squad内 / 跨squad 支援结果
        - 再用“热点收拢”覆盖掉尚未接敌、且还没有支援目标的 A队成员
        - 不去新造动作，只复用 current_support_target_A / current_support_mode_A
        """
        if not self._endgame_hotspot_mode_on():
            return

        fight_center, flank_hint = self._get_recent_engagement_hint_for_A()
        if fight_center is None:
            fallback = self._get_teamB_spawn_fallback_target_for_agent_A(None)
            if fallback is None:
                return
            fight_center = (float(fallback[0]), float(fallback[1]))

        cy, cx = fight_center
        if flank_hint is None:
            flank_hint = fight_center
        fy, fx = flank_hint

        # 第三个点：中路推进点
        my = int(round(0.5 * cy + 0.5 * fy))
        mx = int(round(0.5 * cx + 0.5 * fx))

        hotspot_points = [
            (int(round(cy)), int(round(cx))),  # 正面
            (int(round(fy)), int(round(fx))),  # 侧翼
            (int(round(my)), int(round(mx))),  # 中路
        ]

        # 哪些 squad 已经在热点附近 / 本来就在交火，就不强行重定向
        hotspot_group_radius = int(getattr(self, "endgame_hotspot_group_radius", 5))
        hotspot_squads = set(engaged_squads)

        for s_idx, center in enumerate(squad_centers):
            if center is None:
                continue
            sy, sx = center
            d = abs(sy - cy) + abs(sx - cx)
            if d <= hotspot_group_radius:
                hotspot_squads.add(s_idx)

        # 没接敌、还没拿到支援目标的人，才会被残局热点模式接管
        for s_idx, local_indices in enumerate(squads):
            for li in local_indices:
                gi = idx_start + li
                if not self.alive_mask[gi]:
                    continue

                # 正在直接看到敌人的人，不抢他行为
                if can_see_mask[li]:
                    continue

                # 已经有前面原始支援逻辑分到目标的人，不覆盖
                if self.current_support_target_A[gi] is not None:
                    continue

                gy, gx = self.agents_pos[gi]

                # 已经靠近热点的 squad 以正面点为主；较远 squad 在三个点里选最近的一个推进
                if s_idx in hotspot_squads:
                    target = hotspot_points[0]
                else:
                    best_tgt = None
                    best_d = None
                    for ty, tx in hotspot_points:
                        d = abs(gy - ty) + abs(gx - tx)
                        if best_d is None or d < best_d:
                            best_d = d
                            best_tgt = (ty, tx)
                    target = best_tgt

                ty, tx = target
                ty = int(np.clip(ty, 1, self.grid_h - 2))
                tx = int(np.clip(tx, 1, self.grid_w - 2))
                ty, tx = self._find_nearest_free_cell_for_team(
                    ty,
                    tx,
                    TEAM_A,
                    ignore_idx=gi,
                    max_r=int(getattr(self, "endgame_hotspot_assign_radius", 7)),
                )

                self.current_support_target_A[gi] = (ty, tx)
                self.current_support_mode_A[gi] = 3  # 3 = endgame hotspot collapse

    def _get_obs(self):
        obs_dim = 20
        obs = np.zeros((self.n_agents_total, obs_dim), dtype=np.float32)
        max_sound = np.max(self.sound_map) if np.any(self.sound_map) else 1.0

        # 先算 A 队 team_alert
        teamA_visible = self._get_team_visible_enemies(TEAM_A)
        teamA_alert = 1.0 if len(teamA_visible) > 0 else 0.0

        # 防止 current_squad_objectives_A 还没准备好
        squad_objectives_A = getattr(self, "current_squad_objectives_A", None)
        # 最近一次交火中心 / 包抄参考点（没有就返回 None）
        fight_center, flank_hint = self._get_recent_engagement_hint_for_A()

        for i in range(self.n_agents_total):
            if not self.alive_mask[i]:
                continue

            team_id = TEAM_A if i < self.num_team_a else TEAM_B
            hp_norm = self.agents_hp[i] / self.max_hp
            y, x = self.agents_pos[i]
            y_norm = y / max(1, self.grid_h - 1)
            x_norm = x / max(1, self.grid_w - 1)
            t = self.terrain[y, x]
            terrain_norm = t / 2.0
            sound_norm = self.sound_map[y, x] / (max_sound + 1e-6)
            orient_norm = self.agents_orient[i] / 7.0
            posture_norm = self.agents_posture[i] / 1.0  # 0 or 1 映射到 [0,1]

            obs[i, 0] = float(team_id)
            obs[i, 1] = float(hp_norm)
            obs[i, 2] = float(y_norm)
            obs[i, 3] = float(x_norm)
            obs[i, 4] = float(terrain_norm)
            obs[i, 5] = float(sound_norm)
            obs[i, 6] = float(orient_norm)
            obs[i, 7] = float(posture_norm)
            obs[i, 8] = 1.0  # alive_flag
            # ===== 新增维度：只重点给 A 队 =====
            if team_id == TEAM_A:
                # [9] 自己是否看见敌人
                can_see_enemy = 1.0 if self._agent_can_see_enemy(i) else 0.0
                obs[i, 9] = can_see_enemy

                # [10] A队是否有人看见敌人
                obs[i, 10] = teamA_alert

                # [11]-[12] 自己最近可见敌人的相对坐标（归一化到 [-1,1]）
                nearest_enemy_idx = self._get_nearest_visible_enemy_for_agent(i)
                if nearest_enemy_idx is not None:
                    ey, ex = self.agents_pos[nearest_enemy_idx]
                    rel_y = (ey - y) / max(1, self.grid_h - 1)
                    rel_x = (ex - x) / max(1, self.grid_w - 1)
                    obs[i, 11] = np.clip(rel_y, -1.0, 1.0)
                    obs[i, 12] = np.clip(rel_x, -1.0, 1.0)
                else:
                    obs[i, 11] = 0.0
                    obs[i, 12] = 0.0

                # [13]-[14] 当前目标相对坐标：
                # - 平时：沿用原 squad objective
                # - 残局热点模式：若自己没看见敌人，则直接给热点推进目标
                endgame_tgt = None
                if can_see_enemy < 0.5:
                    endgame_tgt = self._get_endgame_hotspot_target_for_agent_A(i)

                if endgame_tgt is not None:
                    oy, ox = endgame_tgt
                    rel_oy = (oy - y) / max(1, self.grid_h - 1)
                    rel_ox = (ox - x) / max(1, self.grid_w - 1)
                    obs[i, 13] = np.clip(rel_oy, -1.0, 1.0)
                    obs[i, 14] = np.clip(rel_ox, -1.0, 1.0)
                elif squad_objectives_A is not None and len(squad_objectives_A) > 0:
                    s_idx = self._get_squad_index_for_agent_A(i)
                    s_idx = min(s_idx, len(squad_objectives_A) - 1)

                    oy, ox = squad_objectives_A[s_idx]
                    rel_oy = (oy - y) / max(1, self.grid_h - 1)
                    rel_ox = (ox - x) / max(1, self.grid_w - 1)

                    obs[i, 13] = np.clip(rel_oy, -1.0, 1.0)
                    obs[i, 14] = np.clip(rel_ox, -1.0, 1.0)
                else:
                    obs[i, 13] = 0.0
                    obs[i, 14] = 0.0

                # [15] support_needed:
                # 平时沿用原逻辑；残局热点模式下，没看见敌人就明确拉高为需要靠拢
                has_recent_engagement = 1.0 if fight_center is not None else 0.0
                if endgame_tgt is not None:
                    obs[i, 15] = 1.0
                else:
                    obs[i, 15] = 1.0 if (
                            can_see_enemy < 0.5 and (teamA_alert > 0.5 or has_recent_engagement > 0.5)
                    ) else 0.0

                # [16]-[17] 最近交火中心相对坐标
                if fight_center is not None:
                    fy, fx = fight_center
                    rel_fy = (fy - y) / max(1, self.grid_h - 1)
                    rel_fx = (fx - x) / max(1, self.grid_w - 1)
                    obs[i, 16] = np.clip(rel_fy, -1.0, 1.0)
                    obs[i, 17] = np.clip(rel_fx, -1.0, 1.0)
                else:
                    obs[i, 16] = 0.0
                    obs[i, 17] = 0.0

                # [18]-[19] 包抄参考点相对坐标
                if flank_hint is not None:
                    gy, gx = flank_hint
                    rel_gy = (gy - y) / max(1, self.grid_h - 1)
                    rel_gx = (gx - x) / max(1, self.grid_w - 1)
                    obs[i, 18] = np.clip(rel_gy, -1.0, 1.0)
                    obs[i, 19] = np.clip(rel_gx, -1.0, 1.0)
                else:
                    obs[i, 18] = 0.0
                    obs[i, 19] = 0.0

            else:
                # B 队新增维度先全置 0，不额外增强
                obs[i, 9] = 0.0
                obs[i, 10] = 0.0
                obs[i, 11] = 0.0
                obs[i, 12] = 0.0
                obs[i, 13] = 0.0
                obs[i, 14] = 0.0
                obs[i, 15] = 0.0
                obs[i, 16] = 0.0
                obs[i, 17] = 0.0
                obs[i, 18] = 0.0
                obs[i, 19] = 0.0

        return obs

    # ======================================================================
    # 队伍级“战场认知地图” (occupancy grid)
    # ======================================================================

    def _init_known_maps(self):
        """
        初始化两队的已知地图:
          0 = 未知
          1 = 边界/障碍
          2 = 可走区域
        开局默认把地图四周的边界标记为 1（已知不可越界）。
        """
        H, W = self.grid_h, self.grid_w
        self.known_map_A = np.zeros((H, W), dtype=np.int8)
        self.known_map_B = np.zeros((H, W), dtype=np.int8)

        for y in range(H):
            for x in range(W):
                if y == 0 or y == H - 1 or x == 0 or x == W - 1:
                    self.known_map_A[y, x] = 1
                    self.known_map_B[y, x] = 1

    def _project_to_local(self, ori: int, dy: int, dx: int):
        """
        把 (dy,dx) 从全局坐标投影到 agent 局部坐标:
          pf > 0 表示在前方, pf < 0 在后方
          pl > 0 表示在右侧, pl < 0 在左侧
        """
        fy, fx = self._dir2vec(ori)  # 朝向向量
        ry, rx = -fx, fy  # 向右的正交向量
        pf = dy * fy + dx * fx
        pl = dy * ry + dx * rx
        return pf, pl

    def _update_team_knowledge(self, team_id: int):
        """
        用当前所有存活队员的矩形感知区域，更新本队的 known_map。
        - 看到的墙/边界 -> 1
        - 看到的可走格 -> 2
        未看到的仍为 0
        """
        if team_id == TEAM_A:
            kmap = self.known_map_A
            idx_start, idx_end = 0, self.num_team_a
        else:
            kmap = self.known_map_B
            idx_start, idx_end = self.num_team_a, self.n_agents_total

        H, W = self.grid_h, self.grid_w

        for i in range(idx_start, idx_end):
            if not self.alive_mask[i]:
                continue
            sy, sx = self.agents_pos[i]
            ori = int(self.agents_orient[i])

            # 遍历整张地图，用局部坐标 + 矩形窗口筛选
            for y in range(H):
                for x in range(W):
                    dy = y - sy
                    dx = x - sx
                    if dy == 0 and dx == 0:
                        # 自己脚下格子必然可走
                        if self.terrain[y, x] == WALL:
                            kmap[y, x] = 1
                        else:
                            kmap[y, x] = 2
                        continue

                    pf, pl = self._project_to_local(ori, dy, dx)

                    # 前后不同感知距离 + 左右宽度
                    if pf >= 0:
                        max_forward = self.perc_front
                    else:
                        max_forward = -self.perc_back  # pf 为负数，绝对值不超过 perc_back

                    # 前后方向距离约束
                    if pf > max_forward or pf < -self.perc_back:
                        continue

                    # 左右宽度约束
                    if abs(pl) > self.perc_side:
                        continue

                    # 视线被墙挡住则看不到
                    if not self._has_line_of_sight(sy, sx, y, x):
                        continue

                    # 能看到的格子根据地形更新已知信息
                    if self.terrain[y, x] == WALL:
                        kmap[y, x] = 1
                    else:
                        # PLAIN / COVER 都视为“可走”（战术上可再细分）
                        if kmap[y, x] == 0:  # 未知才更新，避免被手雷炸平后覆盖旧信息的话可以再精细化
                            kmap[y, x] = 2

    # ======================================================================
    # 动作 / 朝向 / 移动 / 声音
    # ======================================================================

    def _dir2vec(self, d: int):
        """
        0..7 -> (dy, dx)
        0: 上, 1: 右上, 2: 右, 3: 右下,
        4: 下, 5: 左下, 6: 左, 7: 左上
        """
        d = int(d) % 8
        if d == 0:
            return -1, 0
        elif d == 1:
            return -1, 1
        elif d == 2:
            return 0, 1
        elif d == 3:
            return 1, 1
        elif d == 4:
            return 1, 0
        elif d == 5:
            return 1, -1
        elif d == 6:
            return 0, -1
        else:
            return -1, -1

    def _within_fov(self, i: int, ty: int, tx: int) -> bool:
        """
        判断目标 (ty,tx) 是否在 agent i 的视野锥内。
        使用朝向向量 vs 目标方向向量的 cos(angle) 与 fov_cos 比较。
        """
        if not self.alive_mask[i]:
            return False
        sy, sx = self.agents_pos[i]
        dy = ty - sy
        dx = tx - sx
        if dy == 0 and dx == 0:
            return True

        oy, ox = self._dir2vec(self.agents_orient[i])
        ori_vec = np.array([ox, oy], dtype=np.float32)
        tgt_vec = np.array([dx, dy], dtype=np.float32)

        ori_norm = np.linalg.norm(ori_vec) + 1e-6
        tgt_norm = np.linalg.norm(tgt_vec) + 1e-6
        cos_angle = float(np.dot(ori_vec, tgt_vec) / (ori_norm * tgt_norm))
        return cos_angle >= self.fov_cos

    def _ori_from_delta(self, dy: float, dx: float) -> int:
        if abs(dy) < 1e-6 and abs(dx) < 1e-6:
            return 0
        angle = np.arctan2(dy, dx)
        angle = (angle + np.pi / 2) % (2 * np.pi)
        return int(np.round(angle / (np.pi / 4))) % 8

    def _orient_only_towards(self, i: int, ty: int, tx: int) -> int:
        sy, sx = self.agents_pos[i]
        ideal_ori = self._ori_from_delta(ty - sy, tx - sx)
        ori = int(self.agents_orient[i])
        diff = (ideal_ori - ori) % 8
        if diff == 0:
            return 0
        return 2 if diff <= 4 else 1

    def _facing_score_towards(self, i: int, ty: float, tx: float) -> float:
        if not self.alive_mask[i]:
            return 0.0
        sy, sx = self.agents_pos[i]
        dy = ty - sy
        dx = tx - sx
        if abs(dy) < 1e-6 and abs(dx) < 1e-6:
            return 1.0

        oy, ox = self._dir2vec(self.agents_orient[i])
        ori_vec = np.array([ox, oy], dtype=np.float32)
        tgt_vec = np.array([dx, dy], dtype=np.float32)
        ori_norm = np.linalg.norm(ori_vec) + 1e-6
        tgt_norm = np.linalg.norm(tgt_vec) + 1e-6
        cos_angle = float(np.dot(ori_vec, tgt_vec) / (ori_norm * tgt_norm))
        return 0.5 * (cos_angle + 1.0)

    def _facing_score_with_ori(self, i: int, ori: int, ty: float, tx: float) -> float:
        if not self.alive_mask[i]:
            return 0.0
        sy, sx = self.agents_pos[i]
        dy = ty - sy
        dx = tx - sx
        if abs(dy) < 1e-6 and abs(dx) < 1e-6:
            return 1.0

        oy, ox = self._dir2vec(int(ori))
        ori_vec = np.array([ox, oy], dtype=np.float32)
        tgt_vec = np.array([dx, dy], dtype=np.float32)
        ori_norm = np.linalg.norm(ori_vec) + 1e-6
        tgt_norm = np.linalg.norm(tgt_vec) + 1e-6
        cos_angle = float(np.dot(ori_vec, tgt_vec) / (ori_norm * tgt_norm))
        return 0.5 * (cos_angle + 1.0)

    def _teamA_filter_target(self, i: int):
        fight_center_hint, _ = self._get_recent_engagement_hint_for_A()
        if fight_center_hint is not None:
            return fight_center_hint

        support_targets_A = getattr(self, "current_support_target_A", None)
        if support_targets_A is not None and i < len(support_targets_A):
            tgt = support_targets_A[i]
            if tgt is not None:
                return tgt

        visible_enemy_idx = self._get_nearest_visible_enemy_for_agent(i)
        if visible_enemy_idx is not None:
            vy, vx = self.agents_pos[visible_enemy_idx]
            return (int(vy), int(vx))

        return self._get_teamB_spawn_fallback_target_for_agent_A(i)

    def _teamA_move_blocked(self, i: int, action: int) -> bool:
        if action not in (3, 7):
            return False

        ori = int(self.agents_orient[i])
        test_ori, move_steps = self._edge_move_test_ori(ori, int(action))
        if move_steps <= 0:
            return False

        y, x = self.agents_pos[i]
        dy, dx = self._dir2vec(test_ori)
        for step in range(1, move_steps + 1):
            ny = y + dy * step
            nx = x + dx * step
            if not (0 <= ny < self.grid_h and 0 <= nx < self.grid_w):
                return True
            if self.terrain[ny, nx] == WALL:
                return True
            if self._is_cell_occupied_by_team(ny, nx, TEAM_A, ignore_idx=i):
                return True
        return False

    def _is_outward_facing_at_edge(self, i: int, margin: int = 1) -> bool:
        y, x = self.agents_pos[i]
        dy, dx = self._dir2vec(int(self.agents_orient[i]))

        if y <= margin and dy < 0:
            return True
        if y >= self.grid_h - 1 - margin and dy > 0:
            return True
        if x <= margin and dx < 0:
            return True
        if x >= self.grid_w - 1 - margin and dx > 0:
            return True
        return False

    '''
    def _best_inward_ori_near_edge(self, i: int, margin: int = 1):
        y, x = self.agents_pos[i]
        dist_to_edge = min(y, self.grid_h - 1 - y, x, self.grid_w - 1 - x)
        if dist_to_edge > margin:
            return None

        fight_center, _ = self._get_recent_engagement_hint_for_A()
        if fight_center is not None:
            cy, cx = fight_center
        else:
            cy = (self.grid_h - 1) / 2.0
            cx = (self.grid_w - 1) / 2.0

        base_ori = self._ori_from_delta(cy - y, cx - x)
        candidates = [
            base_ori,
            (base_ori - 1) % 8, (base_ori + 1) % 8,
            (base_ori - 2) % 8, (base_ori + 2) % 8,
            (base_ori + 4) % 8,
        ]

        best_ori = None
        best_score = None
        for cand in candidates:
            dy, dx = self._dir2vec(cand)
            ny, nx = y + dy, x + dx
            if not (0 <= ny < self.grid_h and 0 <= nx < self.grid_w):
                continue

            turn_cost = min((cand - base_ori) % 8, (base_ori - cand) % 8)
            score = 4.0 * min(ny, self.grid_h - 1 - ny, nx, self.grid_w - 1 - nx)
            if self.terrain[ny, nx] == WALL:
                score -= 3.0
            score -= 0.15 * turn_cost

            if best_score is None or score > best_score:
                best_score = score
                best_ori = cand

        return best_ori

    def _force_turn_from_edge_for_A(self, i: int, raw_action: int) -> int:
        if i >= self.num_team_a or not self.alive_mask[i]:
            return raw_action
        if self._agent_can_see_enemy(i):
            return raw_action

        inward_ori = self._best_inward_ori_near_edge(i, margin=self.edge_turn_margin_A)
        if inward_ori is None:
            return raw_action

        ori = int(self.agents_orient[i])

        # 已经贴边且朝外看 -> 优先强制转身
        if self._is_outward_facing_at_edge(i, margin=self.edge_turn_margin_A):
            diff = (inward_ori - ori) % 8
            if diff == 0:
                return raw_action
            return 2 if diff <= 4 else 1

        # 即将走出地图 -> 提前纠正
        if raw_action in (3, 4, 5, 6, 7):
            test_ori = ori
            if raw_action == 4:
                test_ori = (ori + 4) % 8
            elif raw_action == 5:
                test_ori = (ori - 2) % 8
            elif raw_action == 6:
                test_ori = (ori + 2) % 8

            dy, dx = self._dir2vec(test_ori)
            ny = self.agents_pos[i][0] + dy
            nx = self.agents_pos[i][1] + dx
            if not (0 <= ny < self.grid_h and 0 <= nx < self.grid_w):
                diff = (inward_ori - ori) % 8
                if diff == 0:
                    return 3
                return 2 if diff <= 4 else 1

        return raw_action
        '''

    def _edge_move_test_ori(self, ori: int, action: int):
        """
        给定当前朝向 ori 和动作 action，返回：
          - 这个动作对应的“第一步测试朝向” test_ori
          - move_steps (0/1/2)
        """
        test_ori = ori
        move_steps = 0

        if action == 3:  # WalkForward
            move_steps = 1
            test_ori = ori
        elif action == 4:  # WalkBackward
            move_steps = 1
            test_ori = (ori + 4) % 8
        elif action == 5:  # StrafeLeft
            move_steps = 1
            test_ori = (ori - 2) % 8
        elif action == 6:  # StrafeRight
            move_steps = 1
            test_ori = (ori + 2) % 8
        elif action == 7:  # SprintForward
            move_steps = 2
            test_ori = ori

        return test_ori, move_steps

    def _add_sound(self, y: int, x: int, base: float):
        self.sound_map[y, x] += float(base)

    def _attempt_fire(self, shooter_idx):
        """
        处理射击尝试，考虑：
          - 若在眩晕或装弹中 → 不能开火
          - 若弹药为 0 → 开始装弹（本回合不射击）
          - 若有弹且不在装弹 → 消耗1发，返回目标信息
        返回:
          None 或 (shooter_idx, target_type, target_id)
        """
        if not self.alive_mask[shooter_idx]:
            return None

        # 眩晕中不能射击

        # 正在装弹中
        if self.agent_reload_timer is not None and self.agent_reload_timer[shooter_idx] > 0:
            return None

        # 没子弹 → 开始装弹
        if self.agent_ammo is not None and self.agent_ammo[shooter_idx] <= 0:
            wt = int(self.agent_weapon_type[shooter_idx])
            self.agent_reload_timer[shooter_idx] = self.weapon_reload_steps[wt]
            # 下回合才有子弹，这回合不射击
            self.agent_ammo[shooter_idx] = self.weapon_mag_size[wt]
            return None

        # 有弹 → 消耗 1 发，选择目标
        if self.agent_ammo is not None:
            self.agent_ammo[shooter_idx] -= 1

        shot = self._choose_shot_target(shooter_idx)
        if shot is None:
            return None
        return (shooter_idx, shot[0], shot[1])

    def _apply_team_actions(self, team_id, actions: np.ndarray):
        #  print(f"Team {team_id} actions: {actions.tolist()}")
        if team_id == TEAM_A:
            idx_start = 0
            idx_end = self.num_team_a
        else:
            idx_start = self.num_team_a
            idx_end = self.n_agents_total

        actions = np.asarray(actions, dtype=np.int64)
        override_actions = actions.copy()
        count = idx_end - idx_start

        if team_id == TEAM_A:
            # A 队是 PPO 训练对象，不能硬改动作。
            # 盲射/无效射击通过 shoot_bad 负奖励约束，保证 action、logprob、状态转移一致。
            pass
        else:
            # B 队是启发式对手，可以保留自动开火和无效射击抑制。
            for local_i in range(count):
                i = idx_start + local_i
                if i >= idx_end or not self.alive_mask[i]:
                    continue
                if not self._agent_can_see_enemy(i):
                    continue

                has_valid_target = False
                sy, sx = self.agents_pos[i]
                wt = int(self.agent_weapon_type[i]) if hasattr(self, 'agent_weapon_type') else 0
                eff_range = self.gun_range + self.weapon_range_bonus.get(wt, 0)

                for j in range(self.n_agents_total):
                    if not self.alive_mask[j]:
                        continue
                    team_j = TEAM_A if j < self.num_team_a else TEAM_B
                    if team_j == TEAM_B:
                        continue

                    ty, tx = self.agents_pos[j]
                    dist = abs(ty - sy) + abs(tx - sx)

                    if (dist <= eff_range and
                            self._within_fov(i, ty, tx) and
                            self._has_line_of_sight(sy, sx, ty, tx)):
                        has_valid_target = True
                        break

                if has_valid_target:
                    override_actions[local_i] = 9
                elif override_actions[local_i] == 9:
                    override_actions[local_i] = 0

        actions = override_actions
        # print(f"Team {team_id} actions (after override): {actions.tolist()}")  # 调试用，看是否把无效9改成0

        shots = []
        shoot_good = 0  # 能打到敌人并且选择了 Shoot
        shoot_bad = 0  # 选择了 Shoot 但其实打不到任何目标

        # 1) movement / posture 同原来
        for local_i, a in enumerate(actions):
            i = idx_start + local_i
            if i >= idx_end or not self.alive_mask[i]:
                continue
            self._apply_single_action_movement_posture(i, int(a))

        # 2) 射击
        for local_i, a in enumerate(actions):
            i = idx_start + local_i
            if i >= idx_end or not self.alive_mask[i]:
                continue
            if int(a) == 9:  # Shoot
                shot = self._attempt_fire(i)
                if shot is not None:
                    self.shot_this_step[i] = True

                    shots.append(shot)

                    if shot[1] == "enemy":
                        shoot_good += 1
                    else:
                        # 打到平民/人质以后，有大惩罚，单独处理
                        pass
                else:
                    # 没任何目标，其实是浪费子弹/暴露位置
                    shoot_bad += 1

        return shots, shoot_good, shoot_bad

    def _apply_single_action_movement_posture(self, i: int, a: int):
        """
        处理单个 agent 的朝向变化 / 位移 / 姿态变化，并写入声音。
        """

        if not self.alive_mask[i]:
            return
        y, x = self.agents_pos[i]
        ori = int(self.agents_orient[i])
        post = int(self.agents_posture[i])

        # 若被“掩护火力”压制，则禁止位移类动作（3~7），只能转向/换姿态/射击/扔弹
        if getattr(self, "suppressed", None) is not None and self.suppressed[i]:
            if a in (3, 4, 5, 6, 7):
                # 尝试移动会被火力压制，只能原地
                a = 0

        # 0: Idle
        if a == 0:
            return

        # 1: TurnLeft, 2: TurnRight
        if a == 1:
            self.agents_orient[i] = (ori - 1) % 8
            self._add_sound(y, x, base=self.sound_turn)
            return
        if a == 2:
            self.agents_orient[i] = (ori + 1) % 8
            self._add_sound(y, x, base=self.sound_turn)
            return

        # 姿态切换: 8 （站立/蹲之间切换）
        if a == 8:  # ToggleCrouch（简单两态切换）
            if post == 0:
                post_new = 1
            else:
                post_new = 0
            self.agents_posture[i] = post_new
            self._add_sound(y, x, base=self.sound_turn)
            return


        # 移动: 3~7
        dir_y, dir_x = self._dir2vec(ori)
        left_ori = (ori - 2) % 8
        right_ori = (ori + 2) % 8
        ldy, ldx = self._dir2vec(left_ori)
        rdy, rdx = self._dir2vec(right_ori)

        steps = 0
        step_vec = (0, 0)
        sprint = False

        if a == 3:  # WalkForward
            steps = 1;
            step_vec = (dir_y, dir_x)
        elif a == 4:  # WalkBackward
            steps = 1;
            step_vec = (-dir_y, -dir_x)
        elif a == 5:  # StrafeLeft
            steps = 1;
            step_vec = (ldy, ldx)
        elif a == 6:  # StrafeRight
            steps = 1;
            step_vec = (rdy, rdx)
        elif a == 7:  # SprintForward
            steps = 2;
            step_vec = (dir_y, dir_x);
            sprint = True

        # 根据当前血量调整最大步数（受伤减速）
        """
        if steps > 0:
            hp_ratio = self.agents_hp[i] / max(1, self.max_hp)
            if hp_ratio <= self.critical_threshold:
                # 重伤：禁止冲刺，只能走 1 格
                steps = min(steps, 1)
                sprint = False
            elif hp_ratio <= self.wounded_threshold:
                # 轻伤：冲刺从2格减为1格
                if sprint:
                    steps = 1
        """
        if steps <= 0:
            return

        ny, nx = y, x
        for _ in range(steps):
            ty = ny + step_vec[0]
            tx = nx + step_vec[1]
            team_i = TEAM_A if i < self.num_team_a else TEAM_B
            if not (0 <= ty < self.grid_h and 0 <= tx < self.grid_w):
                # 越界：不再使用旧的边缘强制转向 helper
                # B队保持原先的简单反向处理
                if team_i == TEAM_B:
                    self.agents_orient[i] = (ori + 4) % 8

                # A队：仅在未接敌时做一次“轻量朝内转向”，避免贴边一直发呆
                elif team_i == TEAM_A and not self._agent_can_see_enemy(i):
                    y, x = self.agents_pos[i]
                    center_y = (self.grid_h - 1) / 2.0
                    center_x = (self.grid_w - 1) / 2.0

                    dy = center_y - y
                    dx = center_x - x

                    if abs(dy) < 1e-6 and abs(dx) < 1e-6:
                        inward_ori = ori
                    else:
                        angle = np.arctan2(dy, dx)
                        angle = (angle + np.pi / 2) % (2 * np.pi)
                        inward_ori = int(np.round(angle / (np.pi / 4))) % 8

                    self.agents_orient[i] = inward_ori

                break
            # ---------B队不重复占位------------
            blocked = (self.terrain[ty, tx] == WALL)

            team_i = TEAM_A if i < self.num_team_a else TEAM_B
            if self._is_cell_occupied(ty, tx):
                blocked = True

            if blocked:
                break
            # ----------结束-------------------
            ny, nx = ty, tx

            # 每成功走一步，写脚步声
            if sprint:
                vol = self.sound_strength * self.sound_sprint
            else:
                if post == 0:
                    vol = self.sound_strength * self.sound_walk
                elif post == 1:
                    vol = self.sound_strength * self.sound_walk * self.sound_crouch_factor
                else:
                    vol = self.sound_strength * self.sound_walk * self.sound_prone_factor
            self._add_sound(ny, nx, base=vol)

        moved = (ny != y) or (nx != x)
        self.agents_pos[i] = (ny, nx)
        if moved:
            self.moved_this_step[i] = True

    def _is_cell_occupied(self, y: int, x: int, ignore_team=None) -> bool:
        for k in range(self.n_agents_total):
            if not self.alive_mask[k]:
                continue
            if ignore_team is not None:
                team_k = TEAM_A if k < self.num_team_a else TEAM_B
                if team_k == ignore_team:
                    continue
            if (self.agents_pos[k, 0] == y) and (self.agents_pos[k, 1] == x):
                return True
        return False

    # ---------新增B队不重复占位代码工具函数-----------------

    def _is_cell_occupied_by_team(self, y: int, x: int, team_id: int, ignore_idx: int = None) -> bool:
        if team_id == TEAM_A:
            start, end = 0, self.num_team_a
        else:
            start, end = self.num_team_a, self.n_agents_total

        for k in range(start, end):
            if not self.alive_mask[k]:
                continue
            if ignore_idx is not None and k == ignore_idx:
                continue
            if self.agents_pos[k, 0] == y and self.agents_pos[k, 1] == x:
                return True
        return False

    def _find_nearest_free_cell_for_team(self, y: int, x: int, team_id: int, ignore_idx: int = None, max_r: int = 4):
        if (
                0 <= y < self.grid_h and 0 <= x < self.grid_w
                and self.terrain[y, x] != WALL
                and not self._is_cell_occupied_by_team(y, x, team_id, ignore_idx=ignore_idx)
        ):
            return (y, x)

        for r in range(1, max_r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dy) + abs(dx) > r:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < self.grid_h and 0 <= nx < self.grid_w):
                        continue
                    if self.terrain[ny, nx] == WALL:
                        continue
                    if self._is_cell_occupied_by_team(ny, nx, team_id, ignore_idx=ignore_idx):
                        continue
                    return (ny, nx)

        return (y, x)

    def _count_team_neighbors(self, y: int, x: int, team_id: int, radius: int = 2, ignore_idx: int = None) -> int:
        if team_id == TEAM_A:
            start, end = 0, self.num_team_a
        else:
            start, end = self.num_team_a, self.n_agents_total

        cnt = 0
        for k in range(start, end):
            if not self.alive_mask[k]:
                continue
            if ignore_idx is not None and k == ignore_idx:
                continue
            ky, kx = self.agents_pos[k]
            if abs(int(ky) - y) + abs(int(kx) - x) <= radius:
                cnt += 1
        return cnt

    def _find_low_density_cell_for_team(
            self,
            y: int,
            x: int,
            team_id: int,
            ignore_idx: int = None,
            max_r: int = 6,
            density_radius: int = 2,
    ):
        best = None
        best_score = None

        for r in range(0, max_r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dy) + abs(dx) > r:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < self.grid_h and 0 <= nx < self.grid_w):
                        continue
                    if self.terrain[ny, nx] == WALL:
                        continue
                    if self._is_cell_occupied_by_team(ny, nx, team_id, ignore_idx=ignore_idx):
                        continue

                    density = self._count_team_neighbors(
                        ny, nx, team_id, radius=density_radius, ignore_idx=ignore_idx
                    )
                    dist = abs(dy) + abs(dx)
                    score = (density, dist)
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (ny, nx)

            if best is not None and best_score is not None and best_score[0] <= 1:
                return best

        return best if best is not None else (y, x)

    # ---------------新增结束--------------------------------

    def _compute_suppression(self):
        """
        根据当前状态计算每个 agent 是否被敌方“掩护火力”压制。
        条件：
          - 某个敌方在 COVER 上
          - 敌方能在 FOV + 射程 + LOS 内看到你
        被压制的单位在本回合不能执行移动类动作（3~7）。
        """
        self.suppressed = np.zeros(self.n_agents_total, dtype=bool)

        for j in range(self.n_agents_total):
            if not self.alive_mask[j]:
                continue
            jy, jx = self.agents_pos[j]
            if self.terrain[jy, jx] != COVER:
                continue
            team_j = TEAM_A if j < self.num_team_a else TEAM_B
            if self.agent_weapon_type is not None:
                wtype = int(self.agent_weapon_type[j])
            else:
                wtype = 0
            max_range = self.gun_range + self.weapon_range_bonus.get(wtype, 0)

            # 找被这个掩体火力压制的敌人
            for i in range(self.n_agents_total):
                if not self.alive_mask[i]:
                    continue
                team_i = TEAM_A if i < self.num_team_a else TEAM_B
                if team_i == team_j:
                    continue

                iy, ix = self.agents_pos[i]
                dist = abs(iy - jy) + abs(ix - jx)
                if (
                        dist <= max_range
                        and self._within_fov(j, iy, ix)
                        and self._has_line_of_sight(jy, jx, iy, ix)
                ):
                    # 距离比例（越近越小）
                    dist_ratio = dist / max(1, self.gun_range)
                    # 概率范围：0.3 ~ 0.8
                    prob = 0.2 + 0.3 * (1.0 - dist_ratio)
                    if self.np_random.rand() < prob:
                        self.suppressed[i] = True

    # ======================================================================
    # 敌方启发式策略（同构动作空间）
    # ======================================================================

    def _get_team_visible_enemies(self, team_id: int):
        """返回当前队伍至少有一个成员能看见的敌人全局索引列表（去上帝视角核心）"""
        if team_id == TEAM_A:
            idx_start, count = 0, self.num_team_a
        else:
            idx_start, count = self.num_team_a, self.num_team_b

        visible = set()
        for local_i in range(count):
            gi = idx_start + local_i
            if not self.alive_mask[gi]:
                continue
            # 复用你已有的 _agent_can_see_enemy 判断（FOV + 射程 + LOS）
            # 但我们需要知道具体是哪个敌人
            for j in range(self.n_agents_total):
                if not self.alive_mask[j]:
                    continue
                team_j = TEAM_A if j < self.num_team_a else TEAM_B
                if team_j == (TEAM_A if team_id == TEAM_A else TEAM_B):
                    continue
                # 精确检查（和 _agent_can_see_enemy 逻辑完全一致）
                sy, sx = self.agents_pos[gi]
                ty, tx = self.agents_pos[j]
                dist = abs(ty - sy) + abs(tx - sx)
                wt = int(self.agent_weapon_type[gi]) if self.agent_weapon_type is not None else 0
                eff_range = self.gun_range + self.weapon_range_bonus.get(wt, 0)
                if (dist <= eff_range and
                        self._within_fov(gi, ty, tx) and
                        self._has_line_of_sight(sy, sx, ty, tx)):
                    visible.add(j)
                    break  # 一个B队员看见就够了（情报共享）
        return list(visible)

    def _heuristic_team_actions(self, team_id):
        """
        为某个 team 产生动作（使用相同动作空间 0..9）。
        这里主要给 Team B 用，但逻辑对 Team A 也同样适用（能力完全对称）。

        目标：更专业的“战略 + 战术”层级决策：
          - 战略层（虚拟指挥官）：
                根据兵力对比 / 伤亡 / 接触情况 / 地形类型 / 任务目标，选择 team-level strategy
                如: ADVANCE / DEFEND / FLANK_LEFT / FLANK_RIGHT / RESCUE / CAUTIOUS / COUNTER_ATTACK
          - 战术层（小队长）：
                按 strategy 为本小队规划 objective（前沿接触线、包抄拐点、掩体、防线、人质房间等）
                选择“进攻/防御/机动/掩护火力”的节奏。
          - 执行层（小队队员）：
                优先服从小队长的机动，保持编队；但在自己看到敌人时有一定战术自主权（开火/短撤退）。

        另外保留“发现敌人后优先射击”的规则：
          - 一旦本方任意一名士兵在视野 + 射程 + LOS 内看到敌人，则 team_alert = True。
          - 任一士兵如果自己也能看到敌人，则在不需要立刻撤退的前提下优先执行 Shoot 动作。
        """
        # 如果为 B 队且挂了神经网络控制器，则直接交给 NN 处理
        if team_id == TEAM_B and getattr(self, "nn_controller_B", None) is not None:
            return self.nn_controller_B(self, team_id)

        # ---------- 1) 基本索引 ----------
        if team_id == TEAM_A:
            idx_start = 0
            count = self.num_team_a
        else:
            idx_start = self.num_team_a
            count = self.num_team_b

        actions = np.zeros(count, dtype=np.int64)

        # ---------- 2) 军事编制：划分小队 & 确定小队长 ----------
        squad_size = self._get_effective_squad_size(team_id)
        squads = []
        squad_leaders = []  # 存储 agent 全局 index
        for s in range(0, count, squad_size):
            local_indices = list(range(s, min(s + squad_size, count)))
            squads.append(local_indices)
            # 小队长：本小队中第一个存活的成员（若全灭则 None）
            leader_idx = None
            for li in local_indices:
                gi = idx_start + li
                if self.alive_mask[gi]:
                    leader_idx = gi
                    break
            squad_leaders.append(leader_idx)

        # ---------- 工具函数：巡逻分路 & 局部兵力对比 & 最近敌人 ----------
        def _get_b_lane_bounds(s_idx: int):
            """
            给 B 队每个 squad 固定一个横向 sector:
              0 -> 左路
              1 -> 中路
              2 -> 右路
            超过3个 squad 就循环。
            """
            lane_id = s_idx % 3
            W = self.grid_w

            if lane_id == 0:
                # 左路
                x_min = 1
                x_max = max(2, int(W * 0.26))
            elif lane_id == 1:
                # 中路
                x_min = max(1, int(W * 0.38))
                x_max = min(W - 2, int(W * 0.62))
            else:
                # 右路
                x_min = min(W - 2, int(W * 0.74))
                x_max = W - 2

            return x_min, x_max

        def _clip_objective_to_b_lane(y: int, x: int, s_idx: int, leader_idx: int = None):
            """
            把 B 队 objective 裁剪到本 squad 的 lane 内，
            再找最近的可用格。
            """
            x_min, x_max = _get_b_lane_bounds(s_idx)

            y = int(np.clip(y, 1, self.grid_h - 2))
            x = int(np.clip(x, x_min, x_max))

            y, x = self._find_low_density_cell_for_team(
                y, x, TEAM_B, ignore_idx=leader_idx, max_r=9, density_radius=3
            )
            return y, x

        def _force_ratio_around(team_id: int, radius: int = 5) -> float:
            """
            在队伍质心附近，统计 radius 内友军 / 敌军数量，
            用于估计局部兵力对比（>1 优势，<1 劣势）。
            """
            if team_id == TEAM_A:
                idx0, idx1 = 0, self.num_team_a
            else:
                idx0, idx1 = self.num_team_a, self.n_agents_total

            friends = [i for i in range(idx0, idx1) if self.alive_mask[i]]
            if team_id == TEAM_A:
                enemies = [
                    i for i in self._get_team_visible_enemies(TEAM_A)
                    if self.alive_mask[i]
                ]
            else:
                enemies = [i for i in range(self.n_agents_total) if self.alive_mask[i] and i < self.num_team_a]

            if len(friends) == 0:
                return 0.0

            cy = float(np.mean([self.agents_pos[i][0] for i in friends]))
            cx = float(np.mean([self.agents_pos[i][1] for i in friends]))

            f_cnt = 0
            e_cnt = 0
            for i in friends:
                y, x = self.agents_pos[i]
                if abs(y - cy) + abs(x - cx) <= radius:
                    f_cnt += 1
            for j in enemies:
                y, x = self.agents_pos[j]
                if abs(y - cy) + abs(x - cx) <= radius:
                    e_cnt += 1

            if e_cnt == 0:
                return float(f_cnt) if f_cnt > 0 else 1.0
            return f_cnt / max(1, e_cnt)

        def _nearest_enemy_to_team(team_id: int):
            """只用本队可见敌人，避免把不可见敌人位置写进 A 队教范目标。"""
            if team_id == TEAM_A:
                idx0, idx1 = 0, self.num_team_a
                friends = [i for i in range(idx0, idx1) if self.alive_mask[i]]
                visible_enemies = set(self._get_team_visible_enemies(TEAM_A))
                if len(friends) == 0 or len(visible_enemies) == 0:
                    return None
                best = None
                best_d = None
                for i in friends:
                    sy, sx = self.agents_pos[i]
                    for j in visible_enemies:
                        ty, tx = self.agents_pos[j]
                        d = abs(ty - sy) + abs(tx - sx)
                        if best_d is None or d < best_d:
                            best_d = d
                            best = (i, j)
                return best  # (friend_idx, enemy_idx)
            # B队这里不再用 team 级视角
            return None

        def _nearest_enemy_to_squad(team_id: int, local_indices):
            """
            B队小队级视角：
            只统计当前 squad 里成员“自己能看见”的敌人，
            然后从这些可见敌人中选一个离本小队最近的目标。
            返回 (friend_idx, enemy_idx)
            """
            if team_id != TEAM_B:
                return _nearest_enemy_to_team(team_id)

            friends = []
            visible_pairs = []  # (friend_idx, enemy_idx, dist)

            for li in local_indices:
                gi = idx_start + li
                if not self.alive_mask[gi]:
                    continue
                friends.append(gi)

                sy, sx = self.agents_pos[gi]
                wt = int(self.agent_weapon_type[gi]) if self.agent_weapon_type is not None else 0
                eff_range = self.gun_range + self.weapon_range_bonus.get(wt, 0)

                for j in range(self.n_agents_total):
                    if not self.alive_mask[j]:
                        continue
                    if j < self.num_team_a:
                        ty, tx = self.agents_pos[j]
                        dist = abs(ty - sy) + abs(tx - sx)

                        if (dist <= eff_range and
                                self._within_fov(gi, ty, tx) and
                                self._has_line_of_sight(sy, sx, ty, tx)):
                            visible_pairs.append((gi, j, dist))

            if len(friends) == 0 or len(visible_pairs) == 0:
                return None

            # 用本小队质心选最近可见敌人
            cy = float(np.mean([self.agents_pos[i][0] for i in friends]))
            cx = float(np.mean([self.agents_pos[i][1] for i in friends]))

            best_friend = friends[0]
            best_enemy = None
            best_d = None
            for _, j, _ in visible_pairs:
                ty, tx = self.agents_pos[j]
                d = abs(ty - cy) + abs(tx - cx)
                if best_d is None or d < best_d:
                    best_d = d
                    best_enemy = j

            return (best_friend, best_enemy) if best_enemy is not None else None

        # ---------- 3) 高层指挥官：根据局势选择 strategy ----------
        def _decide_team_strategy(team_id: int) -> str:
            H, W = self.grid_h, self.grid_w

            if team_id == TEAM_A:
                idx0, idx1 = 0, self.num_team_a
                ent = self.entrance_A
            else:
                idx0, idx1 = self.num_team_a, self.n_agents_total
                ent = self.entrance_B

            alive_idx = [i for i in range(idx0, idx1) if self.alive_mask[i]]
            if len(alive_idx) == 0:
                return "DEFEND"

            hp_ratio_mean = float(np.mean([self.agents_hp[i] for i in alive_idx])) / max(1, self.max_hp)

            # 这里只保留 team 级 posture / 背景态势判断
            # 不再使用 team-contact 去触发整队 COUNTER_ATTACK
            fr_ratio = _force_ratio_around(team_id, radius=6)

            # 若室内 + 有人质：优先 RESCUE
            if self.map_type == "indoor" and len(self.hostages) > 0:
                if hp_ratio_mean > 0.4:
                    return "RESCUE"
                else:
                    # 血量太低时，先 CAUTIOUS 再视情况救人
                    return "CAUTIOUS"

            # 血量整体很低 -> CAUTIOUS
            if hp_ratio_mean < 0.10:
                return "CAUTIOUS"

            # 整体兵力明显处于劣势 -> CAUTIOUS
            if fr_ratio < 0.7:
                return "CAUTIOUS"

            # team级不再因为局部接敌就切成 COUNTER_ATTACK
            # 只保留“总体 posture”
            y0, y1, x0, x1 = ent
            ent_y = (y0 + y1) // 2
            ent_x = (x0 + x1) // 2
            dist_to_ent = np.mean([
                abs(self.agents_pos[i][0] - ent_y) + abs(self.agents_pos[i][1] - ent_x)
                for i in alive_idx
            ])

            # A队只有在极端劣势且仍贴近己方入口时才 DEFEND
            if team_id == TEAM_A:
                if dist_to_ent < min(H, W) * 0.08 and hp_ratio_mean < 0.10 and fr_ratio < 0.7:
                    return "DEFEND"
                else:
                    return "ADVANCE"
            else:
                if dist_to_ent < min(H, W) * 0.10 and hp_ratio_mean < 0.2:
                    return "DEFEND"
                else:
                    return "ADVANCE"

        strategy = _decide_team_strategy(team_id)
        squad_strategies = [strategy] * len(squads)
        # ---------- A队交火支援配置 ----------
        fight_center, flank_hint = (None, None)
        team_alert = False
        direct_support_max_dist = 3

        if team_id == TEAM_A:
            fight_center, flank_hint = self._get_recent_engagement_hint_for_A()
            visible_enemies = self._get_team_visible_enemies(TEAM_A)
            team_alert = len(visible_enemies) > 0
            prev_support_targets_A = list(self.current_support_target_A) if self.current_support_target_A is not None else [None] * self.num_team_a
            prev_support_modes_A = self.current_support_mode_A.copy() if self.current_support_mode_A is not None else np.zeros(self.num_team_a, dtype=np.int8)
            prev_support_timers_A = self.support_persist_timer_A.copy() if self.support_persist_timer_A is not None else np.zeros(self.num_team_a, dtype=np.int32)
            self.current_support_target_A = [None] * self.num_team_a
            self.current_support_mode_A = np.zeros(self.num_team_a, dtype=np.int8)
            self.support_persist_timer_A = np.zeros(self.num_team_a, dtype=np.int32)

        # ---------- 4) 预计算：每个士兵是否“能看到敌人” ----------
        can_see_mask = [False] * count
        squad_alerts = [False] * len(squads)

        for s_idx, local_indices in enumerate(squads):
            for li in local_indices:
                gi = idx_start + li
                if not self.alive_mask[gi]:
                    continue

                # ===== [改动] A队已经被派出去支援的 agent，不参与决定整个 squad_alert =====
                if team_id == TEAM_A and getattr(self, "current_support_target_A", None) is not None:
                    if self.current_support_target_A[gi] is not None:
                        continue

                if self._agent_can_see_enemy(gi):
                    can_see_mask[li] = True
                    squad_alerts[s_idx] = True

        # ---------- 4.1) 按 squad 局部接敌情况覆写 squad_strategies ----------
        # ---------- 4.1) 按 squad 局部接敌情况覆写 squad_strategies ----------
        for s_idx, local_indices in enumerate(squads):
            alive_members = []
            alive_main_members = []

            for li in local_indices:
                gi = idx_start + li
                if not self.alive_mask[gi]:
                    continue

                alive_members.append(gi)

                # A队：被派出去支援的 agent，不参与本 squad 主体中心/血量/兵力比计算
                if team_id == TEAM_A and getattr(self, "current_support_target_A", None) is not None:
                    if self.current_support_target_A[gi] is not None:
                        continue

                alive_main_members.append(gi)

            # 如果整个 squad 都没有活人
            if len(alive_members) == 0:
                squad_strategies[s_idx] = "DEFEND"
                continue

            # 主体成员优先；如果主体成员被支援抽空了，再退化为全部存活成员
            core_members = alive_main_members if len(alive_main_members) > 0 else alive_members

            squad_hp_ratio = float(np.mean([self.agents_hp[gi] for gi in core_members])) / max(1, self.max_hp)

            cy = float(np.mean([self.agents_pos[gi][0] for gi in core_members]))
            cx = float(np.mean([self.agents_pos[gi][1] for gi in core_members]))

            friend_cnt = 0
            enemy_cnt = 0
            local_radius = 6
            visible_enemy_set_A = set(self._get_team_visible_enemies(TEAM_A)) if team_id == TEAM_A else None

            for j in range(self.n_agents_total):
                if not self.alive_mask[j]:
                    continue

                jy, jx = self.agents_pos[j]
                if abs(jy - cy) + abs(jx - cx) > local_radius:
                    continue

                team_j = TEAM_A if j < self.num_team_a else TEAM_B
                if team_j == team_id:
                    friend_cnt += 1
                else:
                    if team_id == TEAM_A and j not in visible_enemy_set_A:
                        continue
                    enemy_cnt += 1

            squad_fr_ratio = friend_cnt / max(1, enemy_cnt)

            # 默认继承 team posture
            local_strategy = squad_strategies[s_idx]

            # 只有本 squad 接敌，才允许本 squad 进入 COUNTER_ATTACK
            if squad_alerts[s_idx]:
                if squad_hp_ratio > 0.3 and squad_fr_ratio > 1.1:
                    local_strategy = "COUNTER_ATTACK"
                elif local_strategy == "DEFEND":
                    local_strategy = "DEFEND"
                else:
                    local_strategy = "CAUTIOUS"
            else:
                # 没接敌的 squad 不因为别处接敌而被连坐成 counter
                if local_strategy == "COUNTER_ATTACK":
                    local_strategy = "ADVANCE"

            squad_strategies[s_idx] = local_strategy

        # ===== [新增] A队掉队 squad 纠正：仅前中期启用；按相对位置输出 ADVANCE / FLANK =====
        if (
                team_id == TEAM_A
                and self.current_step <= min(int(self.max_steps * 0.65), self.endgame_hotspot_start_step)
        ):
            valid_centers = [c for c in squad_centers if c is not None]
            if len(valid_centers) > 0:
                front_y = max(c[0] for c in valid_centers)  # A队从上往下推，y越大越靠前
                front_x_mean = float(np.mean([c[1] for c in valid_centers]))

                # 用全队横向离散度估一个“中路带宽”
                xs = [c[1] for c in valid_centers]
                x_std = float(np.std(xs)) if len(xs) > 1 else 0.0
                mid_band = max(2.0, min(self.grid_w * 0.12, 1.2 * x_std + 1.5))

                for s_idx, center in enumerate(squad_centers):
                    if center is None:
                        continue
                    if squad_alerts[s_idx]:
                        continue  # 已接敌的不动
                    if squad_strategies[s_idx] == "RESCUE":
                        continue  # 支援中的不动
                    if squad_strategies[s_idx] == "DEFEND":
                        continue  # 极低血/明显防守态的不硬推

                    cy, cx = center
                    y_lag = front_y - cy
                    dx_from_center = cx - front_x_mean

                    # 只纠正“明显掉队”的 squad；轻微落后不管
                    if y_lag < 5:
                        continue

                    # 按横向位置决定中路推进还是左右包抄
                    if dx_from_center <= -mid_band:
                        squad_strategies[s_idx] = "FLANK_LEFT"
                    elif dx_from_center >= mid_band:
                        squad_strategies[s_idx] = "FLANK_RIGHT"
                    else:
                        squad_strategies[s_idx] = "ADVANCE"

        # ===== A队：多交火区支援 =====
        if team_id == TEAM_A:
            internal_support_radius = max(9, int(round(min(self.grid_h, self.grid_w) * 0.45)))
            reserve_reaction_radius = max(24, int(round((self.grid_h + self.grid_w) * 0.70)))
            recent_fight_capture_radius = max(6, int(round(min(self.grid_h, self.grid_w) * 0.30)))

            def _mean_point(points):
                if not points:
                    return None
                return (
                    float(np.mean([p[0] for p in points])),
                    float(np.mean([p[1] for p in points])),
                )

            def _safe_for_support(gi: int) -> bool:
                if not self.alive_mask[gi]:
                    return False
                if self._agent_can_see_enemy(gi):
                    return False
                perceived = self._perception_target(gi)
                if perceived is not None:
                    py, px = perceived
                    gy, gx = self.agents_pos[gi]
                    if abs(gy - py) + abs(gx - px) <= 4:
                        return False
                if getattr(self, "suppressed", None) is not None and self.suppressed[gi]:
                    return False

                hp_ratio = self.agents_hp[gi] / max(1, self.max_hp)
                morale = self.agent_morale[gi] if self.agent_morale is not None else 1.0
                return hp_ratio >= 0.25

            def _anchor_points_seen_by_squad(s_idx: int):
                pts = []
                for li in squads[s_idx]:
                    gi = idx_start + li
                    if not self.alive_mask[gi]:
                        continue

                    sy, sx = self.agents_pos[gi]
                    wt = int(self.agent_weapon_type[gi]) if self.agent_weapon_type is not None else 0
                    eff_range = self.gun_range + self.weapon_range_bonus.get(wt, 0)

                    for j in range(self.num_team_a, self.n_agents_total):
                        if not self.alive_mask[j]:
                            continue
                        ty, tx = self.agents_pos[j]
                        dist = abs(ty - sy) + abs(tx - sx)
                        if (
                                dist <= eff_range
                                and self._within_fov(gi, ty, tx)
                                and self._has_line_of_sight(sy, sx, ty, tx)
                               
                        ):
                            pts.append((float(ty), float(tx)))
                return pts

            for s_idx, local_indices in enumerate(squads):
                alive_members = [idx_start + li for li in local_indices if self.alive_mask[idx_start + li]]
                if not alive_members:
                    continue

                squad_centers[s_idx] = _mean_point([tuple(self.agents_pos[m]) for m in alive_members])
                pts = _anchor_points_seen_by_squad(s_idx)

                if not pts and fight_center is not None and squad_centers[s_idx] is not None:
                    cy, cx = squad_centers[s_idx]
                    fy, fx = fight_center
                    if abs(cy - fy) + abs(cx - fx) <= recent_fight_capture_radius:
                        pts = [(float(fy), float(fx))]

                anchor = _mean_point(pts)
                if anchor is not None:
                    squad_fire_anchors[s_idx] = anchor
                    engaged_squads.append(s_idx)

            same_squad_support_count = max(1, int(getattr(self, "teamA_same_squad_support_count", 2)))
            for s_idx in engaged_squads:
                ay, ax = squad_fire_anchors[s_idx]

                candidates = []
                for li in squads[s_idx]:
                    gi = idx_start + li
                    if not self.alive_mask[gi]:
                        continue
                    if can_see_mask[li]:
                        continue  # 正在直接交火的人不算“支援者”

                    gy, gx = self.agents_pos[gi]
                    d = abs(gy - ay) + abs(gx - ax)

                    if not _safe_for_support(gi):
                        continue
                    if d <= internal_support_radius:
                        candidates.append((d, li, gi))

                candidates.sort(key=lambda x: x[0])

                if len(candidates) == 0:
                    continue

                for _, li, gi in candidates[:same_squad_support_count]:
                    ty = int(np.clip(round(ay), 1, self.grid_h - 2))
                    tx = int(np.clip(round(ax), 1, self.grid_w - 2))
                    ty, tx = self._find_nearest_free_cell_for_team(
                        ty, tx, TEAM_A, ignore_idx=gi, max_r=4
                    )
                    self.current_support_target_A[gi] = (ty, tx)
                    self.current_support_mode_A[gi] = 1
                    self.support_persist_timer_A[gi] = int(getattr(self, "teamA_support_persist_steps", 3))

            # 3) squad 外：每个安全 squad 最多派 1 名最近成员去最近交火区，避免整队被抽空
            cross_squad_support_count = max(0, int(getattr(self, "teamA_cross_squad_support_count", 1)))
            for s_idx, local_indices in enumerate(squads):
                if cross_squad_support_count <= 0:
                    break
                if s_idx in engaged_squads:
                    continue
                if squad_centers[s_idx] is None or len(engaged_squads) == 0:
                    continue

                cy, cx = squad_centers[s_idx]
                nearest_zone = None
                nearest_zone_d = None
                for z_idx in engaged_squads:
                    ay, ax = squad_fire_anchors[z_idx]
                    d = abs(cy - ay) + abs(cx - ax)
                    if d <= reserve_reaction_radius and (nearest_zone_d is None or d < nearest_zone_d):
                        nearest_zone = z_idx
                        nearest_zone_d = d

                if nearest_zone is None:
                    continue

                ay, ax = squad_fire_anchors[nearest_zone]
                reserve_candidates = []
                for li in local_indices:
                    gi = idx_start + li
                    if self.current_support_target_A[gi] is not None:
                        continue
                    if not _safe_for_support(gi):
                        continue
                    gy, gx = self.agents_pos[gi]
                    reserve_candidates.append((li, abs(gy - ay) + abs(gx - ax)))

                reserve_candidates.sort(key=lambda x: x[1])

                for li, _ in reserve_candidates[:cross_squad_support_count]:
                    gi = idx_start + li
                    ty = int(np.clip(round(ay), 1, self.grid_h - 2))
                    tx = int(np.clip(round(ax), 1, self.grid_w - 2))
                    ty, tx = self._find_nearest_free_cell_for_team(
                        ty, tx, TEAM_A, ignore_idx=gi, max_r=5
                    )
                    self.current_support_target_A[gi] = (ty, tx)
                    self.current_support_mode_A[gi] = 2
                    self.support_persist_timer_A[gi] = int(getattr(self, "teamA_support_persist_steps", 3))

            self._apply_endgame_hotspot_support_A(
                squads=squads,
                idx_start=idx_start,
                can_see_mask=can_see_mask,
                squad_centers=squad_centers,
                squad_fire_anchors=squad_fire_anchors,
                engaged_squads=engaged_squads,
            )

            # 4) 若本步没有重新分配支援，则短时保留上一目标，减少任务抖动
            for gi in range(self.num_team_a):
                if not self.alive_mask[gi]:
                    continue
                if self.current_support_target_A[gi] is not None:
                    continue
                if self._agent_can_see_enemy(gi) or self._has_shootable_enemy(gi):
                    continue
                if not _safe_for_support(gi):
                    continue
                if prev_support_timers_A[gi] <= 0:
                    continue
                prev_tgt = prev_support_targets_A[gi]
                if prev_tgt is None:
                    continue

                ty, tx = prev_tgt
                if fight_center is not None:
                    fy, fx = fight_center
                    if abs(ty - fy) + abs(tx - fx) > reserve_reaction_radius:
                        continue

                self.current_support_target_A[gi] = (int(ty), int(tx))
                self.current_support_mode_A[gi] = int(prev_support_modes_A[gi])
                self.support_persist_timer_A[gi] = int(prev_support_timers_A[gi] - 1)

        # ---------- 5) 工具：找掩体 / 决定小队目标 ----------
        H, W = self.grid_h, self.grid_w
        map_center = (H // 2, W // 2)

        def _nearest_cover(y: int, x: int, max_radius: int = 5):
            best = None
            best_d = None
            for r in range(1, max_radius + 1):
                for dy in range(-r, r + 1):
                    dx = r - abs(dy)
                    for sx_sign in [-1, 1]:
                        yy = y + dy
                        xx = x + sx_sign * dx
                        if yy < 0 or yy >= H or xx < 0 or xx >= W:
                            continue
                        if self.terrain[yy, xx] == COVER:
                            # 避免直接走到已被占据的格子（简单过滤）
                            occupied = False
                            for k in range(self.n_agents_total):
                                if self.alive_mask[k] and tuple(self.agents_pos[k]) == (yy, xx):
                                    occupied = True
                                    break
                            if occupied:
                                continue
                            d = abs(yy - y) + abs(xx - x)
                            if best_d is None or d < best_d:
                                best_d = d
                                best = (yy, xx)
            return best

        # -----新增B队不重复之后随机扰动------------
        def _get_b_spread_offset(s_idx: int, local_rank: int):
            # 加强横向和纵向拉开，避免前沿连成连续火力线
            offsets = [
                (0, 0),
                (-6, -3), (6, 3), (-2, -8), (2, 8),
                (-5, 4), (5, -4), (-8, 2), (8, -2),
                (-4, -8), (4, 8), (-9, 0), (9, 0),
                (0, -9), (0, 9), (-6, 6), (6, -6),
            ]
            dy, dx = offsets[local_rank % len(offsets)]

            # lane bias 缩小，避免整个 squad 一起正面铺开
            lane_id = s_idx % 3
            if lane_id == 0:
                dx -= 1
            elif lane_id == 2:
                dx += 1

            # 再加大一点抖动，减少多条 lane 在同一 y 上排成火力线
            dy += self.np_random.randint(-4, 5)
            dx += self.np_random.randint(-4, 5)

            return dy, dx

        # --------结束——————————————————

        def _choose_objective_for_leader(leader_idx: int, strategy: str, s_idx: int):
            if leader_idx is None:
                return map_center
            sy, sx = self.agents_pos[leader_idx]

            # ===== B队早期浅层警戒：尽快离开出生区，但只在B半场前沿活动 =====
            if team_id == TEAM_B and self.current_step < 120:
                route = self.patrol_routes_B[s_idx % len(self.patrol_routes_B)]
                early_limit_y = self._get_team_b_early_limit_y()

                launch_table = getattr(self, "b_squad_launch_step", None)
                launch_step = int(launch_table[s_idx]) if (
                            launch_table is not None and s_idx < len(launch_table)) else 0

                # 没到出发时间：也要先往出生区外的浅层警戒带移动，避免门口抱团
                if self.current_step < launch_step:
                    hold_y, hold_x = self._get_team_b_forward_hold_point(
                        s_idx, leader_idx=leader_idx
                    )
                    py, px = _clip_objective_to_b_lane(hold_y, hold_x, s_idx, leader_idx=leader_idx)
                    return (min(py, early_limit_y), px)

                local_t = self.current_step - launch_step
                squad_phase = (5 * s_idx + self.np_random.randint(0, 4)) % 10

                if s_idx % 3 == 0:
                    patrol_div = 20
                elif s_idx % 3 == 1:
                    patrol_div = 18
                else:
                    patrol_div = 16

                waypoint_idx = ((local_t + squad_phase) // patrol_div) % len(route)
                base_y, base_x = route[waypoint_idx]
                base_y = min(base_y, early_limit_y)

                base_y += self.np_random.randint(-1, 2)
                base_x += self.np_random.randint(-2, 3)

                py, px = _clip_objective_to_b_lane(base_y, base_x, s_idx, leader_idx=leader_idx)
                return (min(py, early_limit_y), px)

            # ========== B队盲区行为：没看见敌人就走这里 ==========
            if team_id == TEAM_B:
                pair = _nearest_enemy_to_squad(team_id, squads[s_idx])
                if pair is None:
                    if self.current_step < 120:
                        pass
                    else:
                        # 放慢 patrol -> attack 切换，继续保留较强巡逻倾向，避免三路推进后过强
                        # 160 -> 260 之间才逐渐从 0 平滑到 1
                        attack_progress = np.clip((self.current_step - 160) / 100.0, 0.0, 1.0)

                        # 继续巡逻的概率：先高，后期再缓慢下降
                        patrol_prob = 0.96 - 0.16 * attack_progress

                        # 巡逻速度不再明显加快，保持更像巡逻方
                        patrol_div = 26 if attack_progress < 0.5 else 22

                        if self.np_random.rand() < patrol_prob:
                            route = self.patrol_routes_B[s_idx % len(self.patrol_routes_B)]
                            waypoint_idx = (self.current_step // patrol_div) % len(route)
                            base_y, base_x = route[waypoint_idx]

                            py, px = _clip_objective_to_b_lane(
                                base_y, base_x, s_idx, leader_idx=leader_idx
                            )
                            return (py, px)
                        else:
                            # 三路盲推：仍然保守，不直接给到A门口，让A保留展开空间
                            y0, y1, x0, x1 = self.entrance_A
                            ent_cy = (y0 + y1) // 2
                            ent_cx = (x0 + x1) // 2

                            lane_id = s_idx % 3
                            lane_push = max(2, self.grid_w // 7)
                            if lane_id == 0:
                                lane_bias_x = -lane_push
                            elif lane_id == 2:
                                lane_bias_x = lane_push
                            else:
                                lane_bias_x = 0

                            # 目标停在中场偏B侧，不让盲推过深
                            base_y = int(np.clip(
                                0.65 * self._get_team_b_early_limit_y() + 0.35 * ent_cy
                                + self.np_random.randint(-2, 3),
                                1,
                                self.grid_h - 2,
                            ))
                            base_x = int(np.clip(ent_cx + lane_bias_x + self.np_random.randint(-4, 5), 1, self.grid_w - 2))

                            ent_y, ent_x = _clip_objective_to_b_lane(
                                base_y, base_x, s_idx, leader_idx=leader_idx
                            )
                            return (ent_y, ent_x)

            # 如果上面都没return，就走你原来的其他逻辑（追可见敌人、找掩体等）

            # 找最近敌人
            if team_id == TEAM_B:
                pair = _nearest_enemy_to_squad(team_id, squads[s_idx])
            else:
                pair = _nearest_enemy_to_team(team_id)

            nearest_enemy_idx = pair[1] if pair is not None else None

            # COUNTER_ATTACK：向最近敌人略微靠近，但优先找掩体
            if strategy == "COUNTER_ATTACK" and nearest_enemy_idx is not None:
                ey, ex = self.agents_pos[nearest_enemy_idx]
                cv = _nearest_cover(ey, ex, max_radius=4)
                if cv is not None:
                    return cv
                return (ey, ex)

            if strategy == "ADVANCE":
                if team_id == TEAM_A:
                    y0, y1, x0, x1 = self.entrance_B
                else:
                    y0, y1, x0, x1 = self.entrance_A
                ent_y = (y0 + y1) // 2
                ent_x = (x0 + x1) // 2
                # 小队总数
                if team_id == TEAM_A:
                    num_squads = max(1, int(np.ceil(self.num_team_a / squad_size)))
                else:
                    num_squads = max(1, int(np.ceil(self.num_team_b / squad_size)))

                # 默认：主力直推
                ty, tx = ent_y, ent_x

                # A队：主力推进，小股包抄
                # 约定：
                #   最左 1 支小队 -> 左侧偏移
                #   最右 1 支小队 -> 右侧偏移
                #   中间其余小队 -> 主力直推
                if team_id == TEAM_A and num_squads >= 3:
                    flank_offset = max(2, W // 10)

                    if s_idx == 0:
                        tx = max(0, ent_x - flank_offset)
                    elif s_idx == num_squads - 1:
                        tx = min(W - 1, ent_x + flank_offset)
                    else:
                        tx = ent_x

                if team_id == TEAM_B:
                    lane_id = s_idx % 3
                    lane_offset = max(2, W // 8)
                    if lane_id == 0:
                        tx -= lane_offset
                    elif lane_id == 2:
                        tx += lane_offset

                    # B队推进目标保持一些抖动和纵向保守，避免三路推进后压迫感过强
                    ty += self.np_random.randint(1, 4)
                    tx += self.np_random.randint(-2, 3)
                    ty, tx = _clip_objective_to_b_lane(ty, tx, s_idx, leader_idx=leader_idx)

                return (ty, tx)

            if strategy in ["FLANK_LEFT", "FLANK_RIGHT"]:
                if nearest_enemy_idx is not None:
                    ey, ex = self.agents_pos[nearest_enemy_idx]
                else:
                    if team_id == TEAM_A:
                        y0, y1, x0, x1 = self.entrance_B
                    else:
                        y0, y1, x0, x1 = self.entrance_A
                    ey = (y0 + y1) // 2
                    ex = (x0 + x1) // 2

                if strategy == "FLANK_LEFT":
                    ex = max(0, ex - 3)
                elif strategy == "FLANK_RIGHT":
                    ex = min(W - 1, ex + 3)

                if team_id == TEAM_B:
                    ey, ex = _clip_objective_to_b_lane(ey, ex, s_idx, leader_idx=leader_idx)
                return (ey, ex)

            if strategy == "RESCUE" and len(self.hostage_rooms) > 0:
                (ry0, ry1, rx0, rx1) = self.hostage_rooms[0]
                return ((ry0 + ry1) // 2, (rx0 + rx1) // 2)

            if strategy == "DEFEND":
                if team_id == TEAM_A:
                    y0, y1, x0, x1 = self.entrance_A
                else:
                    y0, y1, x0, x1 = self.entrance_B
                ent_y = (y0 + y1) // 2
                ent_x = (x0 + x1) // 2
                cv = _nearest_cover(ent_y, ent_x, max_radius=5)
                return cv if cv is not None else (ent_y, ent_x)

            if strategy == "CAUTIOUS":
                if nearest_enemy_idx is not None:
                    ey, ex = self.agents_pos[nearest_enemy_idx]
                else:
                    if team_id == TEAM_A:
                        y0, y1, x0, x1 = self.entrance_B
                    else:
                        y0, y1, x0, x1 = self.entrance_A
                    ey = (y0 + y1) // 2
                    ex = (x0 + x1) // 2

                cy, cx = self.agents_pos[leader_idx]
                dy = ey - cy
                dx = ex - cx

                step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
                step_x = 0 if dx == 0 else (1 if dx > 0 else -1)

                probe_y = int(np.clip(cy + 3 * step_y, 1, H - 2))
                probe_x = int(np.clip(cx + 3 * step_x, 1, W - 2))

                cv = _nearest_cover(probe_y, probe_x, max_radius=4)
                if cv is not None:
                    return cv
                return (probe_y, probe_x)

            return map_center

        squad_objectives = []
        for s_idx, leader_idx in enumerate(squad_leaders):
            local_strategy = squad_strategies[s_idx]
            obj = _choose_objective_for_leader(leader_idx, local_strategy, s_idx)
            squad_objectives.append(obj)
        # A队：把支援目标写回 squad objective，便于 RL 直接观察到“本 squad 当前该去哪里支援”
        if team_id == TEAM_A and getattr(self, "current_support_target_A", None) is not None:
            for s_idx, local_indices in enumerate(squads):
                support_points = []
                for li in local_indices:
                    gi = idx_start + li
                    if not self.alive_mask[gi]:
                        continue
                    tgt = self.current_support_target_A[gi]
                    if tgt is not None:
                        support_points.append(tgt)

                if support_points:
                    oy = int(round(np.mean([p[0] for p in support_points])))
                    ox = int(round(np.mean([p[1] for p in support_points])))
                    oy = int(np.clip(oy, 1, self.grid_h - 2))
                    ox = int(np.clip(ox, 1, self.grid_w - 2))
                    oy, ox = self._find_nearest_free_cell_for_team(
                        oy, ox, TEAM_A, ignore_idx=squad_leaders[s_idx], max_r=5
                    )
                    squad_objectives[s_idx] = (oy, ox)

        # 记录当前队伍的“教范策略”和小队目标，供分层封装读取
        if team_id == TEAM_A:
            self.current_strategy_A = strategy
            self.current_squad_strategies_A = list(squad_strategies)
            self.current_squad_objectives_A = list(squad_objectives)
        else:
            self.current_strategy_B = strategy
            self.current_squad_strategies_B = list(squad_strategies)
            self.current_squad_objectives_B = list(squad_objectives)

        # ---------- 6) 第一轮：根据 strategy & 视野初步生成动作 ----------
        for s_idx, local_indices in enumerate(squads):
            leader_idx = squad_leaders[s_idx]
            obj_y, obj_x = squad_objectives[s_idx]
            squad_alert = squad_alerts[s_idx]
            local_strategy = squad_strategies[s_idx]

            for li in local_indices:
                gi = idx_start + li
                if not self.alive_mask[gi]:
                    actions[li] = 0
                    continue

                hp_ratio = self.agents_hp[gi] / max(1, self.max_hp)
                morale = self.agent_morale[gi] if self.agent_morale is not None else 1.0
                gy, gx = self.agents_pos[gi]
                suspect_tgt = None
                if team_id == TEAM_B and not can_see_mask[li]:
                    perceived = self._perception_target(gi)
                    if perceived is not None:
                        suspect_tgt = (int(perceived[0]), int(perceived[1]))
                    else:
                        heard = self._loudest_sound_near(gi, radius=5)
                        if heard is not None:
                            suspect_tgt = (int(heard[0]), int(heard[1]))

                # 只对本小队接敌，不再全队联动
                if squad_alert and can_see_mask[li]:
                    if local_strategy in ["CAUTIOUS", "DEFEND"] and hp_ratio < 0.2:
                        cy, cx = self.agents_pos[gi]
                        cv = _nearest_cover(cy, cx, max_radius=5)
                        if cv is not None:
                            actions[li] = self._orient_and_move_towards(gi, cv[0], cv[1])
                        else:
                            if team_id == TEAM_A:
                                y0, y1, x0, x1 = self.entrance_A
                            else:
                                y0, y1, x0, x1 = self.entrance_B
                            ent_y = (y0 + y1) // 2
                            ent_x = (x0 + x1) // 2
                            actions[li] = self._orient_and_move_towards(gi, ent_y, ent_x)
                    else:
                        if team_id == TEAM_B:
                            visible_enemy_idx = self._get_nearest_visible_enemy_for_agent(gi)
                            shootable_now = self._has_shootable_enemy(gi)

                            # 2026-04-27: B队在明确可射击时更果断一些，
                            # 但仍保留少量转向/找掩体的战术分支，避免火力过满。
                            if shootable_now and self.np_random.rand() < 0.84:
                                actions[li] = 9  # Shoot
                            elif visible_enemy_idx is not None:
                                ey, ex = self.agents_pos[visible_enemy_idx]
                                if self.np_random.rand() < 0.78:
                                    actions[li] = self._orient_only_towards(gi, ey, ex)
                                else:
                                    cv = _nearest_cover(gy, gx, max_radius=3)
                                    if cv is not None and self.np_random.rand() < 0.45:
                                        actions[li] = self._orient_and_move_towards(gi, cv[0], cv[1])
                                    else:
                                        actions[li] = self._orient_and_move_towards(gi, ey, ex)
                            else:
                                actions[li] = self._random_scan_action_B()
                        else:
                            actions[li] = 9  # Shoot
                    continue

                if team_id == TEAM_B and suspect_tgt is not None:
                    sy, sx = suspect_tgt
                    face_score = self._facing_score_towards(gi, sy, sx)
                    if face_score >= 0.68 and self.np_random.rand() < 0.10:
                        actions[li] = 9  # 低概率压制性盲射/误射
                        continue
                    if face_score < 0.60 and self.np_random.rand() < 0.22:
                        turn_act = self._orient_only_towards(gi, sy, sx)
                        if turn_act != 0:
                            actions[li] = turn_act
                            continue

                # 2) A队：squad内/跨squad支援都优先去自己的支援目标
                support_tgt = None
                support_mode = 0
                if team_id == TEAM_A and getattr(self, "current_support_target_A", None) is not None:
                    support_tgt = self.current_support_target_A[gi]
                    if getattr(self, "current_support_mode_A", None) is not None:
                        support_mode = int(self.current_support_mode_A[gi])

                if support_tgt is not None:
                    ty, tx = support_tgt
                    d_to_support = abs(gy - ty) + abs(gx - tx)
                    stop_dist = 2 if support_mode in (1, 3) else direct_support_max_dist

                    if d_to_support <= stop_dist:
                        # 接近支援区后仍由策略自己决定朝向；这里只保留原有的落脚点引导。
                        face_act = self._orient_only_towards(gi, ty, tx)
                        if face_act != 0:
                            actions[li] = face_act
                        else:
                            actions[li] = self._orient_and_move_towards(gi, ty, tx)
                    else:
                        actions[li] = self._orient_and_move_towards(gi, ty, tx)
                    continue

                # 3) 其他人：维持原有正常行动
                if leader_idx is not None and gi == leader_idx:
                    if team_id == TEAM_B and not self._agent_can_see_enemy(gi):
                        if self.np_random.rand() < 0.18:
                            actions[li] = self._random_scan_action_B()
                            continue
                        d_obj = abs(gy - obj_y) + abs(gx - obj_x)
                        if d_obj <= 1:
                            # leader 到点后继续做更大范围的 lane 内漂移，避免火力和朝向都锁死
                            if self.np_random.rand() < 0.40:
                                actions[li] = self._random_scan_action_B()
                            else:
                                next_y = int(np.clip(obj_y + self.np_random.randint(-3, 4), 1, self.grid_h - 2))
                                next_x = int(np.clip(obj_x + self.np_random.randint(-3, 4), 1, self.grid_w - 2))
                                next_y, next_x = _clip_objective_to_b_lane(
                                    next_y, next_x, s_idx, leader_idx=gi
                                )
                                actions[li] = self._orient_and_move_towards(gi, next_y, next_x)
                        else:
                            actions[li] = self._orient_and_move_towards(gi, obj_y, obj_x)
                    else:
                        actions[li] = self._orient_and_move_towards(gi, obj_y, obj_x)
                else:
                    if leader_idx is not None and self.alive_mask[leader_idx]:
                        if team_id == TEAM_B:
                            if not self._agent_can_see_enemy(gi) and self.np_random.rand() < 0.16:
                                actions[li] = self._random_scan_action_B()
                                continue
                            rank_in_squad = local_indices.index(li)
                            off_y, off_x = _get_b_spread_offset(s_idx, rank_in_squad)
                            ly, lx = self.agents_pos[leader_idx]

                            # B follower 不再硬贴 leader/objective，而是保留更强的“半自主游走”。
                            roam_y = int(round(0.55 * gy + 0.45 * obj_y))
                            roam_x = int(round(0.55 * gx + 0.45 * obj_x))

                            phase_x = ((gi + 2 * s_idx) % 5) - 2
                            phase_y = ((gi + s_idx) % 3) - 1

                            target_y = int(np.clip(roam_y + off_y + 2 * phase_y + self.np_random.randint(-3, 4), 1, self.grid_h - 2))
                            target_x = int(np.clip(roam_x + off_x + 2 * phase_x + self.np_random.randint(-3, 4), 1, self.grid_w - 2))

                            leader_gap = abs(gy - ly) + abs(gx - lx)
                            if leader_gap <= 3 and not self._agent_can_see_enemy(gi):
                                away_y = 0 if gy == ly else (1 if gy > ly else -1)
                                away_x = 0 if gx == lx else (1 if gx > lx else -1)
                                if away_y == 0 and away_x == 0:
                                    away_x = -1 if (gi + s_idx) % 2 == 0 else 1
                                target_y = int(np.clip(target_y + 2 * away_y, 1, self.grid_h - 2))
                                target_x = int(np.clip(target_x + 2 * away_x, 1, self.grid_w - 2))

                            target_y, target_x = _clip_objective_to_b_lane(
                                target_y, target_x, s_idx, leader_idx=gi
                            )
                            target_y, target_x = self._find_low_density_cell_for_team(
                                target_y, target_x, TEAM_B, ignore_idx=gi, max_r=11, density_radius=4
                            )
                            d_follow = abs(gy - target_y) + abs(gx - target_x)

                            if not self._agent_can_see_enemy(gi) and d_follow <= 2:
                                # 到近点后更倾向继续漂，而不是围住同一目标格站死。
                                if self.np_random.rand() < 0.30:
                                    actions[li] = self._random_scan_action_B()
                                else:
                                    drift_y = int(np.clip(target_y + self.np_random.randint(-4, 5), 1, self.grid_h - 2))
                                    drift_x = int(np.clip(target_x + self.np_random.randint(-4, 5), 1, self.grid_w - 2))
                                    drift_y, drift_x = _clip_objective_to_b_lane(
                                        drift_y, drift_x, s_idx, leader_idx=gi
                                    )
                                    actions[li] = self._orient_and_move_towards(gi, drift_y, drift_x)
                            else:
                                actions[li] = self._orient_and_move_towards(gi, target_y, target_x)
                        else:
                            ly, lx = self.agents_pos[leader_idx]
                            rank_in_squad = local_indices.index(li)
                            spread_offsets = [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, -2), (0, 2)]
                            off_y, off_x = spread_offsets[rank_in_squad % len(spread_offsets)]

                            blend = float(np.clip(self.teamA_follow_objective_blend, 0.0, 1.0))
                            anchor_y = int(round((1.0 - blend) * ly + blend * obj_y))
                            anchor_x = int(round((1.0 - blend) * lx + blend * obj_x))
                            target_y = int(np.clip(anchor_y + off_y, 1, self.grid_h - 2))
                            target_x = int(np.clip(anchor_x + off_x, 1, self.grid_w - 2))
                            target_y, target_x = self._find_low_density_cell_for_team(
                                target_y, target_x, TEAM_A, ignore_idx=gi, max_r=5, density_radius=3
                            )

                            leader_gap = abs(gy - ly) + abs(gx - lx)
                            target_gap = abs(gy - target_y) + abs(gx - target_x)
                            if (
                                not self._agent_can_see_enemy(gi)
                                and max(leader_gap, target_gap) >= self.teamA_follow_sprint_gap
                            ):
                                desired_ori = self._ori_from_delta(target_y - gy, target_x - gx)
                                if int(self.agents_orient[gi]) == desired_ori:
                                    actions[li] = 7
                                else:
                                    actions[li] = self._orient_and_move_towards(gi, target_y, target_x)
                            else:
                                actions[li] = self._orient_and_move_towards(gi, target_y, target_x)
                    else:
                        actions[li] = self._orient_and_move_towards(gi, obj_y, obj_x)

        # ---------- 7) 第二轮：极端低 HP + 士气低时强制撤退 ----------
        for local_i in range(count):
            gi = idx_start + local_i
            if not self.alive_mask[gi]:
                continue
            hp_ratio = self.agents_hp[gi] / max(1, self.max_hp)
            morale = self.agent_morale[gi] if self.agent_morale is not None else 1.0
            if hp_ratio < 0.05:  # self.critical_threshold : #and morale < 0.2:
                # 朝自己阵营入口方向移动（尝试远离战场）
                if team_id == TEAM_A:
                    y0, y1, x0, x1 = self.entrance_A
                else:
                    y0, y1, x0, x1 = self.entrance_B
                ent_y = (y0 + y1) // 2
                ent_x = (x0 + x1) // 2
                actions[local_i] = self._orient_and_move_towards(gi, ent_y, ent_x)

        # ==========================================================
        # 紧急强制降速：解决B队开局抱团猛冲问题（推荐先用这个）
        # ==========================================================
        if team_id == TEAM_B and self.current_step < 130:  # 前130步强制慢下来，可调80~150
            for li in range(count):
                if not self.alive_mask[idx_start + li]:
                    continue
                if actions[li] == 7:  # SprintForward → 改成 WalkForward
                    actions[li] = 3

                elif actions[li] == 3 and self.np_random.rand() < 0.25:
                    actions[li] = 0  # Idle（小停顿，增加真实感）

        return actions

    def _perception_target(self, i: int):
        """
        矩形静态感知：
          - 敌人不在掩体 (COVER)
          - 敌人在前/侧/后矩形感知范围内（前方距离远，侧面中等，后方较近）
          - 中间无遮挡（LOS）
        不要求敌人移动、不看脚步声，也不要求已经在 FOV 里（360°“感觉到”有人）。
        返回最近的这样的敌人 (ty, tx)，找不到则返回 None。
        """
        if not self.alive_mask[i]:
            return None

        sy, sx = self.agents_pos[i]
        team_i = TEAM_A if i < self.num_team_a else TEAM_B
        ori = int(self.agents_orient[i])

        best = None
        best_d = None

        for j in range(self.n_agents_total):
            if not self.alive_mask[j]:
                continue
            team_j = TEAM_A if j < self.num_team_a else TEAM_B
            if team_j == team_i:
                continue

            jy, jx = self.agents_pos[j]

            # 敌人在掩体里 -> 难以被“直接感知”（只能靠更近或脚步/射击声）
            if self.terrain[jy, jx] == COVER:
                continue

            dy = jy - sy
            dx = jx - sx
            pf, pl = self._project_to_local(ori, dy, dx)

            # 前后方向距离限制
            if pf >= 0:
                if pf > self.perc_front:
                    continue
            else:
                if -pf > self.perc_back:
                    continue

            # 左右宽度限制
            if abs(pl) > self.perc_side:
                continue

            # 视线被墙挡住则感知不到
            if not self._has_line_of_sight(sy, sx, jy, jx):
                continue

            d = abs(dy) + abs(dx)
            if (best_d is None) or (d < best_d):
                best_d = d
                best = (jy, jx)

        return best

    def _loudest_sound_near(self, i: int, radius: int = 4):
        """
        在一定半径内找到最大的 sound cell 作为“脚步声目标”。
        """
        sy, sx = self.agents_pos[i]
        best_cell = None
        best_val = self.hearing_threshold
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                    v = self.sound_map[ny, nx]
                    if v > best_val:
                        best_val = v
                        best_cell = (ny, nx)
        return best_cell

    def _orient_and_move_towards(self, i: int, ty: int, tx: int):
        """统一入口：自动分流 A/B"""
        if i < self.num_team_a:
            return self._orient_and_move_towards_A(i, ty, tx)
        else:
            return self._orient_and_move_towards_B(i, ty, tx)

    def _orient_and_move_towards_A(self, i: int, ty: int, tx: int) -> int:
        sy, sx = self.agents_pos[i]
        dy, dx = ty - sy, tx - sx

        if abs(dy) < 1 and abs(dx) < 1:
            return 0

        ideal_ori = self._ori_from_delta(dy, dx)
        ori = int(self.agents_orient[i])
        diff = (ideal_ori - ori) % 8

        if diff == 0:
            dist = abs(dy) + abs(dx)
            act = 3 if dist <= 3 else 7
        else:
            act = 2 if diff <= 4 else 1

        return act

    def _random_scan_action_B(self):
        r = self.np_random.rand()
        if r < 0.28:
            return 1  # TurnLeft
        elif r < 0.56:
            return 2  # TurnRight
        else:
            return 0  # Idle

    def _orient_and_move_towards_B(self, i: int, ty: int, tx: int) -> int:
        sy, sx = self.agents_pos[i]
        dy, dx = ty - sy, tx - sx

        if abs(dy) < 1 and abs(dx) < 1:
            return 0

        # B队未接敌时尽量不要把朝向稳定锁到 patrol/objective 上。
        # 让它在当前随机朝向下，用前后左右这些相对移动去“凑近区域”，
        # 这样位置还能巡逻，但枪口不会天然持续朝向 A 队方向。
        if not self._agent_can_see_enemy(i):
            ori = int(self.agents_orient[i])

            def move_delta_for_action(action: int):
                if action == 3:  # WalkForward
                    test_ori = ori
                    steps = 1
                elif action == 4:  # WalkBackward
                    test_ori = (ori + 4) % 8
                    steps = 1
                elif action == 5:  # StrafeLeft
                    test_ori = (ori - 2) % 8
                    steps = 1
                elif action == 6:  # StrafeRight
                    test_ori = (ori + 2) % 8
                    steps = 1
                elif action == 7:  # SprintForward
                    test_ori = ori
                    steps = 2
                else:
                    return None

                dy0, dx0 = self._dir2vec(test_ori)
                ny, nx = sy, sx
                moved = 0
                for _ in range(steps):
                    ty0 = ny + dy0
                    tx0 = nx + dx0
                    if not (0 <= ty0 < self.grid_h and 0 <= tx0 < self.grid_w):
                        break
                    if self.terrain[ty0, tx0] == WALL:
                        break
                    if self._is_cell_occupied_by_team(ty0, tx0, TEAM_B, ignore_idx=i):
                        break
                    ny, nx = ty0, tx0
                    moved += 1

                return (ny, nx, moved)

            move_candidates = []
            cur_d = abs(ty - sy) + abs(tx - sx)
            for act in (3, 4, 5, 6, 7):
                res = move_delta_for_action(act)
                if res is None:
                    continue
                ny, nx, moved = res
                if moved <= 0:
                    continue
                new_d = abs(ty - ny) + abs(tx - nx)
                gain = cur_d - new_d
                move_candidates.append((gain, act, moved))

            if move_candidates:
                move_candidates.sort(key=lambda x: (x[0], x[2]), reverse=True)
                best_gain = move_candidates[0][0]
                near_best = [act for gain, act, _ in move_candidates if gain >= best_gain - 1]

                # 2026-04-27: 让B队未接敌时有小概率先朝巡逻/目标点转头，
                # 但大多数时间仍保持原有的随机相对移动，避免天然火力朝向过强。
                turn_act = self._orient_only_towards(i, ty, tx)
                if turn_act in (1, 2) and self.np_random.rand() < 0.12:
                    return int(turn_act)
                if self.np_random.rand() < 0.72:
                    return int(self.np_random.choice(near_best))
                if self.np_random.rand() < 0.65:
                    return self._random_scan_action_B()
                return 0

        angle = np.arctan2(dy, dx)
        angle = (angle + np.pi / 2) % (2 * np.pi)
        ideal_ori = int(np.round(angle / (np.pi / 4))) % 8

        ori = int(self.agents_orient[i])
        diff = (ideal_ori - ori) % 8

        def free_cell(y, x):
            if not (0 <= y < self.grid_h and 0 <= x < self.grid_w):
                return False
            if self.terrain[y, x] == WALL:
                return False
            if self._is_cell_occupied_by_team(y, x, TEAM_B, ignore_idx=i):
                return False
            return True

        # 正前方
        fdy, fdx = self._dir2vec(ori)
        fy, fx = sy + fdy, sx + fdx

        # 左右侧绕行（相对当前朝向）
        left_ori = (ori - 2) % 8
        right_ori = (ori + 2) % 8

        ldy, ldx = self._dir2vec(left_ori)
        rdy, rdx = self._dir2vec(right_ori)

        ly, lx = sy + ldy, sx + ldx
        ry, rx = sy + rdy, sx + rdx

        if diff == 0:
            if free_cell(fy, fx):
                dist = abs(dy) + abs(dx)

                # 近距离目标基本走路，避免抱团猛冲
                if dist <= 4:
                    return 3

                # 中距离：大部分走路，小概率冲刺
                if dist <= 8:
                    return 7 if self.np_random.rand() < 0.20 else 3

                # 远距离：给一点冲刺概率，但仍然不是默认冲
                return 7 if self.np_random.rand() < 0.40 else 3

            left_score = abs(ly - ty) + abs(lx - tx) if free_cell(ly, lx) else 10 ** 9
            right_score = abs(ry - ty) + abs(rx - tx) if free_cell(ry, rx) else 10 ** 9

            if left_score < right_score:
                return 5
            elif right_score < left_score:
                return 6
            else:
                return self.np_random.choice([1, 2])

        if diff <= 4:
            return 2
        else:
            return 1

    # ======================================================================
    # 射击 & LOS
    # ======================================================================

    def _bresenham_line(self, y1, x1, y2, x2):
        points = []
        dy = y2 - y1
        dx = x2 - x1
        steps = max(abs(dy), abs(dx))
        if steps == 0:
            return [(y1, x1)]
        for k in range(steps + 1):
            yy = y1 + round(dy * k / steps)
            xx = x1 + round(dx * k / steps)
            points.append((yy, xx))
        return points

    def _has_line_of_sight(self, y1, x1, y2, x2):
        for (y, x) in self._bresenham_line(y1, x1, y2, x2)[1:-1]:
            if self.terrain[y, x] == WALL:
                return False
        return True

    def _agent_can_see_enemy(self, agent_idx: int) -> bool:
        """判断某个 agent 是否在武器射程 + 视野锥 + 直线 LOS 内看到任意敌人。"""
        if not self.alive_mask[agent_idx]:
            return False
        team_i = TEAM_A if agent_idx < self.num_team_a else TEAM_B
        sy, sx = self.agents_pos[agent_idx]

        # 当前武器射程
        if self.agent_weapon_type is not None:
            wtype = int(self.agent_weapon_type[agent_idx])
        else:
            wtype = 0
        base_range = getattr(self, "gun_range", 6)
        bonus = self.weapon_range_bonus.get(wtype, 0)
        max_range = base_range + bonus

        for j in range(self.n_agents_total):
            if not self.alive_mask[j]:
                continue
            team_j = TEAM_A if j < self.num_team_a else TEAM_B
            if team_j == team_i:
                continue
            ty, tx = self.agents_pos[j]
            dist = abs(ty - sy) + abs(tx - sx)
            if dist > max_range:
                continue
            if not self._within_fov(agent_idx, ty, tx):
                continue
            if not self._has_line_of_sight(sy, sx, ty, tx):
                continue
            return True
        return False

    def _choose_shot_target(self, shooter_idx):
        """
        只能对 视野锥 + 射程 + LOS 内的敌人开火。
        同时考虑弹道上的平民与人质。
        """
        if not self.alive_mask[shooter_idx]:
            return None
        team_s = TEAM_A if shooter_idx < self.num_team_a else TEAM_B
        sy, sx = self.agents_pos[shooter_idx]

        if self.agent_weapon_type is not None:
            wtype = int(self.agent_weapon_type[shooter_idx])
        else:
            wtype = 0
        max_range = self.gun_range + self.weapon_range_bonus.get(wtype, 0)  # <<< 修改

        # 先找所有可射击的敌人
        enemy_candidates = []
        for j in range(self.n_agents_total):
            if not self.alive_mask[j]:
                continue
            team_j = TEAM_A if j < self.num_team_a else TEAM_B
            if team_j == team_s:
                continue
            jy, jx = self.agents_pos[j]
            dist = abs(jy - sy) + abs(jx - sx)
            if (
                    dist <= max_range
                    and self._within_fov(shooter_idx, jy, jx)
                    and self._has_line_of_sight(sy, sx, jy, jx)
            ):
                enemy_candidates.append((j, dist))

        if not enemy_candidates:
            return None

        enemy_candidates.sort(key=lambda x: x[1])
        if team_s == TEAM_B and len(enemy_candidates) >= 2:
            top_k = min(4, len(enemy_candidates))
            candidate_pool = enemy_candidates[:top_k]
            weights = np.array([1.0 / np.sqrt(max(1.0, dist)) for _, dist in candidate_pool], dtype=np.float32)
            weights = weights / np.sum(weights)
            target_idx = int(candidate_pool[self.np_random.choice(np.arange(top_k), p=weights)][0])
        else:
            target_idx = enemy_candidates[0][0]
        ty, tx = self.agents_pos[target_idx]

        # 检查弹道上的平民 / 人质
        line_cells = self._bresenham_line(sy, sx, ty, tx)
        for (cy, cx) in line_cells[1:-1]:
            # 人质
            if self.map_type == "indoor" and self.hostage_alive and self.hostage_pos is not None:
                if (cy, cx) == self.hostage_pos:
                    return ("hostage", None)
            # 平民
            for c_idx, (py, px) in enumerate(self.civilians_pos):
                if self.civilians_alive[c_idx] and (py, px) == (cy, cx):
                    return ("civilian", c_idx)

        return ("enemy", target_idx)

    def _resolve_shooting(self, shots_A, shots_B):
        """
        射击结算（带武器类型 / 弹药 / 命中率 / 压制 / 士气）:

        - shots_A, shots_B: 列表[(shooter_idx, target_type, target_id)]
          target_type ∈ {"enemy", "civilian", "hostage"}

        - 敌人:
            * 若同一 step 有 >=2 名敌人同时打同一个目标 → 交叉火力：
                - 目标必死（HP=0）
                - 在射手中随机一人掉 1 HP（交火风险）
            * 单人射击 → 按武器 & 环境计算命中概率:
                - 武器基准命中率 weapon_base_hit[wt]
                - 掩体 COVER, 姿态 站/蹲/趴
                - 距离衰减（各武器射程不同）
                - 烟雾
                - 射手本回合是否移动 (moved_this_step)
                - 射手是否被压制 (suppressed_timer)
                - 护甲 (agent_armor)

            命中后扣 weapon_damage[wt] HP (shotgun 多弹丸)

        - 平民/人质:
            被选中即死亡（不走命中率）

        - 压制:
            每次对敌格 (ty,tx) 射击（无论命中与否）
            在其周围 suppression_radius 内的所有单位 suppressed_timer 提升到 suppression_steps。

        - 士气:
            * 每击杀一名敌人 → 射手所在队伍士气 + morale_gain_on_kill
            * 每有队友阵亡 → 同队其他人士气 - morale_loss_per_friend_death
            * 自己进入重伤区 (<critical_threshold) → 士气额外下降
        """

        stats = {
            "shots_total": 0,
            "hits_total": 0,
            "kills_team_A": 0,
            "kills_team_B": 0,
            "friendly_hp_loss": 0,
            "civilian_killed": 0,
            "hostage_killed": 0,
            "suppressed_units": 0,
        }

        all_shots = list(shots_A) + list(shots_B)
        stats["shots_total"] = len(all_shots)

        prev_alive = self.alive_mask.copy()

        # 1) 聚合目标
        victims_enemy = {}
        civilians_hit = set()
        hostage_hit = False

        for shooter_idx, ttype, tidx in all_shots:
            if ttype == "enemy":
                if tidx is None:
                    continue
                victims_enemy.setdefault(tidx, []).append(shooter_idx)
            elif ttype == "civilian":
                if tidx is not None:
                    civilians_hit.add(tidx)
            elif ttype == "hostage":
                hostage_hit = True

        # 2) 平民/人质
        for c_idx in civilians_hit:
            if 0 <= c_idx < len(self.civilians_alive):
                if self.civilians_alive[c_idx]:
                    self.civilians_alive[c_idx] = False
                    stats["civilian_killed"] += 1

        if self.map_type == "indoor" and self.hostage_alive and hostage_hit:
            self.hostage_alive = False
            stats["hostage_killed"] += 1

        # 3) 压制函数
        def apply_suppression(center_y: int, center_x: int):
            H, W = self.grid_h, self.grid_w
            r = self.suppression_radius
            newly = 0
            for idx in range(self.n_agents_total):
                if not self.alive_mask[idx]:
                    continue
                y, x = self.agents_pos[idx]
                if abs(y - center_y) + abs(x - center_x) <= r:
                    before = int(self.suppressed_timer[idx])
                    self.suppressed_timer[idx] = max(self.suppressed_timer[idx], self.suppression_steps)
                    if before == 0 and self.suppressed_timer[idx] > 0:
                        newly += 1
            stats["suppressed_units"] += newly

        # 4) 单次射击命中计算（支持不同武器）
        def single_shot(shooter_idx: int, target_idx: int):
            if not self.alive_mask[target_idx]:
                return 0  # 无伤害

            sy, sx = self.agents_pos[shooter_idx]
            ty, tx = self.agents_pos[target_idx]
            terrain_t = self.terrain[ty, tx]
            posture_t = int(self.agents_posture[target_idx])
            wt = int(self.agent_weapon_type[shooter_idx]) if self.agent_weapon_type is not None else 0

            # 基础命中率
            hit_prob = self.weapon_base_hit.get(wt, self.base_hit_prob)

            # 掩体
            if terrain_t == COVER:
                hit_prob *= 0.6

            # 姿态（0=站立, 1=蹲）
            if posture_t == 0:
                hit_prob *= 1.0
            else:
                hit_prob *= 0.7

            # 距离 & 武器射程
            dist = abs(ty - sy) + abs(tx - sx)
            # 有效射程 = self.gun_range + bonus
            eff_range = max(1, self.gun_range + self.weapon_range_bonus.get(wt, 0))
            # 超出有效射程的部分，加重衰减
            if dist > eff_range:
                extra = dist - eff_range
                dist_factor = 1.0 - self.dist_decay_per_tile * (extra + 1)
            else:
                dist_factor = 1.0 - self.dist_decay_per_tile * max(0, dist - 1)
            dist_factor = float(np.clip(dist_factor, self.dist_min_factor, 1.0))
            hit_prob *= dist_factor

            # Shotgun 近距离强化
            if wt == 2:  # Shotgun
                if dist <= self.shotgun_optimal_range:
                    hit_prob *= 1.2
                else:
                    hit_prob *= 0.6  # 远距离基本刮痧

            # 烟雾

            # 射手移动惩罚
            if self.moved_this_step[shooter_idx]:
                hit_prob *= (1.0 - self.moving_shoot_penalty)

            # 射手被压制
            if self.suppressed_timer[shooter_idx] > 0:
                hit_prob *= (1.0 - self.suppressed_penalty)

            # 护甲
            armor = float(self.agent_armor[target_idx]) if self.agent_armor is not None else 1.0
            hit_prob *= (1.0 / max(1.0, armor))

            hit_prob = float(np.clip(hit_prob, 0.05, 0.99))

            # 压制（无论是否命中）
            apply_suppression(ty, tx)

            dmg_total = 0

            # Shotgun: 多弹丸
            if wt == 2:
                for _ in range(self.shotgun_pellets):
                    if self.np_random.rand() < hit_prob:
                        dmg_total += self.weapon_damage[wt]
                        stats["hits_total"] += 1
            else:
                if self.np_random.rand() < hit_prob:
                    dmg_total += self.weapon_damage[wt]
                    stats["hits_total"] += 1

            return dmg_total

        def apply_hit_reorient(target_idx: int, shooters: list):
            if not shooters:
                return
            if not self.alive_mask[target_idx]:
                return
            if self._agent_can_see_enemy(target_idx):
                return

            # 小概率不转身，表示继续原动作/反应较慢
            if self.np_random.rand() < self.hit_reorient_keep_prob:
                return

            ty, tx = self.agents_pos[target_idx]

            # 多个射手时，用平均来袭方向，避免最后一枪覆盖一切
            sy = float(np.mean([self.agents_pos[s][0] for s in shooters if self.alive_mask[s]]))
            sx = float(np.mean([self.agents_pos[s][1] for s in shooters if self.alive_mask[s]]))

            dy = sy - ty
            dx = sx - tx

            if abs(dy) < 1e-6 and abs(dx) < 1e-6:
                return

            self.agents_orient[target_idx] = self._ori_from_delta(dy, dx)
            self._add_sound(ty, tx, base=0.5 * self.sound_turn)

        # 5) 处理所有敌方目标
        for target_idx, shooters in victims_enemy.items():
            if not self.alive_mask[target_idx]:
                continue

            ty, tx = self.agents_pos[target_idx]

            # 多人交叉火力
            if len(shooters) >= 2:
                apply_suppression(ty, tx)
                total_dmg = 0.0
                for shooter_idx in shooters:
                    dmg = single_shot(shooter_idx, target_idx)
                    if dmg > 0:
                        total_dmg += dmg
                # ===== 协同增益（双方共用同一套结算，保持规则公平）=====
                synergy = 1.0 + 0.25 * (1 - np.exp(-(len(shooters) - 1)))
                total_dmg *= synergy
                self.agents_hp[target_idx] -= total_dmg
                if (
                        target_idx < self.num_team_a
                        and total_dmg > 0
                        and self.agents_hp[target_idx] > 0
                ):
                    if self.np_random.rand() < self.hit_reorient_prob:
                        apply_hit_reorient(target_idx, shooters)
                stats["hits_total"] += 1

            else:
                shooter_idx = shooters[0]
                dmg = single_shot(shooter_idx, target_idx)
                if dmg > 0:
                    self.agents_hp[target_idx] -= dmg
                    if target_idx < self.num_team_a and self.agents_hp[target_idx] > 0:
                        if self.np_random.rand() < self.hit_reorient_prob:
                            apply_hit_reorient(target_idx, shooters)

        # 6) 更新存活 & 统计击杀
        self.agents_hp = np.maximum(self.agents_hp, 0)
        self.alive_mask = self.agents_hp > 0

        # 记录击杀情况 & 更新士气
        def adjust_morale_on_kill(killer_idx, victim_idx):
            team_k = TEAM_A if killer_idx < self.num_team_a else TEAM_B
            team_v = TEAM_A if victim_idx < self.num_team_a else TEAM_B
            if team_k == team_v:
                return
            # killer 队伍所有存活队友 morale+
            if self.agent_morale is None:
                return
            if team_k == TEAM_A:
                rng = range(0, self.num_team_a)
            else:
                rng = range(self.num_team_a, self.n_agents_total)
            for i in rng:
                if not self.alive_mask[i]:
                    continue
                self.agent_morale[i] = float(
                    np.clip(self.agent_morale[i] + self.morale_gain_on_kill, self.morale_min, self.morale_max)
                )

        # 统计击杀
        for j in range(self.num_team_a, self.n_agents_total):
            if prev_alive[j] and not self.alive_mask[j]:
                stats["kills_team_A"] += 1
                # A 队某人杀掉 B 队；这里追踪不到精确是谁杀的，只能给 A 队统一加 morale
                if self.agent_morale is not None:
                    for i in range(self.num_team_a):
                        if self.alive_mask[i]:
                            self.agent_morale[i] = float(
                                np.clip(self.agent_morale[i] + self.morale_gain_on_kill,
                                        self.morale_min, self.morale_max)
                            )

        for j in range(self.num_team_a):
            if prev_alive[j] and not self.alive_mask[j]:
                stats["kills_team_B"] += 1
                if self.agent_morale is not None:
                    for i in range(self.num_team_a, self.n_agents_total):
                        if self.alive_mask[i]:
                            self.agent_morale[i] = float(
                                np.clip(self.agent_morale[i] + self.morale_gain_on_kill,
                                        self.morale_min, self.morale_max)
                            )

        # 友军阵亡 → 士气下降
        if self.agent_morale is not None:
            # A 队内部阵亡
            deaths_A = [i for i in range(self.num_team_a) if prev_alive[i] and not self.alive_mask[i]]
            if deaths_A:
                for i in range(self.num_team_a):
                    if self.alive_mask[i]:
                        self.agent_morale[i] = float(
                            np.clip(self.agent_morale[i] - self.morale_loss_per_friend_death,
                                    self.morale_min, self.morale_max)
                        )
            # B 队内部阵亡
            deaths_B = [i for i in range(self.num_team_a, self.n_agents_total)
                        if prev_alive[i] and not self.alive_mask[i]]
            if deaths_B:
                for i in range(self.num_team_a, self.n_agents_total):
                    if self.alive_mask[i]:
                        self.agent_morale[i] = float(
                            np.clip(self.agent_morale[i] - self.morale_loss_per_friend_death,
                                    self.morale_min, self.morale_max)
                        )

            # 自己重伤 → 士气下降
            for i in range(self.n_agents_total):
                if not self.alive_mask[i]:
                    continue
                hp_ratio = self.agents_hp[i] / max(1, self.max_hp)
                if hp_ratio < self.critical_threshold:
                    self.agent_morale[i] = float(
                        np.clip(self.agent_morale[i] - self.morale_loss_when_critical,
                                self.morale_min, self.morale_max)
                    )

        self.last_combat_stats = stats

        # ======================================================================

    # 手雷 & 烟雾：丢出去→生效
    # ======================================================================

    # ======================================================================
    # 平民移动
    # ======================================================================

    def _step_civilians(self):
        if self.map_type != "city":
            return
        for i in range(len(self.civilians_pos)):
            if not self.civilians_alive[i]:
                continue
            y, x = self.civilians_pos[i]
            if self.np_random.rand() < 0.3:
                dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
                dy, dx = dirs[self.np_random.randint(0, len(dirs))]
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                    if self.terrain[ny, nx] != WALL:
                        self.civilians_pos[i] = (ny, nx)

    # ======================================================================
    # 奖励 & 终止
    # ======================================================================

    # ======================================================================
    # 奖励 & 终止（改进版：最小伤亡 + 快速歼灭 / 解救人质）
    # ======================================================================

    # ======================================================================
    # 进攻推进奖励: Team A 到目标点的平均距离
    # ======================================================================
    def _get_teamA_progress_target(self):
        # 定义一个“推进目标点”:
        #   - indoor: 使用人质房间中心
        #   - 其他地图: 使用 B 队入口中心
        #   - 如果上述都不可用，则默认地图中心
        H, W = self.grid_size

        # 非残局阶段固定压 B 入口，不因 B 队巡逻/前压而把推进目标拉回中场。
        if self.entrance_B is not None and not self._endgame_hotspot_mode_on():
            y0, y1, x0, x1 = self.entrance_B
            cy = (y0 + y1) / 2.0
            cx = (x0 + x1) / 2.0
            return cy, cx

        # 残局阶段优先切到最近交火热点；没有热点时退回 B 出生点，
        # 不使用真实 B 队位置做全图透视目标。
        if self._endgame_hotspot_mode_on():
            fight_center, _ = self._get_recent_engagement_hint_for_A()
            if fight_center is not None:
                return fight_center

            fallback = self._get_teamB_spawn_fallback_target_for_agent_A(None)
            if fallback is not None:
                return float(fallback[0]), float(fallback[1])

        # 室内地图: 人质房间中心
        if self.map_type == "indoor" and getattr(self, "hostage_room_mask", None) is not None:
            ys, xs = np.where(self.hostage_room_mask)
            if len(ys) > 0:
                cy = float(ys.mean())
                cx = float(xs.mean())
                return cy, cx

        # 默认: B 队入口中心
        if self.entrance_B is not None:
            y0, y1, x0, x1 = self.entrance_B
            cy = (y0 + y1) / 2.0
            cx = (x0 + x1) / 2.0
            return cy, cx

        # 兜底: 地图几何中心
        return (H - 1) / 2.0, (W - 1) / 2.0

    def _teamA_progress_score(self):
        """
        A队推进分数：
          - 主体：所有存活成员到目标点的平均距离
          - 附加：最慢那部分成员（尾部）的平均距离
        分数越小越好。
        这样能防止前排推进、后排长期留在出生点附近。
        """
        target_y, target_x = self._get_teamA_progress_target()

        dists = []
        for i in range(self.num_team_a):
            if not self.alive_mask[i]:
                continue
            y, x = self.agents_pos[i]
            d = np.sqrt((y - target_y) ** 2 + (x - target_x) ** 2)
            dists.append(float(d))

        if len(dists) == 0:
            return 0.0

        dists = np.array(dists, dtype=np.float32)
        mean_dist = float(dists.mean())

        # 最慢的25%队员，作为“尾部”
        k = max(1, int(np.ceil(len(dists) * 0.25)))
        tail_dist = float(np.sort(dists)[-k:].mean())

        # 尾部给较大权重，逼后排跟上
        return mean_dist + 0.9 * tail_dist

    def _teamA_route_progress_score(self):
        """
        A队沿“出生点 -> 当前推进目标”主轴的纵深推进分数。
        分数越大表示整体越深入，用于鼓励穿过中场后继续向 B 出生点方向压进。
        """
        target_y, target_x = self._get_teamA_progress_target()

        if self.entrance_A is not None:
            ay0, ay1, ax0, ax1 = self.entrance_A
            start_y = (ay0 + ay1) / 2.0
            start_x = (ax0 + ax1) / 2.0
        else:
            start_y = (self.grid_h - 1) / 2.0
            start_x = (self.grid_w - 1) / 2.0

        dir_y = float(target_y - start_y)
        dir_x = float(target_x - start_x)
        route_len = max(1e-6, float(np.sqrt(dir_y * dir_y + dir_x * dir_x)))

        progresses = []
        for i in range(self.num_team_a):
            if not self.alive_mask[i]:
                continue
            y, x = self.agents_pos[i]
            proj = ((y - start_y) * dir_y + (x - start_x) * dir_x) / route_len
            progresses.append(float(np.clip(proj / route_len, 0.0, 1.25)))

        if len(progresses) == 0:
            return 0.0

        progresses = np.asarray(progresses, dtype=np.float32)
        mean_progress = float(progresses.mean())
        k = max(1, int(np.ceil(len(progresses) * 0.25)))
        tail_progress = float(np.sort(progresses)[:k].mean())
        return mean_progress + 0.8 * tail_progress

    def _iter_teamA_squads(self):
        squad_size = getattr(self, "default_squad_size", 5)
        if squad_size <= 0:
            squad_size = 5
        for start in range(0, self.num_team_a, squad_size):
            yield list(range(start, min(start + squad_size, self.num_team_a)))

    def _teamA_squad_progress_scores(self):
        """
        返回 A队每个 squad 的推进分数列表。
        分数越小越好；用于补充 squad 级 progress shaping。

        改动：
        - 优先只统计 squad 主体成员
        - 已被派出去支援的 agent 不参与本 squad 的推进分数
        - 若主体成员被支援抽空，则退化回全部存活成员，避免空 squad
        - 使用 squad 主体中心 + 相对前线落后量，替代尾部分位数约束。
          这个信号比排序后的 tail_dist 更平滑，更容易被 PPO 学到。
        """
        target_y, target_x = self._get_teamA_progress_target()

        support_targets_A = getattr(self, "current_support_target_A", None)
        squad_centers = []

        for squad in self._iter_teamA_squads():
            alive_members = []
            main_members = []

            for i in squad:
                if not self.alive_mask[i]:
                    continue

                alive_members.append(i)

                # 被派出去支援的 agent，不参与本 squad 主体推进统计
                if support_targets_A is not None and i < len(support_targets_A):
                    if support_targets_A[i] is not None:
                        continue

                main_members.append(i)

            # 主体成员优先；若主体成员为空，则退化回全部存活成员
            core_members = main_members if len(main_members) > 0 else alive_members

            if len(core_members) == 0:
                squad_centers.append(None)
                continue

            ys = [float(self.agents_pos[i][0]) for i in core_members]
            xs = [float(self.agents_pos[i][1]) for i in core_members]
            squad_centers.append((float(np.mean(ys)), float(np.mean(xs))))

        valid_centers = [c for c in squad_centers if c is not None]
        if len(valid_centers) == 0:
            return [0.0 for _ in squad_centers]

        if self.entrance_A is not None:
            ay0, ay1, ax0, ax1 = self.entrance_A
            start_y = (ay0 + ay1) / 2.0
            start_x = (ax0 + ax1) / 2.0
        else:
            start_y = (self.grid_h - 1) / 2.0
            start_x = (self.grid_w - 1) / 2.0

        dir_y = float(target_y - start_y)
        dir_x = float(target_x - start_x)
        dir_len = max(1e-6, float(np.sqrt(dir_y * dir_y + dir_x * dir_x)))

        progresses = []
        for center in squad_centers:
            if center is None:
                progresses.append(None)
                continue
            cy, cx = center
            progress = ((cy - start_y) * dir_y + (cx - start_x) * dir_x) / dir_len
            progresses.append(float(progress))

        valid_progresses = [p for p in progresses if p is not None]
        front_progress = max(valid_progresses) if valid_progresses else 0.0

        scores = []
        free_lag = 2.0
        lag_weight = 0.60
        use_line_lag = not self._endgame_hotspot_mode_on()
        for center, progress in zip(squad_centers, progresses):
            if center is None or progress is None:
                scores.append(0.0)
                continue

            cy, cx = center
            center_dist = float(np.sqrt((cy - target_y) ** 2 + (cx - target_x) ** 2))
            line_lag = max(0.0, front_progress - progress - free_lag) if use_line_lag else 0.0
            scores.append(center_dist + lag_weight * line_lag)

        return scores

    def _teamA_spawn_lag_count(self, lag_radius: int = 4):
        """
        统计仍然滞留在A队出生区附近的存活人数。
        用于对“有人长期留在出生点附近探索”单独惩罚。
        """
        if self.entrance_A is None:
            return 0

        y0, y1, x0, x1 = self.entrance_A
        ent_cy = (y0 + y1) / 2.0
        ent_cx = (x0 + x1) / 2.0

        cnt = 0
        for i in range(self.num_team_a):
            if not self.alive_mask[i]:
                continue
            y, x = self.agents_pos[i]
            d = abs(y - ent_cy) + abs(x - ent_cx)
            if d <= lag_radius:
                cnt += 1
        return cnt

    def _compute_reward_and_terminated(self, prev_hp):
        reward = 0.0
        info = {}
        terminated = False

        # Simplified reward path: keep only low-conflict shaping plus sparse task
        # outcomes. The previous dense shaping below is intentionally bypassed.
        reward -= 0.03

        stall_penalty = 0.0
        for i in range(self.num_team_a):
            if not self.alive_mask[i]:
                continue

            y, x = self.agents_pos[i]
            near_enemy = False
            for j in range(self.num_team_a, self.n_agents_total):
                if not self.alive_mask[j]:
                    continue
                ey, ex = self.agents_pos[j]
                if abs(y - ey) + abs(x - ex) <= 6:
                    near_enemy = True
                    break

            if near_enemy:
                continue

            pc = self.passive_counter[i]
            if pc >= 10:
                stall_penalty -= 0.04 * min(float(pc - 9), 6.0)

        stall_penalty = float(np.clip(stall_penalty, -1.0, 0.0))
        reward += stall_penalty
        info["stall_penalty"] = float(stall_penalty)
        self.ep_stall_penalty += float(stall_penalty)
        info["ep_stall_penalty"] = float(self.ep_stall_penalty)

        good_shot_reward = 0.0
        bad_shot_penalty = 0.0
        missed_shot_penalty = 0.0
        pre_visible = getattr(self, "pre_visible_enemy_A", np.zeros(self.num_team_a, dtype=bool))
        pre_shootable = getattr(self, "pre_shootable_enemy_A", np.zeros(self.num_team_a, dtype=bool))
        for i in range(self.num_team_a):
            if not self.alive_mask[i]:
                continue
            a = int(self.last_action_A[i])
            if a == 9:
                if bool(pre_shootable[i]):
                    good_shot_reward += 0.45
                elif not bool(pre_visible[i]):
                    bad_shot_penalty -= 0.25
                else:
                    # 看见但暂时不可射，轻罚即可，避免模型完全不敢开枪。
                    bad_shot_penalty -= 0.08
            elif bool(pre_shootable[i]) and a in (0, 1, 2, 8):
                # 有明确射击窗口时，Idle/原地转向/蹲起会错失火力机会。
                # 移动类动作不罚，允许寻找更好射界或推进。
                missed_shot_penalty -= 0.06

        good_shot_reward = min(good_shot_reward, 2.0)
        missed_shot_penalty = float(np.clip(missed_shot_penalty, -0.8, 0.0))
        reward += good_shot_reward + bad_shot_penalty + missed_shot_penalty
        info["shoot_reward"] = float(good_shot_reward + bad_shot_penalty + missed_shot_penalty)

        edge_penalty = 0.0
        H, W = self.grid_size
        for i in range(self.num_team_a):
            if not self.alive_mask[i]:
                continue

            y, x = self.agents_pos[i]
            dist_to_edge = min(y, H - 1 - y, x, W - 1 - x)
            outward_now = self._is_outward_facing_at_edge(i, margin=self.edge_turn_margin_A)
            outward_prev = bool(self.prev_outward_edge_A[i])

            if outward_now and dist_to_edge <= 1 and not self._agent_can_see_enemy(i):
                edge_penalty -= 0.04
                if outward_prev:
                    edge_penalty -= 0.02

            self.prev_outward_edge_A[i] = 1 if outward_now else 0

        edge_penalty = float(np.clip(edge_penalty, -0.8, 0.0))
        reward += edge_penalty
        info["edge_penalty"] = float(edge_penalty)
        self.ep_edge_penalty += float(edge_penalty)
        info["ep_edge_penalty"] = float(self.ep_edge_penalty)

        # 这三类局部 shaping 对主任务帮助有限，还会和射击/推进信号抢权重。
        # 先关掉，保留更直接的推进、跟进、伤害和终局反馈。
        support_reward = 0.0
        facing_reward = 0.0
        endgame_hotspot_reward = 0.0
        info["support_reward"] = 0.0
        info["facing_reward"] = 0.0
        info["endgame_hotspot_reward"] = 0.0

        teamA_visible = self._get_team_visible_enemies(TEAM_A)
        contact_reward = 0.0
        if (not self.teamA_first_contact) and len(teamA_visible) > 0:
            contact_reward += 0.8
            self.teamA_first_contact = True
            info["first_contact_bonus"] = 0.8
        if len(teamA_visible) > 0:
            contact_reward += 0.05
        if any(self._has_shootable_enemy(i) for i in range(self.num_team_a) if self.alive_mask[i]):
            contact_reward += 0.10

        contact_reward = min(contact_reward, 0.95)
        reward += contact_reward
        info["contact_reward"] = float(contact_reward)

        progress_reward = 0.0
        route_push_reward = 0.0
        support_follow_reward = 0.0
        if (
                getattr(self, "current_step", None) is not None
                and self.current_step <= self.progress_steps
                and getattr(self, "prev_squadA_progress", None) is not None
        ):
            frac = self.current_step / max(1, self.progress_steps)

            if getattr(self, "prev_teamA_progress", None) is not None:
                new_team_score = self._teamA_progress_score()
                team_delta = self.prev_teamA_progress - new_team_score
                team_weight = self.progress_coef * (1.0 - 0.35 * frac)
                progress_reward += float(team_weight * team_delta)
                self.prev_teamA_progress = new_team_score

            if getattr(self, "prev_teamA_route_progress", None) is not None:
                new_route_score = self._teamA_route_progress_score()
                route_delta = new_route_score - self.prev_teamA_route_progress
                route_weight = 2.2 * (1.0 - 0.20 * frac)
                route_push_reward = float(route_weight * route_delta)
                reward += route_push_reward
                self.prev_teamA_route_progress = new_route_score

            new_squad_scores = self._teamA_squad_progress_scores()
            squad_progress_reward = 0.0
            for old_s, new_s in zip(self.prev_squadA_progress, new_squad_scores):
                squad_progress_reward += 1.2 * (old_s - new_s)

            progress_reward += float(np.clip(squad_progress_reward, -1.0, 1.0))
            reward += progress_reward
            self.prev_squadA_progress = new_squad_scores

            support_targets_A = getattr(self, "current_support_target_A", None)
            if support_targets_A is not None:
                raw_follow_reward = 0.0
                support_followers = 0
                for i in range(self.num_team_a):
                    if not self.alive_mask[i]:
                        continue
                    tgt = support_targets_A[i]
                    if tgt is None:
                        continue
                    if self._agent_can_see_enemy(i) or self._has_shootable_enemy(i):
                        continue

                    prev_d = self.prev_support_dist[i]
                    if prev_d < 0:
                        continue

                    ty, tx = tgt
                    y, x = self.agents_pos[i]
                    new_d = abs(y - ty) + abs(x - tx)
                    delta = prev_d - new_d

                    if delta > 0:
                        raw_follow_reward += 0.34 * min(float(delta), 2.0)
                    elif (not self.moved_this_step[i]) and new_d >= prev_d:
                        raw_follow_reward -= 0.12

                    if new_d <= 8:
                        raw_follow_reward += 0.06
                    if new_d <= 4:
                        raw_follow_reward += 0.10
                    support_followers += 1

                if support_followers > 0:
                    support_follow_reward = float(np.clip(raw_follow_reward, -0.6, 1.4))
                    reward += support_follow_reward

        info["progress_reward"] = float(progress_reward)
        info["squad_progress_reward"] = float(progress_reward)
        info["route_push_reward"] = float(route_push_reward)
        info["support_follow_reward"] = float(support_follow_reward)
        info["spawn_lag_count"] = int(self._teamA_spawn_lag_count(lag_radius=5))

        hp_diff = prev_hp - self.agents_hp
        hit_bonus = 2.0
        kill_bonus = 10.0
        hurt_pen = 2.0
        dead_pen = 15.0

        for i in range(self.num_team_a, self.n_agents_total):
            if hp_diff[i] > 0:
                reward += hit_bonus * hp_diff[i]
                if prev_hp[i] > 0 and self.agents_hp[i] == 0:
                    reward += kill_bonus

        for i in range(self.num_team_a):
            if hp_diff[i] > 0:
                reward -= hurt_pen * hp_diff[i]
                if prev_hp[i] > 0 and self.agents_hp[i] == 0:
                    reward -= dead_pen

        civilian_killed = (self.map_type == "city" and any(not a for a in self.civilians_alive))
        hostage_killed = (self.map_type == "indoor" and not self.hostage_alive)

        if civilian_killed or hostage_killed:
            reward -= 120.0
            terminated = True
            info["result"] = "civilian_or_hostage_killed"
            return float(reward), terminated, info

        alive_A = np.any(self.alive_mask[: self.num_team_a])
        alive_B = np.any(self.alive_mask[self.num_team_a:])

        if self.map_type == "indoor":
            enemies_outside = []
            for j in range(self.num_team_a, self.n_agents_total):
                if not self.alive_mask[j]:
                    continue
                y, x = self.agents_pos[j]
                if not self.hostage_room_mask[y, x]:
                    enemies_outside.append(j)

            if len(enemies_outside) == 0 and alive_A and self.hostage_alive:
                survivors = int(np.sum(self.alive_mask[: self.num_team_a]))
                reward += 150.0 + 3.0 * survivors
                terminated = True
                info["result"] = "win_cleared_outside"
                return float(reward), terminated, info

        if not alive_A and alive_B:
            reward -= 150.0
            terminated = True
            info["result"] = "loss"
        elif alive_A and not alive_B:
            survivors = int(np.sum(self.alive_mask[: self.num_team_a]))
            reward += 150.0 + 3.0 * survivors
            terminated = True
            info["result"] = "win"
        elif not alive_A and not alive_B:
            reward -= 150.0
            terminated = True
            info["result"] = "draw"

        return float(reward), terminated, info

    # ======================================================================
    # 渲染
    # ======================================================================
    def render(self, mode=None):
        """
        mode:
          - None / "human": 在窗口中弹出 Matplotlib 图（交互查看）
          - "rgb_array": 返回 (H,W,3) 的 RGB 图像，用于存盘或论文图
          - "ascii": 保留原来的文本渲染
        """
        mode = mode or self.render_mode or "human"

        if mode == "ascii":
            alive_A = np.sum(self.alive_mask[: self.num_team_a])
            alive_B = np.sum(self.alive_mask[self.num_team_a:])
            print(f"Step {self.current_step} | A:{alive_A} B:{alive_B} | map={self.map_type}")

            if self.grid_h <= 25 and self.grid_w <= 25:
                char_map = {PLAIN: ".", WALL: "#", COVER: "C"}
                grid = [[char_map[self.terrain[y, x]] for x in range(self.grid_w)]
                        for y in range(self.grid_h)]

                for idx, (y, x) in enumerate(self.civilians_pos):
                    if self.civilians_alive[idx]:
                        grid[y][x] = "P"

                if self.map_type == "indoor" and self.hostage_alive and self.hostage_pos is not None:
                    hy, hx = self.hostage_pos
                    grid[hy][hx] = "H"

                for i in range(self.n_agents_total):
                    if not self.alive_mask[i]:
                        continue
                    y, x = self.agents_pos[i]
                    symbol = "A" if i < self.num_team_a else "B"
                    grid[y][x] = symbol

                for row in grid:
                    print(" ".join(row))
            print("-" * 60)
            return

        # 论文级 Matplotlib 渲染
        img = self._render_rgb()

        if mode == "human":
            plt.figure(figsize=(6, 6), dpi=150)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Step {self.current_step} | map={self.map_type}")
            plt.show()
            return

        if mode == "rgb_array":
            return img

    def _render_rgb(self):
        """
        高完成度的 2D 战术小地图渲染（尽量贴近大厂射击游戏的 minimap 风格）：
          - 电影感暗色背景 + 强暗角
          - 地形 pseudo-3D：墙高光 + 阴影，掩体有体积感
          - 雾-of-war：未知区几乎全黑，只保留一点环境光
          - 单位：双层光晕、高亮描边、血条/士气条/武器图标
          - A 队 FOV：聚光灯式渐变锥形
          - 子弹轨迹：发光激光线 + 枪口火光
          - 平民/人质：明显标记光圈
          - 顶部 HUD 状态条
        """
        H, W = self.grid_h, self.grid_w

        # 分辨率再拉高一点，整体观感更细腻
        fig, ax = plt.subplots(figsize=(W / 3.0, H / 3.0), dpi=60)
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.patch.set_facecolor("#040508")
        ax.set_facecolor("#040508")

        # =========================
        # 0) 电影感暗角背景
        # =========================
        yy, xx = np.mgrid[0:H, 0:W].astype(float)
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_max = np.max(rr) + 1e-6
        vignette = 0.25 + 0.75 * (1.0 - rr / r_max)  # 中央亮，四周暗

        bg = np.zeros((H, W, 3), dtype=np.float32)
        # 深蓝偏绿的夜战色调
        bg[..., 0] = 0.03 * vignette
        bg[..., 1] = 0.05 * vignette
        bg[..., 2] = 0.09 * vignette
        ax.imshow(bg, extent=(-0.5, W - 0.5, H - 0.5, -0.5), zorder=0)

        # =========================
        # 1) 雾-of-war + 地形 pseudo-3D
        # =========================
        base_env = get_core_env(self)
        known_map = getattr(base_env, "known_map_A", None)

        terrain_rgb = np.zeros((H, W, 3), dtype=np.float32)

        for y in range(H):
            for x in range(W):
                t = self.terrain[y, x]

                # 未探索：几乎全黑，只给一点“环境光”
                if known_map is not None and known_map[y, x] == 0:
                    terrain_rgb[y, x] = np.array([0.015, 0.018, 0.022])
                    continue

                # 已探索，根据地形上色
                if t == PLAIN:
                    base_col = np.array([0.12, 0.14, 0.18])  # 深灰偏蓝
                elif t == WALL:
                    base_col = np.array([0.28, 0.30, 0.35])  # 稍亮一点，像混凝土
                elif t == COVER:
                    base_col = np.array([0.30, 0.23, 0.13])  # 棕色掩体
                else:
                    base_col = np.array([0.10, 0.10, 0.12])

                # 简单“光源”：往右上偏亮一点
                light_dir = np.array([-0.2, -0.5])  # from top-right
                rel = np.array([y - cy, x - cx])
                lam = np.dot(rel, light_dir) / (np.linalg.norm(rel) + 1e-6)
                light_factor = 0.8 + 0.2 * (1.0 + lam) / 2.0

                # 再叠加一个 radial 光照
                radial_factor = 0.85 + 0.15 * (1.0 - rr[y, x] / r_max)

                col = base_col * light_factor * radial_factor
                terrain_rgb[y, x] = np.clip(col, 0.0, 1.0)

        ax.imshow(
            terrain_rgb,
            extent=(-0.5, W - 0.5, H - 0.5, -0.5),
            interpolation="nearest",
            zorder=1,
        )

        # 墙体“立体感”：右下方向投一层阴影 + 上边缘高光
        shadow = np.zeros((H, W, 4), dtype=np.float32)
        highlight = np.zeros((H, W, 4), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                if self.terrain[y, x] == WALL:
                    # 阴影
                    yy2, xx2 = min(H - 1, y + 1), min(W - 1, x + 1)
                    shadow[yy2, xx2, :3] = np.array([0.01, 0.01, 0.015])
                    shadow[yy2, xx2, 3] = 0.7
                    # 上边缘高光
                    hy, hx = y - 1, x
                    if 0 <= hy < H:
                        highlight[hy, x, :3] = np.array([0.45, 0.48, 0.55])
                        highlight[hy, x, 3] = 0.20

        ax.imshow(shadow, extent=(-0.5, W - 0.5, H - 0.5, -0.5), zorder=2)
        ax.imshow(highlight, extent=(-0.5, W - 0.5, H - 0.5, -0.5), zorder=2.1)

        # 掩体顶部再来一点暗色描边
        for y in range(H):
            for x in range(W):
                if self.terrain[y, x] == COVER:
                    rect = Rectangle(
                        (x - 0.5, y - 0.5),
                        1.0,
                        1.0,
                        linewidth=0.6,
                        edgecolor=(0.05, 0.03, 0.02, 0.6),
                        facecolor=(0, 0, 0, 0),
                        zorder=2.2,
                    )
                    ax.add_patch(rect)

        # 很淡的网格线
        for x in range(W + 1):
            ax.axvline(x - 0.5, color="#222222", linewidth=0.4, alpha=0.35, zorder=3)
        for y in range(H + 1):
            ax.axhline(y - 0.5, color="#222222", linewidth=0.4, alpha=0.35, zorder=3)

        # =========================
        # 2) 烟雾层
        # =========================
        # =========================
        # 3) 平民 & 人质：明显 UI 高亮
        # =========================
        for idx, (y, x) in enumerate(self.civilians_pos):
            if not self.civilians_alive[idx]:
                continue
            halo = Circle(
                (x, y),
                radius=0.55,
                facecolor=(0.0, 1.0, 0.4, 0.32),
                edgecolor="none",
                zorder=5,
            )
            ax.add_patch(halo)
            p = ax.scatter(
                x,
                y,
                marker="s",
                s=80,
                c="#69F0AE",
                edgecolors="#004D40",
                linewidths=0.7,
                zorder=6,
            )
            p.set_path_effects(
                [patheffects.Stroke(linewidth=1.0, foreground="#00130D"), patheffects.Normal()]
            )

        if self.map_type == "indoor" and self.hostage_alive and self.hostage_pos is not None:
            hy, hx = self.hostage_pos
            halo_h = Circle(
                (hx, hy),
                radius=0.65,
                facecolor=(1.0, 0.9, 0.0, 0.35),
                edgecolor="none",
                zorder=6,
            )
            ax.add_patch(halo_h)
            p = ax.scatter(
                hx,
                hy,
                marker="*",
                s=180,
                c="#FFE082",
                edgecolors="#FF6F00",
                linewidths=0.8,
                zorder=7,
            )
            p.set_path_effects(
                [patheffects.Stroke(linewidth=1.1, foreground="#2D1A00"), patheffects.Normal()]
            )

        # =========================
        # 4) A 队 FOV：聚光灯锥形
        # =========================
        fov_range = self.gun_range
        for i in range(self.num_team_a):
            if not self.alive_mask[i]:
                continue
            y, x = self.agents_pos[i]
            ori = int(self.agents_orient[i])
            dyv, dxv = self._dir2vec(ori)
            angle_center = np.degrees(np.arctan2(dyv, dxv))
            theta1 = angle_center - self.fov_half_angle_deg
            theta2 = angle_center + self.fov_half_angle_deg

            # 内圈亮、外圈暗一点
            wedge = Wedge(
                center=(x, y),
                r=fov_range,
                theta1=theta1,
                theta2=theta2,
                facecolor=(0.18, 0.65, 1.0, 0.17),
                edgecolor=(0.30, 0.80, 1.0, 0.45),
                linewidth=0.7,
                zorder=5,
            )
            ax.add_patch(wedge)

        # =========================
        # 5) 单位：光晕 + 血条 + 士气 + 武器
        # =========================
        squad_palette = [
            "#90CAF9",
            "#42A5F5",
            "#1E88E5",
            "#00ACC1",
            "#26C6DA",
            "#7E57C2",
        ]
        squad_size_for_color = 5

        morale_arr = getattr(self, "agent_morale", None)
        morale_max = getattr(self, "morale_max", 1.0)
        weapon_arr = getattr(self, "agent_weapon_type", None)
        weapon_names = {0: "SMG", 1: "SNP", 2: "SGN"}

        for i in range(self.n_agents_total):
            if not self.alive_mask[i]:
                continue

            y, x = self.agents_pos[i]
            hp = float(self.agents_hp[i])
            hp_norm = hp / max(1e-6, self.max_hp)
            posture = int(self.agents_posture[i])
            ori = int(self.agents_orient[i])

            # 姿态控制 marker 大小
            base_size = 150
            if posture == 0:
                size = base_size
            else:
                size = base_size * 0.9

            is_A = i < self.num_team_a

            if is_A:
                squad_id = i // squad_size_for_color
                color = squad_palette[squad_id % len(squad_palette)]
                marker = "o"
                edgecolor = "#E3F2FD"
            else:
                color = "#FF5252"
                marker = "^"
                edgecolor = "#FFCDD2"

            # 外层光晕 + 内层本体
            # halo_color = (0.35, 0.75, 1.0, 0.35) if is_A else (1.0, 0.25, 0.25, 0.38)
            # halo = Circle(
            #    (x, y),
            #    radius=0.60,
            #    facecolor=halo_color,
            #    edgecolor="none",
            #    zorder=8,
            # )
            # ax.add_patch(halo)

            p = ax.scatter(
                x,
                y,
                marker=marker,
                s=size,
                c=color,
                edgecolors=edgecolor,
                linewidths=0.9,
                zorder=9,
            )
            p.set_path_effects(
                [
                    patheffects.Stroke(linewidth=1.5, foreground="black"),
                    patheffects.Normal(),
                ]
            )

            # 朝向箭头（使用 FancyArrowPatch，让线更顺滑）
            dyv, dxv = self._dir2vec(ori)
            arr = FancyArrowPatch(
                (x, y),
                (x + 0.36 * dxv, y + 0.36 * dyv),
                arrowstyle="-|>",
                mutation_scale=8.5,
                linewidth=1.0,
                color="#050505",
                zorder=10,
            )
            ax.add_patch(arr)

            # 血条（头顶）
            bar_w = 0.85
            bar_h = 0.13
            y_hp = y - 0.72
            hp_bg = Rectangle(
                (x - bar_w / 2, y_hp),
                bar_w,
                bar_h,
                linewidth=0,
                edgecolor=None,
                facecolor="#101010",
                zorder=10,
            )
            ax.add_patch(hp_bg)
            hp_color = "#81C784" if is_A else "#EF5350"
            hp_fg = Rectangle(
                (x - bar_w / 2, y_hp),
                bar_w * np.clip(hp_norm, 0.0, 1.0),
                bar_h,
                linewidth=0,
                edgecolor=None,
                facecolor=hp_color,
                zorder=11,
            )
            ax.add_patch(hp_fg)

            # 士气条（脚下）——只画 A 队
            if is_A and morale_arr is not None:
                m = float(morale_arr[i])
                m_norm = m / max(1e-6, morale_max)
                y_m = y + 0.60
                morale_bg = Rectangle(
                    (x - bar_w / 2, y_m),
                    bar_w,
                    bar_h,
                    linewidth=0,
                    edgecolor=None,
                    facecolor="#101010",
                    zorder=10,
                )
                ax.add_patch(morale_bg)
                morale_fg = Rectangle(
                    (x - bar_w / 2, y_m),
                    bar_w * np.clip(m_norm, 0.0, 1.0),
                    bar_h,
                    linewidth=0,
                    edgecolor=None,
                    facecolor="#42A5F5",
                    zorder=11,
                )
                ax.add_patch(morale_fg)

            # 小队编号 + 武器类型（A 队）
            if is_A:
                squad_id = i // squad_size_for_color
                w_txt = ""
                if weapon_arr is not None:
                    w = int(weapon_arr[i])
                    w_txt = weapon_names.get(w, "")
                label = f"S{squad_id}"
                if w_txt:
                    label += f" {w_txt}"
                txt = ax.text(
                    x,
                    y - 1.05,
                    label,
                    color="#FAFAFA",
                    fontsize=7,
                    ha="center",
                    va="center",
                    zorder=12,
                )
                txt.set_path_effects(
                    [
                        patheffects.Stroke(linewidth=1.0, foreground="black"),
                        patheffects.Normal(),
                    ]
                )

        # =========================
        # 6) 射击弹道：发光“激光线”
        # =========================
        def _plot_shots(shots, color_line, color_hit):
            for shooter, ttype, tidx in shots:
                if shooter is None or shooter < 0 or shooter >= self.n_agents_total:
                    continue
                if not self.alive_mask[shooter]:
                    continue
                sy, sx = self.agents_pos[shooter]

                if ttype == "enemy" and tidx is not None and 0 <= tidx < self.n_agents_total:
                    ty, tx = self.agents_pos[tidx]
                    # 先画一条稍粗的暗线，再叠加亮线，营造发光感
                    ax.plot(
                        [sx, tx],
                        [sy, ty],
                        linestyle="-",
                        linewidth=2.0,
                        color=(0, 0, 0, 0.5),
                        zorder=6,
                    )
                    line = ax.plot(
                        [sx, tx],
                        [sy, ty],
                        linestyle="-",
                        linewidth=1.3,
                        color=color_line,
                        zorder=7,
                    )[0]
                    line.set_path_effects(
                        [
                            patheffects.Stroke(linewidth=2.4, foreground=(0, 0, 0, 0.7)),
                            patheffects.Normal(),
                        ]
                    )

                    # 枪口火光
                    muzzle = Circle(
                        (sx, sy),
                        radius=0.25,
                        facecolor="#FFEB3B",
                        edgecolor="#F57F17",
                        linewidth=0.6,
                        zorder=12,
                    )
                    ax.add_patch(muzzle)
                    # 命中火花
                    hit = Circle(
                        (tx, ty),
                        radius=0.22,
                        facecolor=color_hit,
                        edgecolor="#B71C1C",
                        linewidth=0.6,
                        zorder=12,
                    )
                    ax.add_patch(hit)

                elif ttype == "civilian" and tidx is not None and tidx < len(self.civilians_pos):
                    cy, cx = self.civilians_pos[tidx]
                    ax.plot(
                        [sx, cx],
                        [sy, cy],
                        linestyle="--",
                        linewidth=1.0,
                        color=color_line,
                        alpha=0.7,
                        zorder=6,
                    )
                elif ttype == "hostage" and self.hostage_pos is not None:
                    hy, hx = self.hostage_pos
                    ax.plot(
                        [sx, hx],
                        [sy, hy],
                        linestyle="--",
                        linewidth=1.0,
                        color=color_line,
                        alpha=0.7,
                        zorder=6,
                    )

        if hasattr(self, "last_shots_A") and self.last_shots_A:
            _plot_shots(self.last_shots_A, color_line="#BBDEFB", color_hit="#E3F2FD")
        if hasattr(self, "last_shots_B") and self.last_shots_B:
            _plot_shots(self.last_shots_B, color_line="#FF8A80", color_hit="#FFCDD2")

        # =========================
        # 7) 顶部 HUD 状态条
        # =========================
        alive_A = int(np.sum(self.alive_mask[: self.num_team_a]))
        alive_B = int(np.sum(self.alive_mask[self.num_team_a:]))

        hud_bar = Rectangle(
            (-0.5, -1.25),
            W,
            0.9,
            linewidth=0,
            edgecolor=None,
            facecolor=(0.02, 0.02, 0.03, 0.92),
            zorder=20,
        )
        ax.add_patch(hud_bar)

        hud_text = (
            f"STEP {self.current_step:03d}   "
            f"A {alive_A}/{self.num_team_a}   "
            f"B {alive_B}/{self.num_team_b}   "
            f"MAP {self.map_type.upper()}"
        )
        hud = ax.text(
            -0.2,
            -0.8,
            hud_text,
            color="#FAFAFA",
            fontsize=9,
            ha="left",
            va="center",
            zorder=21,
        )
        hud.set_path_effects(
            [
                patheffects.Stroke(linewidth=1.2, foreground="black"),
                patheffects.Normal(),
            ]
        )

        # =========================
        # 8) 输出 RGB
        # =========================
        fig.canvas.draw()
        # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = buf[:, :, :3].copy()  # 去掉 alpha 通道
        plt.close(fig)
        plt.close('all')
        return img
