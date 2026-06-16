import os

from policy import pretrain_squad_bc, set_global_seeds, train_multiagent


if __name__ == "__main__":
    SEED = 20260420
    set_global_seeds(SEED)

    MAP_TYPE = "suburb"
    MAP_size = (20, 25)
    TeamA_n = 15
    BEST_EVAL_EVERY = 50
    BEST_EVAL_EPISODES = 8
    BEST_CANDIDATE_EVAL_EPISODES = 6
    BEST_CANDIDATE_COOLDOWN = 10

    stage_team_b = [10, 15, 20, 25]
    stage_episode_add = [350,800, 500, 350]

    root_model_dir = f"./models_{MAP_TYPE}_{MAP_size[0]}x{MAP_size[1]}"
    os.makedirs(root_model_dir, exist_ok=True)
    train_log_path = "./train_log.csv"

    latest_path = None
    total_episodes = 0

    for stage_idx, (TeamB_n, add_eps) in enumerate(zip(stage_team_b, stage_episode_add), start=1):
        stage_start_episode = total_episodes
        total_episodes += add_eps
        print(f"\n===== Stage {stage_idx}: A={TeamA_n} vs B={TeamB_n}, total_episodes={total_episodes} =====")
        print(
            "[best-eval] "
            f"every={BEST_EVAL_EVERY}, "
            f"eval_eps={BEST_EVAL_EPISODES}, "
            f"candidate_eval_eps={BEST_CANDIDATE_EVAL_EPISODES}, "
            f"candidate_cooldown={BEST_CANDIDATE_COOLDOWN}"
        )

        stage_model_dir = os.path.join(root_model_dir, f"stage_{stage_idx:02d}_B{TeamB_n}")
        os.makedirs(stage_model_dir, exist_ok=True)

        squad_bc = None
        if stage_idx == 1:
            squad_bc = pretrain_squad_bc(
                map_type=MAP_TYPE,
                grid_size=MAP_size,
                num_team_a=TeamA_n,
                num_team_b=TeamB_n,
                num_episodes=20,
                max_steps_per_ep=350,
                batch_size=32,
                bc_epochs=20,
                lr=3e-4,
                label_smoothing=0.05,
                bc_entropy_coef=0.02,
                seed=SEED + stage_idx * 10000,
            )

        train_multiagent(
            num_episodes=total_episodes,   # 关键：传累计 episode
            map_type=MAP_TYPE,
            grid_size=MAP_size,
            num_team_a=TeamA_n,
            num_team_b=TeamB_n,
            max_steps_per_ep=350,   # RL 每局步数改成 350

            squad_pretrained=squad_bc if stage_idx == 1 else None,
            resume_path=latest_path,

            use_nn_for_B=False,
            record_video=True,
            video_folder=f"./videos_{MAP_TYPE}/stage_{stage_idx:02d}_B{TeamB_n}",
            video_freq=40,

            lr_squad=5e-5,
            squad_rl_warmup_episodes=20 if stage_idx == 1 else 0,
            squad_rl_start_weight=0.30,
            entropy_coef_c=0.002,
            entropy_coef_s=0.012,
            best_eval_every=BEST_EVAL_EVERY,
            best_eval_episodes=BEST_EVAL_EPISODES,
            best_candidate_eval_episodes=BEST_CANDIDATE_EVAL_EPISODES,
            best_candidate_cooldown=BEST_CANDIDATE_COOLDOWN,
            stage=stage_idx,
            stage_start_episode=stage_start_episode,
            train_log_path=train_log_path,

            model_folder=stage_model_dir,
            save_latest_every=50,
            save_last_k=1,
            seed=SEED + stage_idx * 10000,
        )

        latest_path = os.path.join(stage_model_dir, "latest.pth")
