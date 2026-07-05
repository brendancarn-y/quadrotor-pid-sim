"""Train a *robustness* PPO policy with domain-randomized disturbances.

Same as train_ppo.py, but each episode injects a random constant disturbance
acceleration (+/- DIST_RANGE m/s^2) that the controller is blind to. The policy
must learn to cope with an unknown push it cannot measure -- domain
randomization. Saved separately so the clean benchmark model is untouched.

    .venv/Scripts/python train_ppo_robust.py
"""
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs.altitude_env import AltitudeEnv

HERE = os.path.dirname(os.path.abspath(__file__))
DIST_RANGE = 4.0   # each episode draws b_true ~ U(-4, 4) m/s^2


def main():
    env = AltitudeEnv(dist_range=DIST_RANGE)
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=1, seed=0)
    model.learn(total_timesteps=300_000)
    save_path = os.path.join(HERE, "ppo_altitude_robust")
    model.save(save_path)
    print(f"saved {save_path}.zip")


if __name__ == "__main__":
    main()
