"""Train a PPO policy on the altitude-control environment and save it.

The environment is noiseless-to-the-controller: it observes the Kalman-filtered
estimate, identical to what the PID controller receives. Run from the repo root:

    .venv/Scripts/python train_ppo.py
"""
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs.altitude_env import AltitudeEnv

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    env = AltitudeEnv()          # trains at the default z_target = 10 m
    check_env(env)               # validates the Gymnasium contract
    model = PPO("MlpPolicy", env, verbose=1, seed=0)
    model.learn(total_timesteps=300_000)
    save_path = os.path.join(HERE, "ppo_altitude")
    model.save(save_path)
    print(f"saved {save_path}.zip")


if __name__ == "__main__":
    main()
