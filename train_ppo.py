"""Train a PPO policy on the altitude-control environment and save it.

The environment is noiseless-to-the-controller: it observes the Kalman-filtered
estimate, identical to what the PID controller receives. Run from the repo root:

    .venv/Scripts/python train_ppo.py
"""
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs.altitude_env import AltitudeEnv


def main():
    env = AltitudeEnv()          # trains at the default z_target = 10 m
    check_env(env)               # validates the Gymnasium contract
    model = PPO("MlpPolicy", env, verbose=1, seed=0)
    model.learn(total_timesteps=300_000)
    model.save("ppo_altitude")
    print("saved ppo_altitude.zip")


if __name__ == "__main__":
    main()
