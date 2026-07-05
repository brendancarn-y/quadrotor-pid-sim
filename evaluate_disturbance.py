"""Robustness benchmark: PID vs clean PPO vs domain-randomized PPO under an
unmodeled constant disturbance the controllers cannot measure.

The clean benchmark (evaluate.py) runs on a disturbance-free plant and PID wins.
Here every episode injects an unknown constant acceleration b_true. We compare:
  - PID            : the hand-tuned PD controller (no disturbance rejection)
  - PPO (clean)    : trained without disturbances (evaluate.py's model)
  - PPO (robust)   : trained with domain-randomized disturbances

    .venv/Scripts/python evaluate_disturbance.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("Loading libraries (first PyTorch import takes a few seconds)...", flush=True)
from stable_baselines3 import PPO

from envs.altitude_env import AltitudeEnv
from controllers.pid import PIDPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
PID_GAINS = dict(kp=1.5, ki=0.0, kd=0.8)
Z_TARGET = 10.0
DIST_VALUES = [-4, -3, -2, -1, 0, 1, 2, 3, 4]   # constant disturbances to test
N_SEEDS = 8
COLORS = {"PID": "tab:blue", "PPO (clean)": "tab:orange", "PPO (robust)": "tab:green"}


def run_episode(controller, b_true, is_rl=False, seed=0):
    env = AltitudeEnv(z_target=Z_TARGET)
    obs, _ = env.reset(seed=seed, options={"b_true": b_true})
    if hasattr(controller, "reset"):
        controller.reset()
    ts, zs, t = [], [], 0.0
    done = False
    while not done:
        action = controller.predict(obs, deterministic=True)[0] if is_rl else controller.act(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        zs.append(env.z); ts.append(t); t += env.dt
        done = terminated or truncated
    return np.array(ts), np.array(zs)


def ss_error(zs):
    return abs(Z_TARGET - float(np.mean(zs[-100:])))


def mean_ss_error(controller, b, is_rl):
    return float(np.mean([ss_error(run_episode(controller, b, is_rl, seed=s)[1])
                          for s in range(N_SEEDS)]))


def main():
    print("Evaluating robustness to unmodeled disturbances (~30 s)...\n", flush=True)
    pid = PIDPolicy(dt=0.01, **PID_GAINS)
    ppo_clean = PPO.load(os.path.join(HERE, "ppo_altitude"))
    ppo_robust = PPO.load(os.path.join(HERE, "ppo_altitude_robust"))
    controllers = [("PID", pid, False),
                   ("PPO (clean)", ppo_clean, True),
                   ("PPO (robust)", ppo_robust, True)]

    # sweep: mean |steady-state error| vs disturbance magnitude
    curve = {name: [] for name, _, _ in controllers}
    print(f"{'disturbance':>11} | " + " ".join(f"{n:>13}" for n, _, _ in controllers))
    for b in DIST_VALUES:
        errs = [mean_ss_error(c, b, rl) for _, c, rl in controllers]
        for (name, _, _), e in zip(controllers, errs):
            curve[name].append(e)
        print(f"{b:>11.1f} | " + " ".join(f"{e:>13.3f}" for e in errs))

    print("\nMean |steady-state error| over the disturbance range (lower = more robust):")
    for name, _, _ in controllers:
        print(f"  {name:>13}: {np.mean(curve[name]):.3f} m   (worst {np.max(curve[name]):.3f} m)")

    # figure: (left) error vs disturbance, (right) trajectories at a strong disturbance
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))
    for name, _, _ in controllers:
        axL.plot(DIST_VALUES, curve[name], "o-", label=name, color=COLORS[name])
    axL.set_title("Steady-state error vs. unmodeled disturbance")
    axL.set_xlabel("disturbance acceleration b (m/s²)")
    axL.set_ylabel("|steady-state error| (m)")
    axL.legend(); axL.grid(True)

    b_show = 4.0
    for name, c, rl in controllers:
        t, z = run_episode(c, b_show, rl, seed=0)
        axR.plot(t, z, label=name, color=COLORS[name], linewidth=1.8)
    axR.axhline(Z_TARGET, color="k", ls=":", lw=1, label="setpoint (10 m)")
    axR.set_title(f"Altitude under a strong disturbance (b = +{b_show:.0f} m/s²)")
    axR.set_xlabel("time (s)"); axR.set_ylabel("altitude (m)")
    axR.legend(); axR.grid(True)

    fig.suptitle("Robustness to an unmodeled disturbance: PID vs clean PPO vs domain-randomized PPO", fontsize=13)
    plt.tight_layout()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "disturbance_robustness.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
