# Quadrotor Altitude PID Simulation
import matplotlib
import os
import sys
if "--no-show" in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Parameters ---
m = 1.0   # mass of drone in kg
g = 9.81  # gravitational acceleration (m/s^2)
dt = 0.01 # timestep in seconds

# --- Sensor / filter noise parameters ---
sigma_z = 0.25  # altitude measurement noise std (m)
sigma_a = 1.0   # process (acceleration) noise std (m/s^2)

# --- Kalman model matrices (state x = [z, v]^T) ---
F = np.array([[1.0, dt],
              [0.0, 1.0]])
G = np.array([[0.5 * dt**2],
              [dt]])
H = np.array([[1.0, 0.0]])
Q = sigma_a**2 * (G @ G.T)
R = np.array([[sigma_z**2]])

# --- Measurement layer ---
def measure(z_true, sigma_z, rng):
    return z_true + rng.normal(0.0, sigma_z)

# --- Kalman filter ---
def kf_predict(x, P, a, F, G, Q):
    return F @ x + G * a, F @ P @ F.T + Q

def kf_update(x_pred, P_pred, y, H, R):
    ytilde = y - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    return x_pred + K @ ytilde, (np.eye(2) - K @ H) @ P_pred

# --- Physics model ---
def update_physics(altitude, velocity, thrust):
    Fg = m * g
    Fnet = thrust - Fg
    a = Fnet / m
    velocity = velocity + a * dt
    altitude = altitude + velocity * dt
    return altitude, velocity

# --- PID controller (derivative supplied externally as edot) ---
def pid_controller(target, current, edot, integral, kP, kI, kD):
    error = target - current
    integral = integral + error * dt
    derivative = edot
    thrust = kP * error + kI * integral + kD * derivative
    thrust = max(0.0, thrust)
    return thrust, integral, error

# --- Run one simulation with given gains ---
# mode selects what the controller acts on:
#   "true"     -> true simulated state (no noise, no filter)
#   "measured" -> raw noisy measurement (derivative from noisy difference)
#   "filtered" -> Kalman estimate of altitude and velocity
def run_sim(kP, kI, kD, mode="true", target=10.0, seed=0):
    altitude = 0.0
    velocity = 0.0
    integral = 0.0
    steps = int(15.0 / dt)
    rng = np.random.default_rng(seed)

    x_hat = np.array([[0.0], [0.0]])  # initial estimate
    P = np.eye(2)                     # initial covariance
    y_prev = 0.0
    a_prev = 0.0                      # last commanded acceleration fed to predict

    times = []
    altitudes = []
    thrusts = []

    for i in range(steps):
        y = measure(altitude, sigma_z, rng)

        # Kalman filter: predict then update.
        # NOTE: the spec's base build predicts with a=0 (constant-velocity).
        # This plant accelerates hard (~40-50 m/s^2), so a=0 makes the estimate
        # lag badly and destabilizes the loop (see checkpoint runs). We feed the
        # previous step's commanded acceleration instead -- the spec's own
        # "tighter build" -- which tracks truth and keeps noise out.
        x_hat, P = kf_predict(x_hat, P, a_prev, F, G, Q)
        x_hat, P = kf_update(x_hat, P, y, H, R)

        # Choose the signal the controller acts on
        if mode == "true":
            pos, vel = altitude, velocity
        elif mode == "measured":
            pos, vel = y, (y - y_prev) / dt
        elif mode == "filtered":
            pos, vel = x_hat[0, 0], x_hat[1, 0]
        else:
            raise ValueError(f"unknown mode: {mode}")

        thrust, integral, _ = pid_controller(target, pos, -vel, integral, kP, kI, kD)
        altitude, velocity = update_physics(altitude, velocity, thrust)

        a_prev = (thrust - m * g) / m
        y_prev = y
        times.append(i * dt)
        altitudes.append(altitude)
        thrusts.append(thrust)

    peak = max(altitudes)
    final = altitudes[-1]
    overshoot_pct = (peak - target) / target * 100
    return times, altitudes, thrusts, peak, final, overshoot_pct


if __name__ == "__main__":
    target = 10.0

    # Three gain configurations for comparison
    configs = [
        ("P-only",  5.0, 0.0, 0.0),
        ("PD",      6.0, 0.0, 4.0),
        ("PID",     8.0, 1.5, 3.5),
    ]

    # Control on true state, raw measurement, or Kalman estimate
    modes = [
        ("true",     "true state",  "tab:green"),
        ("measured", "raw measured", "tab:red"),
        ("filtered", "KF estimate", "tab:blue"),
    ]

    # rows = configs, columns = [altitude, thrust]; each panel overlays the 3 modes
    fig, axes = plt.subplots(len(configs), 2, figsize=(12, 11), sharex=True)

    for row, (label, kP, kI, kD) in enumerate(configs):
        ax_alt, ax_thr = axes[row]
        # collect true+filtered ranges to frame the axes (measured may diverge/clip)
        alt_ref, thr_ref = [], []

        for mode, mode_label, color in modes:
            times, alts, thrusts, peak, final, overshoot = run_sim(kP, kI, kD, mode=mode)
            print(f"{label:8s} {mode:8s}  Peak: {peak:7.2f}m  Final: {final:7.2f}m  Overshoot: {overshoot:7.1f}%")
            ax_alt.plot(times, alts, label=mode_label, color=color, linewidth=1.5)
            ax_thr.plot(times, thrusts, label=mode_label, color=color, linewidth=1.0)
            if mode != "measured":
                alt_ref += alts
                thr_ref += thrusts
        print()

        ax_alt.axhline(y=target, color='k', linestyle='--', linewidth=1, label="Target")
        # frame to true+filtered so those stay readable; diverging measured clips
        amin, amax = min(alt_ref), max(alt_ref)
        ax_alt.set_ylim(min(amin, 0) - 1, max(amax, target) * 1.15)
        tmin, tmax = min(thr_ref), max(thr_ref)
        pad = 0.1 * (tmax - tmin + 1)
        ax_thr.set_ylim(tmin - pad, tmax + pad)

        ax_alt.set_ylabel("Altitude (m)")
        ax_alt.set_title(f"{label}: altitude")
        ax_thr.set_title(f"{label}: thrust")
        ax_thr.set_ylabel("Thrust (N)")
        for ax in (ax_alt, ax_thr):
            ax.legend(loc="best", fontsize=8)
            ax.grid(True)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("PID on true state vs. raw measurement vs. Kalman estimate", fontsize=13)
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "kalman_comparison.png"), dpi=150)
    if "--no-show" not in sys.argv:
        plt.show()
    plt.close()
