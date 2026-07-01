# Quadrotor Altitude PID Simulation
import matplotlib
import sys
if "--no-show" in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Parameters ---
m = 1.0   # mass of drone in kg
g = 9.81  # gravitational acceleration (m/s^2)
dt = 0.01 # timestep in seconds

# --- Physics model ---
def update_physics(altitude, velocity, thrust):
    Fg = m * g
    Fnet = thrust - Fg
    a = Fnet / m
    velocity = velocity + a * dt
    altitude = altitude + velocity * dt
    return altitude, velocity

# --- PID controller ---
def pid_controller(target, current, integral, previous_error, kP, kI, kD):
    error = target - current
    integral = integral + error * dt
    derivative = (error - previous_error) / dt
    thrust = kP * error + kI * integral + kD * derivative
    thrust = max(0.0, thrust)
    return thrust, integral, error

# --- Run one simulation with given gains ---
def run_sim(kP, kI, kD, target=10.0, duration=10.0):
    altitude = 0.0
    velocity = 0.0
    integral = 0.0
    previous_error = target - altitude
    steps = int(15.0 / dt)
    times = []
    altitudes = []

    for i in range(steps):
        thrust, integral, previous_error = pid_controller(
            target, altitude, integral, previous_error, kP, kI, kD
        )
        altitude, velocity = update_physics(altitude, velocity, thrust)
        times.append(i * dt)
        altitudes.append(altitude)

    peak = max(altitudes)
    final = altitudes[-1]
    overshoot_pct = (peak - target) / target * 100
    return times, altitudes, peak, final, overshoot_pct


if __name__ == "__main__":
    target = 10.0

    # Three gain configurations for comparison
    configs = [
        ("P-only",  5.0, 0.0, 0.0),
        ("PD",      6.0, 0.0, 4.0),
        ("PID",     8.0, 1.5, 3.5),
    ]

    fig, axes = plt.subplots(len(configs), 1, figsize=(8, 10), sharex=True)

    for ax, (label, kP, kI, kD) in zip(axes, configs):
        times, alts, peak, final, overshoot = run_sim(kP, kI, kD)
        print(f"{label:8s}  kP={kP} kI={kI} kD={kD}  |  Peak: {peak:.2f}m  Final: {final:.2f}m  Overshoot: {overshoot:.1f}%")

        ax.plot(times, alts, label=label, linewidth=2)
        ax.axhline(y=target, color='r', linestyle='--', label="Target (10m)")
        ax.set_ylabel("Altitude (m)")
        ax.set_title(label)
        ax.legend(loc='lower right')
        ax.grid(True)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("C:/Users/brend/quadrotor-pid-sim/comparison.png", dpi=150)
    if "--no-show" not in sys.argv:
        plt.show()
    plt.close()
