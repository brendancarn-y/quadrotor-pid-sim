# 1D Quadrotor Altitude Control: PID, Kalman Filter, and a Learned (PPO) Benchmark

Vertical altitude control for a quadrotor, in Python: a hand-tuned classical
controller, a Kalman filter that makes it noise-robust, and a reinforcement-learning
policy benchmarked against it on identical dynamics.

## Result: PID vs. a learned policy

![PID vs PPO](results/comparison.png)

I wrapped the altitude simulation as a Gymnasium environment and trained a PPO
policy to compete against the hand-tuned PID, both acting on the same
Kalman-filtered state estimate. **PID beat PPO on every metric at the trained
10 m setpoint** — steady-state error 0.03 m vs 0.10 m, overshoot 0.9% vs 9.1%,
settling time 1.8 s vs 2.8 s — and dominated on settling time and steady-state
error at unseen setpoints (6 m and 14 m) as well.

| setpoint | controller | steady-state err (m) | overshoot | settling (s) |
|---|---|---|---|---|
| 10 m *(trained)* | PID | **0.03** | **0.9%** | **1.84** |
| 10 m *(trained)* | PPO | 0.10 | 9.1% | 2.77 |
| 6 m *(unseen)* | PID | **0.03** | 1.3% | **2.03** |
| 6 m *(unseen)* | PPO | 0.10 | 1.1% | 8.54 |
| 14 m *(unseen)* | PID | **0.03** | **9.1%** | **3.32** |
| 14 m *(unseen)* | PPO | 0.10 | 17.4% | 3.55 |

That is the expected outcome, and reading it correctly is the point. For a
low-dimensional, linear, well-modeled plant, PID is already near-optimal: it
needs no training and generalizes to new setpoints for free, while a policy
trained only at 10 m carries a fixed steady-state offset and slower convergence
into setpoints it never saw. PPO learned a competent controller from reward
alone, but there was no nonlinearity or coupling for it to exploit. Learned
control earns its keep on nonlinear, high-DOF, or hard-to-model systems — this
task is deliberately the opposite, the case where classical control wins.

### How it's set up

- **Plant** — `envs/altitude_env.py`. The altitude dynamics wrapped behind a
  Gymnasium `step(action)` interface, so both controllers drive identical
  physics. Action is normalized thrust in `[-1, 1]` (0 = hover), capped at a
  realistic thrust-to-weight of 2.
- **Shared perception** — a noisy altitude sensor is cleaned by the same 2-state
  Kalman filter described below, and both controllers observe the filtered
  `[error, velocity]` estimate. The filter is a shared front-end, not part of
  the contest.
- **PID** — `controllers/pid.py`. The classical law as a policy object. Because
  the hover offset feeds gravity forward, no integral term is needed here (PD
  suffices); adding one only winds up during the climb.
- **PPO** — `train_ppo.py`. Stable-Baselines3 PPO, 300k timesteps, trained only
  at the 10 m setpoint.
- **Benchmark** — `evaluate.py`. Runs both through identical episodes, reports
  the metrics above, and runs the generalization test at unseen setpoints.

### Reproduce

    python -m venv .venv
    .venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
    pip install "stable-baselines3[extra]" gymnasium numpy matplotlib
    python train_ppo.py               # trains PPO, saves ppo_altitude.zip (~4 min, CPU)
    python evaluate.py                # writes results/comparison.png and the metrics table

---

# Foundation: PID + Kalman filter

![PID / Kalman comparison](comparison.png)

A PID's derivative term differentiates whatever you feed it, and differentiating a noisy signal makes the noise worse. To fix this, I added a Kalman filter to sidestep it by estimating velocity as part of the state instead of differencing the position signal.

## What happens

P-only oscillates forever. The acceleration is proportional to displacement, and there's nothing proportional to velocity to dampen energy. This leaves us with the equation for shm, which oscillates indefinitely.

PD stops oscillating but settles low, around 8.37 m. The derivative term adds the missing damping, so the oscillation stops. However, the P term only pushes in proportion to how far off target the drone is. To generate enough thrust to cancel gravity, it needs some error to push against, so it settles at a fixed distance below the target.

PID hits the target. The integral term accumulates error over time, so its output keeps growing as long as any error remains, unlike P and D, which collapse to zero when the error does. The system can only settle when the error is zero and when the integral's thrust cancels gravity.

## What the Kalman filter does

The controller only sees a noisy sensor reading, which it feeds to the derivative term, which then exaggerates the noise. Adding a filter sits between the sensor and the controller and provides a clean estimate of altitude and velocity instead.

At each step, it predicts where the drone should be using the physics model, then corrects that guess against the new reading. K, the Kalman gain, sets how far to trust the sensor over the model, weighted by their relative uncertainty. A noisy sensor pulls K down and leans on the model, an unreliable model pushes it up, and it rebalances every step.

As shown in the plot, the filter estimates velocity as part of the state rather than differentiating noisy signals. This means the controller can use the filter's velocity estimate directly, rather than differentiating a noisy signal, and that the estimate tracks the true state.

**Note:** The Kalman predict step feeds it the commanded acceleration. I started with a=0, but it blew up in every configuration. P-only got pumped past 130 m, PD had a huge positive bias. This was because, with a=0, the drone can't be seen accelerating, so the estimate lags the truth, creating a phase lag, and destabilizing the feedback loop.

## Run it

    pip install numpy matplotlib
    python sim.py

This writes `comparison.png`, the figure shown above.

## Files

- `sim.py` — standalone PID + Kalman altitude simulation (the foundation figure)
- `envs/altitude_env.py` — dynamics wrapped as a Gymnasium environment
- `controllers/pid.py` — PID control law as a policy object
- `train_ppo.py` — trains and saves the PPO policy
- `evaluate.py` — benchmarks PID vs PPO, writes `results/comparison.png`
