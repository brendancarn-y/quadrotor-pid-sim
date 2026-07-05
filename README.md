# 1D Quadrotor Altitude Control: PID, Kalman Filter, and a Learned (PPO) Benchmark

Vertical altitude control for a quadrotor in Python. Simulating noisy data, running it through a Kalman filter to make it usable, then taking in the usable data and comparing a reinforcement-learning model (PPO) against a PID to determine effectiveness for the given scenario.

## Result: PID vs. a learned policy

![PID vs PPO benchmark: altitude response of PID and PPO at 10 m, 6 m, and 14 m setpoints](results/comparison.png)

Created a Gymnasium environment from the altitude simulation and trained a PPO policy to compete against the PID, both acting on the same Kalman-filtered state estimate. PID beat PPO on every metric at the trained 10 m setpoint, as well as on settling time and steady-state error at the unseen setpoints (6 m and 14 m).

| setpoint | controller | steady-state err (m) | overshoot | settling (s) |
|---|---|---|---|---|
| 10 m (trained) | PID | 0.03 | 0.9% | 1.84 |
| 10 m (trained) | PPO | 0.10 | 9.1% | 2.77 |
| 6 m (unseen) | PID | 0.03 | 1.3% | 2.03 |
| 6 m (unseen) | PPO | 0.10 | 1.1% | 8.54 |
| 14 m (unseen) | PID | 0.03 | 9.1% | 3.32 |
| 14 m (unseen) | PPO | 0.10 | 17.4% | 3.55 |

This is the expected outcome, given that the simulation is low-dimensional, linear, and well-modelled, and that the PID is already near-optimal. The PID requires no training and generalizes to new setpoints automatically, whereas the PPO performs best when trained on a specific task and simulated in a nonlinear or hard-to-model environment. Because of this, PID is better suited to the simulation and wins on metrics.

## Set up

* **Plant:** `envs/altitude_env.py`. Altitude simulation is wrapped behind a Gymnasium `step(action)` interface, so both controllers use the same physics. Action is normalized thrust in [-1, 1] (0 = hover), capped at a realistic thrust-to-weight of 2.
* **Shared perception:** The Kalman filter serves as a shared front-end, ensuring both systems have the same data and allowing us to observe the better strategy regardless of the data.
* **PID:** `controllers/pid.py`. Because the hover offset feeds gravity forward, no integral term is needed here (only needs PD).
* **PPO:** `train_ppo.py`. Stable-Baselines3 PPO, 300k timesteps, trained only at the 10 m setpoint.
* **Benchmark:** `evaluate.py`. Runs both through identical episodes, reports the metrics above, and runs the generalization test at unseen setpoints.

---

## Kalman filter foundation

A PID's derivative term differentiates whatever you feed it, and differentiating a noisy signal makes the noise worse. To fix this, I added a Kalman filter to sidestep it by estimating velocity as part of the state instead of differencing the position signal.

![Altitude and thrust for P-only, PD, and PID: true state vs raw measurement vs Kalman estimate](results/kalman_comparison.png)

### What happens

P-only oscillates forever. The acceleration is proportional to displacement, and there's nothing proportional to velocity to dampen energy. This leaves us with the equation for shm, which oscillates indefinitely.

PD stops oscillating but settles low, around 8.37 m. The derivative term adds the missing damping, so the oscillation stops. However, the P term only pushes in proportion to how far off target the drone is. To generate enough thrust to cancel gravity, it needs some error to push against, so it settles at a fixed distance below the target.

PID hits the target. The integral term accumulates error over time, so its output keeps growing as long as any error remains, unlike P and D, which collapse to zero when the error does. The system can only settle when the error is zero and when the integral's thrust cancels gravity.

### What the Kalman filter does

The controller only sees a noisy sensor reading, which it feeds to the derivative term, which then exaggerates the noise. Adding a filter sits between the sensor and the controller and provides a clean estimate of altitude and velocity instead.

At each step, it predicts where the drone should be using the physics model, then corrects that guess against the new reading. K, the Kalman gain, determines how much to trust the sensor relative to the model, weighted by their respective uncertainties. A noisy sensor pulls K down and leans on the model, an unreliable model pushes it up, and it rebalances every step.

As shown in the plot, the filter estimates velocity as part of the state rather than differentiating noisy signals. This means the controller can use the filter's velocity estimate directly, rather than differentiating a noisy signal, and that the estimate tracks the true state.

> **Note:** The Kalman predict step feeds it the commanded acceleration. I started with a=0, but it blew up in every configuration. P-only got pumped past 130 m, PD had a huge positive bias. This was because, with a=0, the drone can't be seen accelerating, so the estimate lags the truth, creating a phase lag, and destabilizing the feedback loop.

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python sim.py                     # Foundation figure: PID + Kalman (results/kalman_comparison.png)
python train_ppo.py               # Trains PPO, saves ppo_altitude.zip (~4 min, CPU)
python evaluate.py                # Benchmark table + generalization test (results/comparison.png)
```

## Files

* `sim.py` — the standalone PID + Kalman simulation (writes `results/kalman_comparison.png`)
* `envs/altitude_env.py` — the plant as a Gymnasium environment
* `controllers/pid.py` — the PID as a policy object
* `train_ppo.py` — trains the PPO policy
* `evaluate.py` — the PID-vs-PPO benchmark and generalization test
