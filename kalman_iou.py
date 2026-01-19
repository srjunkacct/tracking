"""
Kalman Filter with Integrated Ornstein-Uhlenbeck (IOU) motion model.

State vector: [x, y, vx, vy]
- Position (x, y) is the integral of velocity
- Velocity (vx, vy) follows an Ornstein-Uhlenbeck process (mean-reverting to zero)

The OU process: dv = -β*v*dt + σ*dW
where β = 1/τ (τ is the relaxation time) and σ is the diffusion coefficient.
"""
from idlelib.debugger_r import restart_subprocess_debugger

import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Callable, Any

from numpy import dtype, ndarray
from numpy._core.multiarray import _ScalarT


@dataclass
class IOUParams:
    """Parameters for the Integrated Ornstein-Uhlenbeck model."""
    tau: float  # Relaxation time (seconds) - how quickly velocity reverts to zero
    sigma: float  # Diffusion coefficient - process noise intensity

    @property
    def beta(self) -> float:
        """Mean reversion rate."""
        return 1.0 / self.tau


def iou_transition_matrix(dt: float, beta: float) -> np.ndarray:
    """
    Compute the state transition matrix F for the IOU model.

    For a single dimension, the transition is:
        x(t+dt) = x(t) + (1 - exp(-β*dt))/β * v(t)
        v(t+dt) = exp(-β*dt) * v(t)

    For 2D, we have independent x and y dimensions.
    State order: [x, y, vx, vy]
    """
    exp_neg_beta_dt = np.exp(-beta * dt)
    pos_from_vel = (1.0 - exp_neg_beta_dt) / beta

    F = np.array([
        [1.0, 0.0, pos_from_vel, 0.0],
        [0.0, 1.0, 0.0, pos_from_vel],
        [0.0, 0.0, exp_neg_beta_dt, 0.0],
        [0.0, 0.0, 0.0, exp_neg_beta_dt],
    ])
    return F


def iou_process_noise(dt: float, beta: float, sigma: float) -> np.ndarray:
    """
    Compute the process noise covariance matrix Q for the IOU model.

    Derived from the continuous-time IOU process covariances integrated over dt.
    """
    b = beta
    s2 = sigma ** 2
    exp_neg_bdt = np.exp(-b * dt)
    exp_neg_2bdt = np.exp(-2 * b * dt)

    # Variance terms (derived from IOU process)
    # Position variance
    q_xx = (s2 / (b**3)) * (dt - (2/b)*(1 - exp_neg_bdt) + (1/(2*b))*(1 - exp_neg_2bdt))

    # Velocity variance
    q_vv = (s2 / (2*b)) * (1 - exp_neg_2bdt)

    # Position-velocity covariance
    q_xv = (s2 / (2*b**2)) * (1 - 2*exp_neg_bdt + exp_neg_2bdt)

    # Build 4x4 Q matrix (x and y are independent)
    Q = np.array([
        [q_xx, 0.0,  q_xv, 0.0],
        [0.0,  q_xx, 0.0,  q_xv],
        [q_xv, 0.0,  q_vv, 0.0],
        [0.0,  q_xv, 0.0,  q_vv],
    ])
    return Q


class KalmanFilter:
    """
    Linear Kalman Filter implementation.

    Attributes:
        x: State estimate [x, y, vx, vy]
        P: State covariance matrix (4x4)
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray):
        """
        Initialize the Kalman filter.

        Args:
            x0: Initial state estimate (4,)
            P0: Initial state covariance (4, 4)
        """
        self.x = x0.copy()
        self.P = P0.copy()

    def predict(self, F: np.ndarray, Q: np.ndarray) -> None:
        """
        Prediction step: propagate state and covariance forward in time.

        Args:
            F: State transition matrix
            Q: Process noise covariance
        """
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def return_prediction(self, F: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the predicted new mean and covariance
        """
        return F @ self.x, F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """
        Update step: incorporate a measurement.

        Args:
            z: Measurement vector
            H: Measurement matrix (maps state to measurement space)
            R: Measurement noise covariance
        """
        # Innovation (measurement residual)
        y = z - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P

    def update_likelihood(self, l: Callable[[np.ndarray], float]) -> None:
        """
        Update step:  incorporate an arbitrary likelihood function.
        To do this, we decompose the current state into particles using a Cholesky factorization
        of the covariance.  We then reweight the particles according to the likelihood function,
        then compute the updated mean and covariance.
        """
        x_position = self.x[0:2]
        P_position = self.p[0:2, 0:2]
        cholesky_factorization = P_position.linalg.cholesky(P_position)
        cholesky_vectors = cholesky_factorization[:, 0], cholesky_factorization[:, 1]
        particles = [x_position,
                     x_position + cholesky_vectors[0],
                     x_position - cholesky_vectors[0],
                     x_position + cholesky_vectors[1],
                     x_position - cholesky_vectors[1]]
        particle_weights = [ l(p) for p in particles ]
        sum_weights = np.sum(particle_weights)
        normalized_weights = [ w / sum_weights for w in particle_weights ]
        updated_mean = np.sum( normalized_weights[i] * particles[i] for i in range(0,5) )
        residuals = [ updated_mean - p for p in particles ]
        residual_matrix = np.column_stack(residuals)
        weight_diagonal = np.diag(normalized_weights)
        updated_covariance = residual_matrix @ weight_diagonal @ residual_matrix.T
        self.x = updated_mean.copy()
        self.P = updated_covariance.copy()

class IOUTracker:
    """
    2D position tracker using Kalman filter with IOU motion model.
    """

    def __init__(
        self,
        iou_params: IOUParams,
        measurement_noise_std: float,
        initial_position: Optional[np.ndarray] = None,
        initial_velocity: Optional[np.ndarray] = None,
        initial_position_std: float = 10.0,
        initial_velocity_std: float = 1.0,
    ):
        """
        Initialize the tracker.

        Args:
            iou_params: IOU model parameters
            measurement_noise_std: Standard deviation of position measurements
            initial_position: Initial position [x, y], defaults to origin
            initial_velocity: Initial velocity [vx, vy], defaults to zero
            initial_position_std: Uncertainty in initial position
            initial_velocity_std: Uncertainty in initial velocity
        """
        self.iou_params = iou_params
        self.measurement_noise_std = measurement_noise_std

        # Measurement matrix: we observe position only
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise_std**2

        # Initial state
        pos = initial_position if initial_position is not None else np.zeros(2)
        vel = initial_velocity if initial_velocity is not None else np.zeros(2)
        x0 = np.array([pos[0], pos[1], vel[0], vel[1]])

        # Initial covariance
        P0 = np.diag([
            initial_position_std**2,
            initial_position_std**2,
            initial_velocity_std**2,
            initial_velocity_std**2,
        ])

        self.kf = KalmanFilter(x0, P0)

    def predict(self, dt: float) -> None:
        """Predict state forward by dt seconds."""
        F = iou_transition_matrix(dt, self.iou_params.beta)
        Q = iou_process_noise(dt, self.iou_params.beta, self.iou_params.sigma)
        self.kf.predict(F, Q)

    def return_prediction(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted state"""
        F = iou_transition_matrix(dt, self.iou_params.beta)
        Q = iou_process_noise(dt, self.iou_params.beta, self.iou_params.sigma)
        return self.kf.return_prediction(F, Q)

    def update(self, measurement: np.ndarray) -> None:
        """Update with a position measurement [x, y]."""
        self.kf.update(measurement, self.H, self.R)

    def likelihood_update(self, l: Callable[[np.ndarray], float] ) -> None:
        """Update with a likelihood function"""
        self.kf.update(self, l)

    @property
    def position(self) -> np.ndarray:
        """Current position estimate [x, y]."""
        return self.kf.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate [vx, vy]."""
        return self.kf.x[2:]

    @property
    def position_covariance(self) -> np.ndarray:
        """Position covariance (2x2)."""
        return self.kf.P[:2, :2]


def straight_line_trajectory(
        duration: float,
        dt: float,
        velocity: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a straight line constant speed trajectory in the x-direction, starting at [0,0]
    """
    n_steps = int(duration / dt) + 1
    times = np.linspace(0, duration, n_steps)
    positions = np.zeros((n_steps, 2))
    velocities = np.zeros((n_steps, 2))
    x_positions = times * velocity
    x_velocities = np.ones(n_steps) * velocity
    positions[:, 0] = x_positions
    velocities[:, 0] = x_velocities
    return times, positions, velocities


def simulate_iou_trajectory(
    duration: float,
    dt: float,
    iou_params: IOUParams,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a ground truth trajectory using the IOU process.

    Returns:
        times: Array of time points
        positions: Array of positions (N, 2)
        velocities: Array of velocities (N, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_steps = int(duration / dt) + 1
    times = np.linspace(0, duration, n_steps)

    positions = np.zeros((n_steps, 2))
    velocities = np.zeros((n_steps, 2))

    positions[0] = initial_position
    velocities[0] = initial_velocity

    beta = iou_params.beta
    sigma = iou_params.sigma

    for i in range(1, n_steps):
        # Get transition matrix and process noise
        F = iou_transition_matrix(dt, beta)
        Q = iou_process_noise(dt, beta, sigma)

        # Current state
        state = np.array([
            positions[i-1, 0], positions[i-1, 1],
            velocities[i-1, 0], velocities[i-1, 1]
        ])

        # Propagate with noise
        noise = rng.multivariate_normal(np.zeros(4), Q)
        new_state = F @ state + noise

        positions[i] = new_state[:2]
        velocities[i] = new_state[2:]

    return times, positions, velocities

def generate_single_measurement(
        ground_truth_position: np.ndarray,
        look_location: np.ndarray,
        noise_std: float,
        footprint_radius: float,
        detection_prob: float = 1.0,
        rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """
    Simulate the outcome of looking for the target at the given location.
    Returns: a measurement location if the target was detected, empty if not.
    """

    if rng is None:
        rng = np.random.default_rng()

    if rng.random() < detection_prob:
        if np.linalg.norm( ground_truth_position - look_location ) < footprint_radius:
            noise = rng.normal(0, noise_std, size=2)
            return ground_truth_position + noise

    return None

def generate_measurements(
    positions: np.ndarray,
    noise_std: float,
    detection_prob: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[int], np.ndarray]:
    """
    Generate noisy position measurements with optional missed detections.

    Returns:
        indices: List of time indices where measurements were obtained
        measurements: Array of measurements (M, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_points = len(positions)
    indices = []
    measurements = []

    for i in range(n_points):
        if rng.random() < detection_prob:
            noise = rng.normal(0, noise_std, size=2)
            measurements.append(positions[i] + noise)
            indices.append(i)

    return indices, np.array(measurements)


def run_targeted_sample():
    """Run an example simulation with the sensor aiming the track center, and plot results"""
    # Set random seed for reproducibility
    rng = np.random.default_rng(42)

    # Simulation parameters
    duration = 10 * 3600.0  # seconds
    dt = 60.0  # time step

    # IOU parameters
    iou_params = IOUParams(
        tau=47800.0,  # 47800 second relaxation time
        sigma=0.10,  # diffusion coefficient, sigma^2 *2*tau = velocity variance
    )

    # Measurement parameters
    measurement_noise_std = 5.0 * 1852.0
    footprint_radius = 5.0 * 1852.0
    detection_prob = 0.8  # 80% detection rate

    # Initial conditions
    initial_position = np.array([0.0, 0.0])
    initial_velocity = np.array([15.0, 0.0])

    times, true_positions, true_velocities = straight_line_trajectory(
        duration, dt, initial_velocity[0]
    )

    num_time_steps = math.floor(duration / dt)

    # Initialize tracker with first measurement
    tracker = IOUTracker(
        iou_params=iou_params,
        measurement_noise_std=measurement_noise_std,
        initial_position=initial_position,
        initial_position_std=measurement_noise_std,
        initial_velocity_std=15.0,
    )

    estimated_positions = [tracker.position.copy()]
    estimated_velocities = [tracker.velocity.copy()]
    position_stds = [np.sqrt(np.diag(tracker.position_covariance))]
    filter_times = [0.0]
    meas_indices = [0]
    meas_times = [0.0]
    measurements = [initial_position]

    for time_index in range(1, num_time_steps):
        # Grab the state at the current time, and advance it to the time_index time
        predicted_mean, predicted_covariance = tracker.return_prediction( dt )
        next_measurement = generate_single_measurement(true_positions[time_index],
                                                       tracker.position,
                                                       measurement_noise_std,
                                                       footprint_radius,
                                                       detection_prob,
                                                       rng)
        if next_measurement:
            tracker.update( next_measurement )
            measurements.append( next_measurement)
            meas_indices.append( time_index)
            meas_times.append( time_index * dt )
        else:
            likelihood = lambda location : detection_prob if np.linalg.norm( true_positions[time_index] - tracker.position ) < footprint_radius else 0.0
            tracker.likelihood_update(likelihood)
        estimated_positions.append(tracker.position.copy())
        estimated_velocities.append(tracker.velocity.copy())
        position_stds.append(np.sqrt(np.diag(tracker.position_covariance)))
        filter_times.append( time_index * dt )

    position_error = plot_results(estimated_positions,
                                  estimated_velocities,
                                  filter_times,
                                  meas_indices,
                                  meas_times,
                                  measurement_noise_std,
                                  measurements,
                                  position_stds,
                                  times,
                                  true_positions,
                                  true_velocities)

    print(f"Simulation complete!")
    print(f"Mean position error: {np.nanmean(position_error):.3f}")
    print(f"Final position error: {position_error[-1]:.3f}")

def run_example():
    """Run an example simulation and plot results."""
    # Set random seed for reproducibility
    rng = np.random.default_rng(42)

    # Simulation parameters
    duration = 10 * 3600.0  # seconds
    dt = 60.0  # time step

    # IOU parameters
    iou_params = IOUParams(
        tau=47800.0,  # 47800 second relaxation time
        sigma=0.10 ,  # diffusion coefficient, sigma^2 *2*tau = velocity variance
    )

    # Measurement parameters
    measurement_noise_std = 5.0 * 1852.0
    detection_prob = 0.8  # 80% detection rate

    # Initial conditions
    initial_position = np.array([0.0, 0.0])
    initial_velocity = np.array([15.0, 0.0])

    # Simulate ground truth
    #times, true_positions, true_velocities = simulate_iou_trajectory(
    #    duration, dt, iou_params, initial_position, initial_velocity, rng
    #)

    times, true_positions, true_velocities = straight_line_trajectory(
        duration, dt, initial_velocity[0]
     )

    # Generate measurements
    meas_indices, measurements = generate_measurements(
        true_positions, measurement_noise_std, detection_prob, rng
    )
    meas_times = times[meas_indices]

    # Initialize tracker with first measurement
    tracker = IOUTracker(
        iou_params=iou_params,
        measurement_noise_std=measurement_noise_std,
        initial_position=measurements[0],
        initial_position_std=measurement_noise_std,
        initial_velocity_std=15.0,
    )

    # Run the filter
    estimated_positions = [tracker.position.copy()]
    estimated_velocities = [tracker.velocity.copy()]
    position_stds = [np.sqrt(np.diag(tracker.position_covariance))]
    filter_times = [meas_times[0]]

    meas_idx = 1
    for i in range(1, len(times)):
        t = times[i]

        # Predict
        tracker.predict(dt)

        # Update if we have a measurement at this time
        if meas_idx < len(meas_indices) and meas_indices[meas_idx] == i:
            tracker.update(measurements[meas_idx])
            meas_idx += 1

        estimated_positions.append(tracker.position.copy())
        estimated_velocities.append(tracker.velocity.copy())
        position_stds.append(np.sqrt(np.diag(tracker.position_covariance)))
        filter_times.append(t)

    estimated_positions = np.array(estimated_positions)
    estimated_velocities = np.array(estimated_velocities)
    position_stds = np.array(position_stds)
    filter_times = np.array(filter_times)

    position_error = plot_results(estimated_positions, estimated_velocities, filter_times, meas_indices, meas_times,
                                  measurement_noise_std, measurements, position_stds, times, true_positions,
                                  true_velocities)

    print(f"Simulation complete!")
    print(f"Mean position error: {np.nanmean(position_error):.3f}")
    print(f"Final position error: {position_error[-1]:.3f}")


def plot_results(estimated_positions: ndarray[tuple[Any, ...], dtype[_ScalarT]],
                 estimated_velocities: ndarray[tuple[Any, ...], dtype[_ScalarT]],
                 filter_times: ndarray[tuple[Any, ...], dtype[_ScalarT]], meas_indices: list[int],
                 meas_times: ndarray[tuple[Any, ...], Any], measurement_noise_std: float,
                 measurements: ndarray[tuple[Any, ...], dtype[Any]],
                 position_stds: ndarray[tuple[Any, ...], dtype[_ScalarT]], times: ndarray[tuple[Any, ...], dtype[Any]],
                 true_positions: ndarray[tuple[Any, ...], dtype[Any]],
                 true_velocities: ndarray[tuple[Any, ...], dtype[Any]]) -> Any:
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 2D trajectory plot
    ax = axes[0, 0]
    ax.plot(true_positions[:, 0], true_positions[:, 1], 'b-', label='Ground truth', linewidth=2)
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'r--', label='Estimate', linewidth=1.5)
    ax.scatter(measurements[:, 0], measurements[:, 1], c='g', s=20, alpha=0.5, label='Measurements')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('2D Trajectory')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # X position over time
    ax = axes[0, 1]
    ax.plot(times, true_positions[:, 0], 'b-', label='True', linewidth=2)
    ax.plot(filter_times, estimated_positions[:, 0], 'r--', label='Estimate', linewidth=1.5)
    ax.fill_between(
        filter_times,
        estimated_positions[:, 0] - 2 * position_stds[:, 0],
        estimated_positions[:, 0] + 2 * position_stds[:, 0],
        color='r', alpha=0.2, label='±2σ'
    )
    ax.scatter(meas_times, measurements[:, 0], c='g', s=20, alpha=0.5, label='Measurements')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X position')
    ax.set_title('X Position vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Velocity estimates
    ax = axes[1, 0]
    ax.plot(times, true_velocities[:, 0], 'b-', label='True Vx', linewidth=2)
    ax.plot(times, true_velocities[:, 1], 'b--', label='True Vy', linewidth=2)
    ax.plot(filter_times, estimated_velocities[:, 0], 'r-', label='Est Vx', linewidth=1.5)
    ax.plot(filter_times, estimated_velocities[:, 1], 'r--', label='Est Vy', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity')
    ax.set_title('Velocity Estimates')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Position error
    ax = axes[1, 1]
    # Align true positions with filter times
    true_pos_aligned = true_positions[meas_indices[0]:][:len(filter_times)]
    if len(true_pos_aligned) < len(estimated_positions):
        true_pos_aligned = np.vstack([
            true_pos_aligned,
            np.full((len(estimated_positions) - len(true_pos_aligned), 2), np.nan)
        ])
    position_error = np.linalg.norm(estimated_positions - true_pos_aligned, axis=1)
    ax.plot(filter_times, position_error, 'k-', label='Position error', linewidth=1.5)
    ax.axhline(measurement_noise_std, color='g', linestyle='--', label='Measurement noise σ')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position error')
    ax.set_title('Position Estimation Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kalman_iou_results.png', dpi=150)
    plt.show()
    return position_error


if __name__ == "__main__":
    run_example()
