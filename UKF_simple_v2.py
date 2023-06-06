
import numpy as np
import math


class UnscentedKalmanFilter:
    def __init__(self, state_dim, measurement_dim, process_noise_cov, measurement_noise_cov):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.lamb = 3 - state_dim
        self.alpha = np.sqrt(3/measurement_dim)  # Scaling parameter for selecting sigma points
        self.kappa = 0.0  # Secondary scaling parameter
        self.beta = 3 / measurement_dim - 1   # Weighting parameter for covariance estimation
        self.num_sigma_points = 2 * state_dim + 1
        self.sigma_points = np.zeros((self.num_sigma_points, state_dim))
        self.sigma_points_measurements = np.zeros((self.num_sigma_points,measurement_dim))
        self.weights_mean = np.zeros(self.num_sigma_points)
        self.weights_cov = np.zeros(self.num_sigma_points)
        self.state_mean = np.zeros(state_dim)
        self.state_cov = np.identity(2)*3 # The 3 is arbitrary
        self.measurement_mean = np.zeros(measurement_dim)
        self.measurement_cov = measurement_noise_cov

    def measurement_update(self,measurement):
        pass

    def _generate_sigma_points(self):
        pass

    def _propagate_sigma_points(self):
        pass

    def _calculate_sigma_covariance(self):
        pass

    def _update_kalman_gain(self):
        pass

    def _update_state_estimate_meas(self,measurement):
        pass

    def _update_state_cov(self):
        pass

    def time_update(self,time_diff, speed, heading):
        pass