import numpy as np

class UnscentedKalmanFilter:
    def __init__(self, state_dim, measurement_dim, process_noise_cov, measurement_noise_cov):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.alpha = 0.1  # Scaling parameter for selecting sigma points
        self.kappa = 0.0  # Secondary scaling parameter
        self.beta = 2.0   # Weighting parameter for covariance estimation
        self.num_sigma_points = 2 * state_dim + 1
        self.sigma_points = np.zeros((self.num_sigma_points, state_dim))
        self.weights_mean = np.zeros(self.num_sigma_points)
        self.weights_cov = np.zeros(self.num_sigma_points)
        self.state_mean = np.zeros(state_dim)
        self.state_cov = np.eye(state_dim)
        self.measurement_mean = np.zeros(measurement_dim)
        self.measurement_cov = np.eye(measurement_dim)
    
    def predict(self, delta_time):
        self._generate_sigma_points()
        self._propagate_sigma_points(delta_time)
        self._compute_predicted_state()
        self._compute_predicted_measurement()
        self._compute_predicted_covariance()
    
    def update(self, measurement):
        self._compute_cross_covariance()
        self._compute_kalman_gain()
        self._update_state(measurement)
        self._update_covariance()
    
    def _generate_sigma_points(self):
        sqrt_cov = np.linalg.cholesky(self.state_cov)
        self.sigma_points[0] = self.state_mean
        self.weights_mean[0] = self.alpha/(self.state_dim+self.alpha)
        scaled_sqrt_cov = np.sqrt(self.state_dim + self.alpha) * sqrt_cov
        for i in range(self.state_dim):
            self.sigma_points[i + 1] = self.state_mean + scaled_sqrt_cov[i]
            self.sigma_points[i + 1 + self.state_dim] = self.state_mean - scaled_sqrt_cov[i]
            self.weights_mean[i + 1] = 1 / (2*(self.state_dim+self.alpha))
            self.weights_mean[i + 1 + self.state_dim] = 1 / (2*(self.state_dim+self.alpha))
    
    def _propagate_sigma_points(self, delta_time):
        # Implement the process model to propagate sigma points
        # Update self.sigma_points with the propagated sigma points
        self.sigma_points = np.zeros((self.num_sigma_points, self.state_dim)) # Placeholder
    
    def _compute_predicted_state(self):
        self.state_mean = np.average(self.sigma_points, axis=0, weights=self.weights_mean)
    
    def _compute_predicted_measurement(self):
        # Implement the measurement model to compute predicted measurements
        # Update self.measurement_mean with the predicted measurement mean
        self.measurement_mean = np.zeros(self.measurement_dim) # Placeholder
        
    
    def _compute_predicted_covariance(self):
        centered_points = self.sigma_points - self.state_mean
        self.state_cov = (self.weights_cov[:, np.newaxis] * centered_points).T @ centered_points
        self.state_cov += self.process_noise_cov
    
    def _compute_cross_covariance(self):
        centered_states = self.sigma_points - self.state_mean
        centered_measurements = self.sigma_points - self.measurement_mean
        self.cross_cov = (self.weights_cov[:, np.newaxis] * centered_states).T @ centered_measurements
    
    def _compute_kalman_gain(self):
        innovation_cov = self.measurement_cov + self.measurement_noise_cov
        self.kalman_gain = self.cross_cov @ np.linalg.inv(innovation_cov)
    
    def _update_state(self, measurement):
        innovation = measurement - self.measurement_mean
        self.state_mean += self.kalman_gain @ innovation
    
    def _update_covariance(self):
        self.state_cov -= self.kalman_gain @ self.measurement_cov @ self.kalman_gain.T
    
    def set_initial_state(self, initial_state):
        self.state_mean = initial_state
        self.state_cov = np.eye(self.state_dim)
    
    def set_initial_measurement(self, initial_measurement):
        self.measurement_mean = initial_measurement
        self.measurement_cov = np.eye(self.measurement_dim)
    
    def set_process_noise_cov(self, process_noise_cov):
        self.process_noise_cov = process_noise_cov
    
    def set_measurement_noise_cov(self, measurement_noise_cov):
        self.measurement_noise_cov = measurement_noise_cov
    
    def set_parameters(self, alpha, kappa, beta):
        self.alpha = alpha
        self.kappa = kappa
        self.beta = beta
    
    def get_state(self):
        return self.state_mean
    
    def get_state_covariance(self):
        return self.state_cov
    
    def get_measurement(self):
        return self.measurement_mean
    
    def get_measurement_covariance(self):
        return self.measurement_cov

    # Consider implementing for the case where bouyes come and go
    def update_measurement_dim(self,measurement_dim):
        pass