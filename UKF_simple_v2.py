
import numpy as np
import math


class UnscentedKalmanFilter:
    def __init__(self, state_dim, measurement_dim, process_noise_cov, measurement_noise_cov):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.total_dim = state_dim + measurement_dim
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.alpha = np.sqrt(3/self.total_dim)  # Scaling parameter for selecting sigma points
        self.lamb = self.alpha**2 * (self.total_dim) - self.total_dim
        self.kappa = 0.0  # Secondary scaling parameter
        self.beta = 3 / self.total_dim - 1   # Weighting parameter for covariance estimation
        self.num_sigma_points = 2 * self.total_dim + 1
        self.sigma_points_state = np.zeros((state_dim,self.num_sigma_points))
        self.sigma_points_measurements = np.zeros((measurement_dim,self.num_sigma_points))
        self.sigma_points_total = np.zeros((state_dim+measurement_dim,self.num_sigma_points))
        self.weights_measurement_update = np.zeros(self.num_sigma_points)
        self.weights_cov = np.zeros(self.num_sigma_points)
        self.state_estimate = np.zeros(state_dim)
        self.state_cov = np.identity(2)*3 # The 3 is arbitrary
        self.measurement_estimate = np.zeros(measurement_dim)
        self.measurement_cov = measurement_noise_cov
        self.cross_cov_meas = np.ones([2,4])
        self.measurement_transform_cov = np.array([[self.state_cov,self.cross_cov_meas],
                                                   [self.cross_cov_meas.T,self.measurement_cov]])

    def measurement_update(self,measurement):
        self._generate_sigma_points_measurement()
        self._calculate_sigma_covariance()
        self._update_kalman_gain()
        self._update_state_estimate_meas(measurement)
        self._update_state_cov()
        
    '''
    Generates sigma points from (8.15b) and (A.19a/b)
    '''
    def _generate_sigma_points_measurement(self):
        U,S,Vt = np.linalg.svd(self.measurement_transform_cov)
        self.sigma_points_total[:,0] = np.array([self.state_estimate,self.state_estimate,self.state_estimate])
        self.weights_measurement_update[0] = self.lamb/(self.state_dim+self.lamb)
        for i in range(self.total_dim):
            self.sigma_points_state[:,2*i] = self.sigma_points_state[:,0] + np.sqrt(self.state_dim+self.lamb)*S[i,i]*U[:,i]
            self.sigma_points_state[:,2*i+1] = self.sigma_points_state[:,0] - np.sqrt(self.state_dim+self.lamb)
            self.weights_measurement_update[2*i] = 1/(2*(self.state_dim+self.lamb))
            self.weights_measurement_update[2*i+1] = 1/(2*(self.state_dim+self.lamb))

    def _calculate_sigma_covariance(self):
        mu_z = self.weights_measurement_update[0] * self.sigma_points_total[:,0]
        for i in range(self.total_dim):
            mu_z += self.weights_measurement_update[2*i] * self.sigma_points_state[:,2*i]
            mu_z += self.weights_measurement_update[2*i+1] * self.sigma_points_state[:,2*i+1]
        self.measurement_transform_cov = self.weights_measurement_update[0] * (self.sigma_points_total[:,0]-mu_z)@(self.sigma_points_total[:,0]-mu_z).T
        for i in range(self.total_dim):
            self.measurement_transform_cov += self.weights_measurement_update[2*i] * (self.sigma_points_total[:2*i]-mu_z)@(self.sigma_points_total[:,2*i]-mu_z).T
            self.measurement_transform_cov += self.weights_measurement_update[2*i+1] * (self.sigma_points_total[:2*i+1]-mu_z)@(self.sigma_points_total[:,2*i+1]-mu_z).T
        self.measurement_transform_cov += (1-self.alpha**2+self.beta**2)*(self.sigma_points_total[:,0]-mu_z)@(self.sigma_points_total[:,0]-mu_z).T
        self.state_cov = self.measurement_transform_cov[0:2,0:2]
        self.cross_cov_meas = self.measurement_transform_cov[0:2,2:]
        self.measurement_cov = self.measurement_transform_cov[2:,2:]

    def _update_kalman_gain(self):
        self.kalman_gain = self.cross_cov_meas@np.linalg.inv(self.measurement_cov)

    def _update_state_estimate_meas(self,measurement):
        self.state_estimate += self.kalman_gain@(measurement-np.array([self.state_estimate,self.state_estimate]))

    def _update_state_cov(self):
        self.state_cov -= self.kalman_gain@self.measurement_cov@self.kalman_gain.T

    def time_update(self,time_diff, speed, cog):
        pass

    def _generate_sigma_points_time(self,time_diff,speed,cog):
        pass

    def _calculate_state_cov_time(self):
        pass

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
    
    def set_parameters(self, lamb,alpha, kappa, beta):
        self.lamb = lamb
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