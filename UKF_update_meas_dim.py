import numpy as np
import math


class UnscentedKalmanFilter:
    def __init__(self, state_dim, measurement_dim, process_noise_cov, measurement_noise_cov):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.total_dim = state_dim + measurement_dim
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.alpha_meas = np.sqrt(3/self.total_dim)  # Scaling parameter for selecting sigma points
        self.alpha_time = np.sqrt(3/self.state_dim)
        self.lamb_meas = 3 - self.total_dim
        #self.lamb_meas = 0.3
        self.lamb_time = 3 - self.state_dim
        self.kappa = 0.0  # Secondary scaling parameter
        self.beta_meas = 3 / self.total_dim - 1   # Weighting parameter for covariance estimation
        #self.beta_meas = 2
        self.beta_time = 3 / self.state_dim - 1
        self.num_sigma_points_measurement = 2 * self.total_dim + 1
        self.num_sigma_points_time = 2 * self.state_dim + 1
        self.sigma_points_time = np.zeros((state_dim,self.num_sigma_points_time))
        self.sigma_points_measurement = np.zeros((self.total_dim,self.num_sigma_points_measurement))
        self.weights_measurement_update = np.zeros(self.num_sigma_points_measurement)
        self.weights_time_update = np.zeros(self.num_sigma_points_time)
        self.state_estimate = np.zeros(state_dim)
        #self.state_cov = np.identity(2) # The 3 is arbitrary
        self.state_cov = process_noise_cov
        self.measurement_estimate = np.zeros(measurement_dim)
        self.measurement_cov = measurement_noise_cov
        self.cross_cov_meas = np.ones([self.state_dim,self.measurement_dim])
        self.measurement_transform_cov = np.vstack([np.hstack([self.state_cov,self.cross_cov_meas]),np.hstack([self.cross_cov_meas.T,self.measurement_cov])])


    def measurement_update(self,measurement): #Update such that we only use measurement space in measurement update
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
        sigma_p_init = self.state_estimate
        for i in range(1,self.total_dim//2):
            sigma_p_init = np.hstack([sigma_p_init,self.state_estimate])
        self.sigma_points_measurement[:,0] = sigma_p_init
        self.weights_measurement_update[0] = self.lamb_meas/(self.total_dim+self.lamb_meas)
        for i in range(self.total_dim):
            self.sigma_points_measurement[:,2*i+2] = self.sigma_points_measurement[:,0] + np.sqrt((self.total_dim+self.lamb_meas)*S[i])*Vt[i,:]
            self.sigma_points_measurement[:,2*i+1] = self.sigma_points_measurement[:,0] - np.sqrt((self.total_dim+self.lamb_meas)*S[i])*Vt[i,:]
            self.weights_measurement_update[2*i+2] = 1/(2*(self.total_dim+self.lamb_meas))
            self.weights_measurement_update[2*i+1] = 1/(2*(self.total_dim+self.lamb_meas))

    def _calculate_sigma_covariance(self):
        mu_z = self.weights_measurement_update[0] * self.sigma_points_measurement[:,0]
        for i in range(self.total_dim):
            mu_z += self.weights_measurement_update[2*i+2] * self.sigma_points_measurement[:,2*i+2]
            mu_z += self.weights_measurement_update[2*i+1] * self.sigma_points_measurement[:,2*i+1]
        self.state_estimate = mu_z[:2]
        self.measurement_estimate = mu_z[2:]
        mu_z = np.reshape(mu_z,[self.total_dim,1])
        self.measurement_transform_cov = self.weights_measurement_update[0] * (np.reshape(self.sigma_points_measurement[:,0],[self.total_dim,1])-mu_z)@(np.reshape(self.sigma_points_measurement[:,0],[self.total_dim,1])-mu_z).T
        for i in range(self.total_dim):
            self.measurement_transform_cov += self.weights_measurement_update[2*i+2] * (np.reshape(self.sigma_points_measurement[:,2*i+2],[self.total_dim,1])-mu_z)@(np.reshape(self.sigma_points_measurement[:,2*i+2],[self.total_dim,1])-mu_z).T
            self.measurement_transform_cov += self.weights_measurement_update[2*i+1] * (np.reshape(self.sigma_points_measurement[:,2*i+2],[self.total_dim,1])-mu_z)@(np.reshape(self.sigma_points_measurement[:,2*i+2],[self.total_dim,1])-mu_z).T
        self.measurement_transform_cov += (1-self.alpha_meas**2+self.beta_meas**2)*(np.reshape(self.sigma_points_measurement[:,0],[self.total_dim,1])-mu_z)@(np.reshape(self.sigma_points_measurement[:,0],[self.total_dim,1])-mu_z).T
        self.state_cov = self.measurement_transform_cov[0:2,0:2]
        self.cross_cov_meas = self.measurement_transform_cov[0:2,2:]
        self.measurement_cov = self.measurement_transform_cov[2:,2:]

    def _update_kalman_gain(self):
        self.kalman_gain = self.cross_cov_meas@np.linalg.pinv(self.measurement_cov)

    def _update_state_estimate_meas(self,measurement):
        meas_hat = self.state_estimate
        for i in range(1,self.measurement_dim//2):
            meas_hat = np.hstack([meas_hat,self.state_estimate])
        kalman_inno = (self.kalman_gain@np.reshape(measurement-meas_hat,[self.measurement_dim,1]))
        self.state_estimate += kalman_inno[:,0]

    def _update_state_cov(self):
        self.state_cov -= self.kalman_gain@self.measurement_cov@self.kalman_gain.T


    def time_update(self,delta_time, speed, cog):
        self._generate_sigma_points_time()
        self._propagate_sigma_points_time(delta_time,speed,cog)
        self._calculate_state_cov_time()

    def _generate_sigma_points_time(self):
        U,S,Vt = np.linalg.svd(self.state_cov)
        self.sigma_points_time[:,0] = self.state_estimate
        self.weights_time_update[0] = self.lamb_time/(self.state_dim+self.lamb_time)
        for i in range(self.state_dim):
            self.sigma_points_time[:,2*i+2] = self.sigma_points_time[:,0] + np.sqrt((self.state_dim+self.lamb_time)*S[i])*Vt[i,:]
            self.sigma_points_time[:,2*i+1] = self.sigma_points_time[:,0] - np.sqrt((self.state_dim+self.lamb_time)*S[i])*Vt[i,:]
            self.weights_time_update[2*i+2] = 1/(2*(self.state_dim+self.lamb_time))
            self.weights_time_update[2*i+1] = 1/(2*(self.state_dim+self.lamb_time))

    def _propagate_sigma_points_time(self, delta_time, speed,cog):
        for i in range(self.num_sigma_points_time):
            #propagate = np.array(self.calculate_destination_point(self.sigma_points[i,0],self.sigma_points[i,1],heading,delta_time*speed)) 
            self.sigma_points_time[:,i] = np.array(self.calculate_destination_point2(self.sigma_points_time[0,i],self.sigma_points_time[1,i],cog,delta_time*speed))

    def calculate_destination_point2(self,latitude, longitude, bearing, distance):
        # Statics
        a = 6378137
        b = 6356752.3142

        # Converting to radians
        lat_rad = math.radians(latitude)
        lon_rad = math.radians(longitude)
        bearing_rad = math.radians(bearing)

        # Distances in tangent plane
        d_lat = distance*math.cos(-bearing_rad)
        d_lon = -distance*math.sin(-bearing_rad)
        new_lat = latitude + d_lat * (360/(np.pi*b*2))
        new_lon = 1/((np.pi/180) * a * np.cos(lat_rad)) * d_lon + longitude

        return new_lat, new_lon

    def _calculate_state_cov_time(self):
        mu_z = self.weights_time_update[0] * self.sigma_points_time[:,0]
        for i in range(self.state_dim):
            mu_z += self.weights_time_update[2*i+2] * self.sigma_points_time[:,2*i+2]
            mu_z += self.weights_time_update[2*i+1] * self.sigma_points_time[:,2*i+1]
        self.state_estimate = mu_z
        mu_z = np.reshape(mu_z,[2,1])
        self.state_cov = self.weights_time_update[0] * (np.reshape(self.sigma_points_time[:,0],[2,1])-mu_z)@(np.reshape(self.sigma_points_time[:,0],[2,1])-mu_z).T
        for i in range(self.state_dim):
            self.state_cov += self.weights_time_update[2*i+2] * (np.reshape(self.sigma_points_time[:,2*i+2],[2,1])-mu_z)@(np.reshape(self.sigma_points_time[:,2*i+2],[2,1])-mu_z).T
            self.state_cov += self.weights_time_update[2*i+1] * (np.reshape(self.sigma_points_time[:,2*i+1],[2,1])-mu_z)@(np.reshape(self.sigma_points_time[:,2*i+1],[2,1])-mu_z).T
        self.state_cov += (1-self.alpha_meas**2+self.beta_meas**2)*(np.reshape(self.sigma_points_time[:,0],[2,1])-mu_z)@(np.reshape(self.sigma_points_time[:,0],[2,1])-mu_z).T

    def set_initial_state(self, initial_state):
        self.state_estimate = initial_state
    
    def set_initial_measurement(self, initial_measurement):
        self.measurement_mean = initial_measurement
        self.measurement_cov = np.eye(self.measurement_dim)
    
    def set_process_noise_cov(self, process_noise_cov):
        self.process_noise_cov = process_noise_cov
    
    def set_measurement_noise_cov(self, measurement_noise_cov):
        self.measurement_noise_cov = measurement_noise_cov
    
    def set_parameters_meas(self, lamb,alpha, kappa, beta):
        self.lamb_meas = lamb
        self.alpha_meas = alpha
        self.kappa_meas = kappa
        self.beta_meas = beta

    def set_parameters_time(self, lamb,alpha, kappa, beta):
        self.lamb_time = lamb
        self.alpha_time = alpha
        self.kappa_time = kappa
        self.beta_time = beta
    
    def get_state(self):
        return self.state_estimate
    
    def get_state_covariance(self):
        return self.state_cov
    
    def get_measurement(self):
        return self.measurement_mean
    
    def get_measurement_covariance(self):
        return self.measurement_cov

    def get_measurement_dim(self):
        return self.measurement_dim

    # Consider implementing for the case where bouyes come and go
    def update_measurement_dim(self,measurement_dim):
        curr_dim = self.measurement_dim
        self.measurement_dim = measurement_dim
        if curr_dim > measurement_dim:
            self.measurement_cov = self.measurement_cov[:measurement_dim,:measurement_dim]
            self.measurement_transform_cov = self.measurement_transform_cov[:measurement_dim+self.state_dim,:measurement_dim+self.state_dim]
        else:
            # Do stuff with 
            dim_diff = measurement_dim - curr_dim
            new_meas_cov = np.ones([measurement_dim,measurement_dim])
            new_trans_cov = np.ones([measurement_dim+self.state_dim,measurement_dim+self.state_dim])
            new_meas_cov[:curr_dim,:curr_dim] = self.measurement_cov
            new_trans_cov[:curr_dim+self.state_dim,:curr_dim+self.state_dim] = self.measurement_transform_cov
            for i in range(dim_diff//2):
                new_meas_cov[curr_dim+2*i:curr_dim+2*i+2,curr_dim+2*i:curr_dim+2*i+2] = self.measurement_cov[:2,:2]
                new_trans_cov[curr_dim+self.state_dim+2*i:curr_dim+self.state_dim+2*i+2,curr_dim+self.state_dim+2*i:curr_dim+self.state_dim+2*i+2] = self.measurement_transform_cov[2:4,2:4]
                new_trans_cov[curr_dim+self.state_dim+2*i:curr_dim+self.state_dim+2*i+2,:2] = self.measurement_transform_cov[2:4,:2]
                new_trans_cov[:2,curr_dim+self.state_dim+2*i:curr_dim+self.state_dim+2*i+2] = self.measurement_transform_cov[:2,2:4]
                for j in range(measurement_dim//2):
                    #Her skal jeg så have de der cross covariance først række og så kolonne
                    if not (2*j==curr_dim+2*i) and self.measurement_cov.shape[0] > 2:                    
                        new_meas_cov[curr_dim+2*i:curr_dim+2*i+2,2*j:2*j+2] = self.measurement_cov[2:4,:2]
                        new_meas_cov[2*j:2*j+2,curr_dim+2*i:curr_dim+2*i+2] = self.measurement_cov[:2,2:4]
                        new_trans_cov[curr_dim+self.state_dim+2*i:curr_dim+self.state_dim+2*i+2,self.state_dim+2*j:self.state_dim+2*j+2] = self.measurement_transform_cov[4:6,2:4]
                        new_trans_cov[self.state_dim+2*j:self.state_dim+2*j+2,curr_dim+self.state_dim+2*i:curr_dim+self.state_dim+2*i+2] = self.measurement_transform_cov[2:4,4:6]
            self.measurement_cov = new_meas_cov
            self.measurement_transform_cov = new_trans_cov
        self.total_dim = self.state_dim + measurement_dim
        self.num_sigma_points_measurement = 2 * self.total_dim + 1
        self.sigma_points_measurement = np.zeros((self.total_dim,self.num_sigma_points_measurement))
        self.weights_measurement_update = np.zeros(self.num_sigma_points_measurement)
        self.alpha_meas = np.sqrt(3/self.total_dim)
        self.lamb_meas = 3 - self.total_dim
        self.beta_meas = 3 / self.total_dim - 1