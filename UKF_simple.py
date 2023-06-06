import numpy as np
import math
import pyproj

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
    
    def predict(self, delta_time,speed,heading):
        self._generate_sigma_points()
        self._propagate_sigma_points(delta_time,speed,heading)
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
        self.sigma_points[0,:] = self.state_mean
        self.weights_mean[0] = self.lamb/(self.state_dim+self.lamb)
        scaled_sqrt_cov = np.sqrt(self.state_dim + self.lamb) * sqrt_cov
        for i in range(self.state_dim):
            self.sigma_points[i + 1] = self.state_mean + scaled_sqrt_cov[i]
            self.sigma_points[i + 1 + self.state_dim] = self.state_mean - scaled_sqrt_cov[i]
            self.weights_mean[i + 1] = 1 / (2*(self.state_dim+self.lamb))
            self.weights_mean[i + 1 + self.state_dim] = 1 / (2*(self.state_dim+self.lamb))
        #self.weights_cov = np.cov(self.weights_mean)
        #print(self.weights_cov)
    
    def _propagate_sigma_points(self, delta_time, speed, heading):
        # Implement the process model to propagate sigma points
        # Update self.sigma_points with the propagated sigma points
        for i in range(self.num_sigma_points):
            #propagate = np.array(self.calculate_destination_point(self.sigma_points[i,0],self.sigma_points[i,1],heading,delta_time*speed)) 
            self.sigma_points[i,:] = np.array(self.calculate_destination_point2(self.sigma_points[i,0],self.sigma_points[i,1],heading,delta_time*speed))  
            for j in range(int(self.measurement_dim/2)):
                self.sigma_points_measurements[i,2*j] = self.sigma_points[i,0]
                self.sigma_points_measurements[i,2*j+1] = self.sigma_points[i,1]


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


    def calculate_destination_point(self,latitude, longitude, bearing, distance):
        # Converting to radians
        lat_rad = math.radians(latitude)
        lon_rad = math.radians(longitude)
        bearing_rad = math.radians(bearing)

        # Converting to ECEF
        x ,y , z = self.geodeticToECEF(lat_rad,lon_rad,0)

        # Distances in tangent plane
        d_xt = distance*math.cos(-bearing_rad)
        d_yt = -distance*math.sin(-bearing_rad)
        d_zt = 0
        d_vect = np.array([[d_xt,d_yt,d_zt]]).T

        # Define Transformation matrix
        Rt_e = np.array([[-math.sin(lat_rad)*math.cos(lon_rad),-math.sin(lat_rad)*math.sin(lon_rad),math.cos(lat_rad)],
                        [-math.sin(lon_rad),math.cos(lon_rad),0],
                        [-math.cos(lat_rad)*math.cos(lon_rad),-math.cos(lat_rad)*math.sin(lon_rad),-math.sin(lat_rad)]])
        Re_t = Rt_e.T

        # New ECEF coordinate
        d_vec = Re_t@d_vect
        x_new = x + d_vec[0,0]
        y_new = y + d_vec[1,0]
        z_new = z + d_vec[2,0]
        print("d_vect: ",d_vect)
        print("d_vec: ", d_vec)
        print("z: ",z,"z_new: ", z_new)

        # Convert to geodetic
        lat_new_rad, lon_new_rad, h_new = self.ECEFtoGeodetic(x_new,y_new,z_new)

        transformer = pyproj.Transformer.from_crs({"proj":'geocent',"ellps":'WGS84',"datum":'WGS84'},
                                                  {"proj":'latlong',"ellps":'WGS84',"datum":'WGS84'})
        lon2, lat2, alt2 = transformer.transform(x_new,y_new,z_new,radians=False)
        lat_new_deg = math.degrees(lat_new_rad)
        lon_new_deg = math.degrees(lon_new_rad)
        print("Latitude old: ", latitude)
        print("Latitude new: ", lat_new_deg)
        print("Lat 2", lat2)
        print("alt2: ", alt2)
        return lat_new_deg,lon_new_deg
    
    """
    Converts from geodetic coordinates to ECEF. Takes in latitude and longitude in radians
    """
    def geodeticToECEF(self,latitude,longitude,h):
        # Defining parameters
        a = 6378137
        f  = 1 / 298.257223563
        e = math.sqrt(f*(2-f))
        RN = a / (math.pow(1-e**2*(math.sin(latitude)),0.5))
        # Converting to ECEF
        x = (RN+h)*math.cos(latitude)*math.cos(longitude)
        y = (RN+h)*math.cos(latitude)*math.sin(longitude)
        z = (RN*(1-e**2)+h)*math.sin(latitude)

        return x,y,z

    """
    Converts from ECEF to Geoditic which is returned in radians
    """
    def ECEFtoGeodetic(self,x,y,z):
        # Initialise
        a = 6378137
        f  = 1 / 298.257223563
        e = math.sqrt(f*(2-f))
        longitude = math.atan2(y,x)
        h = 0
        RN = a
        p = math.sqrt(x**2+y**2)
        latitude = 0
        lat_prev = 1
        h_prev = 1

        # Iteration
        while (abs(latitude-lat_prev)>1e-16 or abs(h-h_prev)>1e-5):
            lat_prev = latitude
            h_prev = h
            sinLat = z / ((1-e**2)*RN+h)
            latitude = math.atan((z+e**2*RN*sinLat)/p)
            RN = a / (math.sqrt(1-e**2*sinLat**2))
            h = p / math.cos(latitude) - RN
        return latitude, longitude, h


    def _compute_predicted_state(self):
        self.state_mean = np.average(self.sigma_points, axis=0, weights=self.weights_mean)
    
    def _compute_predicted_measurement(self):
        # Implement the measurement model to compute predicted measurements
        # Update self.measurement_mean with the predicted measurement mean
        for i in range(int(self.measurement_dim/2)):
            self.measurement_mean[0,2*i] = self.state_mean[0]
            self.measurement_mean[0,2*i+1] = self.state_mean[1]
        
    def _compute_predicted_covariance(self):
        centered_points = self.sigma_points - self.state_mean
        self.state_cov = (self.weights_cov[:, np.newaxis] * centered_points).T @ centered_points
        #self.state_cov = (self.weights_cov * centered_points).T @ centered_points
        self.state_cov += self.process_noise_cov + (1-self.alpha**2 + self.beta)*(centered_points[0,:].T@centered_points[0,:])
        print("_compute_predicted_covariance")
        print(self.state_cov)

    def _compute_sigma_covariance(self):
        z_points_mean = np.vstack([self.state_mean,self.measurement_mean])
        comb_sigma_points = np.vstack([self.sigma_points,self.sigma_points_measurements])
        centered_points = comb_sigma_points - z_points_mean
        self.cross_cov = centered_points.T @ centered_points

    def _compute_cross_covariance(self):
        centered_states = self.sigma_points - self.state_mean
        centered_measurements = self.sigma_points_measurements - self.measurement_mean
        print("_compute_cross_covariance")
        print(centered_states)
        print(centered_measurements)
        print(self.weights_cov)
        self.cross_cov = (self.weights_cov[:, np.newaxis] * centered_states).T @ centered_measurements
        #self.cross_cov = (self.weights_cov * centered_states).T @ centered_measurements
    
    def _compute_kalman_gain(self):
        innovation_cov =  self.measurement_noise_cov # Look into updating the measurement_cov during the calculations
        print("_compute_kalman_gain")
        print(innovation_cov)
        print(self.cross_cov)
        self.kalman_gain = self.cross_cov @ np.linalg.inv(innovation_cov)
    
    def _update_state(self, measurement):
        innovation = measurement - self.measurement_mean
        print("_update_state")
        print(innovation)
        print(self.kalman_gain)
        self.state_mean += np.reshape(self.kalman_gain @ innovation.T,[2])
        
    
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