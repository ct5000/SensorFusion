import numpy as np 
import skimage.io
import pandas as pd
import matplotlib.pyplot as plt
import skimage.feature
import math
import UKF_simple
import cv2
import UKF_simple_v2

'''
Converts from cartesian coordinate system in radians to navigational coordinate system in degrees with reference North
'''
def convert_to_navigation_angle(angle_rad):
    angle_deg = math.degrees(angle_rad)
    navigation_angle_deg = (90 - angle_deg) % 360
    return navigation_angle_deg

'''
Takes blob center and radius of the sensing in meter and converts it to a distance and bering to the blob
'''
def read_blob_distance_bearing(blob_center,max_radius):
    v = np.array([blob_center[0]-249.5,-(blob_center[1]-249.5)])
    distance = (math.sqrt((v[0])**2+(v[1])**2)/249.5)*max_radius
    bearing = math.atan2(v[1],v[0])
    return distance, convert_to_navigation_angle(bearing)


"""
Make measurement on a radar image with a given radius 
"""
def make_measurement(img,max_radius,heading):
    blobs = skimage.feature.blob_dog(img,min_sigma=5,max_sigma=20,threshold=1)
    centers = [(blob[1], blob[0]) for blob in blobs]
    meassurements = []
    for center in centers:
        distance, bearing_rel = read_blob_distance_bearing(center,max_radius)
        bearing = (bearing_rel + heading)%360
        meassurements.append([distance,bearing])
    return meassurements

"""
Converts from geodetic coordinates to ECEF. Takes in latitude and longitude in radians
"""
def geodeticToECEF(latitude,longitude,h):
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
def ECEFtoGeodetic(x,y,z):
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
    while (abs(latitude-lat_prev)>1e-10 or abs(h-h_prev)>1e-2):
        lat_prev = latitude
        h_prev = h
        sinLat = z / ((1-e**2)*RN+h)
        latitude = math.atan((z+e**2*RN*sinLat)/p)
        RN = a / (math.sqrt(1-e**2*sinLat**2))
        #print(RN)
        #print(p / math.cos(latitude))
        h = p / math.cos(latitude) - RN
    return latitude, longitude, h


def calculate_destination_point(latitude, longitude, bearing, distance):
    # Converting to radians
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)
    bearing_rad = math.radians(bearing)

    # Converting to ECEF
    x ,y , z = geodeticToECEF(lat_rad,lon_rad,0)

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

    # Convert to geodetic
    lat_new_rad, lon_new_rad, h_new = ECEFtoGeodetic(x_new,y_new,z_new)

    lat_new_deg = math.degrees(lat_new_rad)
    lon_new_deg = math.degrees(lon_new_rad)

    return lat_new_deg,lon_new_deg


def find_buoys_meas(latitude, longitude, true_buoy_pos,measurements):
    buoy_meas = []
    meas_idx = []
    for buoy in true_buoy_pos:
        buoy_meas.append([0,0])
        meas_idx.append(-1)
    for i in range(len(measurements)):
        lat_meas,lon_meas = calculate_destination_point(latitude,longitude,measurements[i][1],measurements[i][0])
        for j in range(len(true_buoy_pos)):
            lat_meas_diff = abs(true_buoy_pos[j][0]-lat_meas)
            lon_meas_diff = abs(true_buoy_pos[j][1]-lon_meas)
            lat_prev_diff = abs(true_buoy_pos[j][0]-buoy_meas[j][0])
            lon_prev_diff = abs(true_buoy_pos[j][1]-buoy_meas[j][1])
            if ((np.sqrt(lat_meas_diff**2+lon_meas_diff**2)<0.01) and 
                (np.sqrt(lat_meas_diff**2+lon_meas_diff**2)<np.sqrt(lat_prev_diff**2+lon_prev_diff**2))):
                buoy_meas[j][0] = lat_meas
                buoy_meas[j][1] = lon_meas
                meas_idx[j] = i
    return buoy_meas, meas_idx

def calculate_my_position(bouey_lat, bouye_lon, bearing, distance):
    # Converting to radians
    lat_rad = math.radians(bouey_lat)
    lon_rad = math.radians(bouye_lon)
    bearing_rad = math.radians(bearing)

    # Convert to ECEF
    x ,y , z = geodeticToECEF(lat_rad,lon_rad,0)

    # Distances in tangent plane
    d_xt = -distance*math.cos(-bearing_rad)
    d_yt = distance*math.sin(-bearing_rad)
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

    # Convert to geodetic
    lat_new_rad, lon_new_rad, h_new = ECEFtoGeodetic(x_new,y_new,z_new)
    lat_new_deg = math.degrees(lat_new_rad)
    lon_new_deg = math.degrees(lon_new_rad)
    return lat_new_deg,lon_new_deg


allData = pd.read_csv("dataset.csv")
size = 65
data = allData.loc[0:size-1,['timestamp','image_data','latitude','longitude','xbr_max_range','heading']]
speed_data_all = pd.read_csv("velocity.csv")
speed_data = speed_data_all.loc[0:5*size-1,['timestamp','sog','cog']]

timestamp = np.zeros([1,size])
radar_im = np.zeros([500,500,size])
xbr_radius = np.zeros([1,size])
longitude = np.zeros([1,size])
latitude = np.zeros([1,size])
heading = np.zeros([1,size])

sog_data = np.zeros([1,size*5])
cog_data = np.zeros([1,size*5])
vel_timestamp = np.zeros([1,size*5])



for i, row in data.iterrows():
    timestamp[:,i] = row['timestamp']
    xbr_radius[:,i] = row['xbr_max_range']
    radar_path = row['image_data']
    radar_im[:,:,i] = skimage.io.imread("imageData/"+radar_path[17:])
    longitude[:,i] = row['longitude']
    latitude[:,i] = row['latitude']
    heading[:,i] = row['heading']

for i, row in speed_data.iterrows():
    if row['sog'] > 0:
        sog_data[:,i] = row['sog']
        cog_data[:,i] = row['cog']
        vel_timestamp[:,i] = row['timestamp']

buoyes_pos_true =[[57+3.408/60,10+3.352/60],[57+3.257/60,10+3.384/60]] #Found on OpenSeaMap


kal_pos = np.zeros([2,size])
pos_est = np.zeros([2,2,size])

# Initialise UKF
UKF = UKF_simple_v2.UnscentedKalmanFilter(2,4,np.array(10*np.identity(2)),np.array(0.01*np.identity(4)))
UKF.set_initial_state(np.array([latitude[0,0],longitude[0,0]]))
measurement = make_measurement(radar_im[:,:,0],xbr_radius[0,0],heading[0,0])
buoy_pos, meas_idx = find_buoys_meas(latitude[0,0],longitude[0,0],buoyes_pos_true,measurement)

for j in range(len(meas_idx)):
    est_pos = calculate_my_position(buoyes_pos_true[j][0],buoyes_pos_true[j][1],measurement[meas_idx[j]][1],measurement[meas_idx[j]][0])
    pos_est[j,:,0] = est_pos
UKF.set_initial_measurement(np.reshape(pos_est[:,:,0],[1,4]))
kal_pos[:,0] = UKF.get_state()


#plt.figure()

#plt.ion() # Turn on interactivity
#plt.show()

vel_idx = 0
for i in range(1,size):
    print(i)
    min_time_diff = abs(timestamp[0,i] - vel_timestamp[0,vel_idx])
    for j in range(vel_idx+1,5*size):
        time_diff = abs(timestamp[0,i] - vel_timestamp[0,j])
        if time_diff < min_time_diff:
            vel_idx = j
        else:
            break
    vel_ms = sog_data[0,vel_idx]*1.852/3.6
    head_cog = cog_data[0,vel_idx]
    print("delta time",timestamp[0,i]-timestamp[0,i-1])
    print("vel",vel_ms)
    print("cog",head_cog)
    UKF.time_update(timestamp[0,i]-timestamp[0,i-1],vel_ms,head_cog)
    measurement = make_measurement(radar_im[:,:,i],xbr_radius[0,i],heading[0,i])
    kal_pos[:,i] = UKF.get_state()
    #buoy_pos, meas_idx = find_buoys_meas(latitude[0,i],longitude[0,i],buoyes_pos_true,measurement)
    buoy_pos, meas_idx = find_buoys_meas(kal_pos[0,i],kal_pos[1,i],buoyes_pos_true,measurement)
    for j in range(len(meas_idx)):
        est_pos = calculate_my_position(buoyes_pos_true[j][0],buoyes_pos_true[j][1],measurement[meas_idx[j]][1],measurement[meas_idx[j]][0])
        pos_est[j,:,i] = est_pos
    UKF.measurement_update(np.reshape(pos_est[:,:,i],[1,4]))
    kal_pos[:,i] = UKF.get_state()

    '''
    plt.clf()
    plt.imshow(radar_im[:,:,i])
    plt.draw()
    plt.pause(0.05)
    '''

plt.figure()
plt.plot(pos_est[0,1,1:],pos_est[0,0,1:],label='Est1')
plt.plot(pos_est[1,1,1:],pos_est[1,0,1:],label='Est2')
plt.plot(longitude[0,1:],latitude[0,1:],label='True')
plt.plot(kal_pos[1,1:],kal_pos[0,1:],label="Kalman")
plt.ylim([57.03,57.07])
plt.xlim([10.04,10.08])
plt.legend()



plt.show()