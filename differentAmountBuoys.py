import numpy as np 
import skimage.io
import pandas as pd
import matplotlib.pyplot as plt
import skimage.feature
import math
import UKF_simple
import cv2
#import UKF_simple_v2
import UKF_update_meas_dim


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
    buoy_idx = list(range(len(true_buoy_pos)))
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
            if ((np.sqrt(lat_meas_diff**2+lon_meas_diff**2)<0.001) and 
                (np.sqrt(lat_meas_diff**2+lon_meas_diff**2)<np.sqrt(lat_prev_diff**2+lon_prev_diff**2))):
                buoy_meas[j][0] = lat_meas
                buoy_meas[j][1] = lon_meas
                
                meas_idx[j] = i
    pop_idx = []
    for i in range(len(meas_idx)):
        if meas_idx[i] == -1:
            pop_idx.append(i)
    for i in range(len(pop_idx)):
        buoy_meas.pop(pop_idx[i]-i)
        meas_idx.pop(pop_idx[i]-i)
        buoy_idx.pop(pop_idx[i]-i)
    return buoy_meas, meas_idx, buoy_idx

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
size = 600 #Has tried 1005. Around 500 it becomes worse
data = allData.loc[0:size-1,['timestamp','enc_data','image_data','latitude','longitude','xbr_max_range','heading']]
speed_data_all = pd.read_csv("velocity.csv")
speed_data = speed_data_all.loc[0:5*size-1,['timestamp','sog','cog']]

timestamp = np.zeros([1,size])
radar_im = np.zeros([500,500,size])
xbr_radius = np.zeros([1,size])
longitude = np.zeros([1,size])
latitude = np.zeros([1,size])
heading = np.zeros([1,size])
enc_data = np.zeros([500,500,4,size])

sog_data = np.zeros([1,size*5])
cog_data = np.zeros([1,size*5])
vel_timestamp = np.zeros([1,size*5])



for i, row in data.iterrows():
    timestamp[:,i] = row['timestamp']
    xbr_radius[:,i] = row['xbr_max_range']
    enc_path = row['enc_data']
    enc_data[:,:,:,i] = skimage.io.imread("enc_data/"+enc_path[17:])
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

buoyes_pos_true =[[57+3.408/60,10+3.352/60],[57+3.257/60,10+3.384/60],[57+3.455/60,10+2.814/60],[57+3.488/60,10+2.889/60],[57+4.255/60,10+2.068/60],
                  [57+4.542/60,10+1.747/60],[57+4.761/60,10+1.483/60],[57+4.774/60,10+1.265/60],[57+5.022/60,10+1.087/60],[57+5.066/60,10+0.444/60],
                  [57+4.990/60,10+0.173/60],[57+5.183/60,10+0.006/60],[57+5.012/60,9+59.516/60],[57+4.940/60,9+59.018/60],[57+4.740/60,9+59.181/60],
                  [57+4.645/60,9+58.798/60],[57+4.154/60,9+57.639/60],[57+4.152/60,9+57.629/60],[57+4.119/60,9+57.703/60],[57+4.117/60,9+57.694/60],
                  [57+4.084/60,9+57.767/60],[57+4.002/60,9+57.758/60],[57+4.049/60,9+57.833/60],[57+4.047/60,9+57.823/60],[57+4.014/60,9+57.897/60],
                  [57+4.011/60,9+57.887/60],[57+3.954/60,9+58.005/60],[57+3.814/60,9+57.976/60],[57+3.595/60,9+57.732/60]] #Found on OpenSeaMap


kal_pos = np.zeros([2,size])
#pos_est = np.zeros([2,2,size])
meas_cov = np.zeros([4,4,size])
state_cov = np.zeros([2,2,size])


# Initialise UKF
UKF = UKF_update_meas_dim.UnscentedKalmanFilter(2,4,np.array(1*np.identity(2)),np.array(0.01*np.identity(4)))
UKF.set_initial_state(np.array([latitude[0,0],longitude[0,0]]))
measurement = make_measurement(radar_im[:,:,0],xbr_radius[0,0],heading[0,0])
buoy_pos, meas_idx, buoy_idx = find_buoys_meas(latitude[0,0],longitude[0,0],buoyes_pos_true,measurement)
pos_est = np.zeros([len(meas_idx),2])
for j in range(len(meas_idx)):
    est_pos = calculate_my_position(buoyes_pos_true[buoy_idx[j]][0],buoyes_pos_true[buoy_idx[j]][1],measurement[meas_idx[j]][1],measurement[meas_idx[j]][0])
    pos_est[j,:] = est_pos
UKF.set_initial_measurement(np.reshape(pos_est[:,:],[1,4]))
kal_pos[:,0] = UKF.get_state()


plt.figure()
plt.subplot(1,2,1)
plt.subplot(1,2,2)
plt.ion() # Turn on interactivity
plt.show()

meas_cov[:,:,0] = UKF.get_measurement_covariance()
state_cov[:,:,0] = UKF.get_state_covariance()

vel_idx = 0
for i in range(1,size):
    print(i)
    min_time_diff = abs(timestamp[0,i] - vel_timestamp[0,vel_idx])
    for j in range(vel_idx+1,5*size):
        if vel_timestamp[0,j] == 0:
            continue
        time_diff = abs(timestamp[0,i] - vel_timestamp[0,j])
        if time_diff < min_time_diff:
            vel_idx = j
        else:
            break
    print("idx",vel_idx)
    vel_ms = sog_data[0,vel_idx]*1.852/3.6
    head_cog = cog_data[0,vel_idx]
    print("delta time",timestamp[0,i]-timestamp[0,i-1])
    print("vel",vel_ms)
    print("cog",head_cog)
    UKF.time_update(timestamp[0,i]-timestamp[0,i-1],vel_ms,head_cog)
    measurement = make_measurement(radar_im[:,:,i],xbr_radius[0,i],heading[0,i])
    kal_pos[:,i] = UKF.get_state()
    #buoy_pos, meas_idx = find_buoys_meas(latitude[0,i],longitude[0,i],buoyes_pos_true,measurement)
    buoy_pos, meas_idx,buoy_idx = find_buoys_meas(kal_pos[0,i],kal_pos[1,i],buoyes_pos_true,measurement)
    if len(meas_idx) > 0:
        if UKF.get_measurement_dim() != 2*len(meas_idx):
            print("here", len(meas_idx))
            UKF.update_measurement_dim(2*len(meas_idx))
        pos_est = np.zeros([len(meas_idx),2])
        print("Num meas: ",len(meas_idx))
        for j in range(len(meas_idx)):
            est_pos = calculate_my_position(buoyes_pos_true[buoy_idx[j]][0],buoyes_pos_true[buoy_idx[j]][1],measurement[meas_idx[j]][1],measurement[meas_idx[j]][0])
            pos_est[j,:] = est_pos
        UKF.measurement_update(np.reshape(pos_est[:,:],[1,2*len(meas_idx)]))
    kal_pos[:,i] = UKF.get_state()
    #meas_cov[:,:,i] = UKF.get_measurement_covariance()
    state_cov[:,:,i] = UKF.get_state_covariance()

    
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(enc_data[:,:,:,i].astype(int))
    plt.subplot(1,2,2)
    plt.imshow(radar_im[:,:,i])
    plt.draw()
    plt.pause(0.05)

plt.ioff()    

plt.figure()
#plt.plot(pos_est[0,1,1:],pos_est[0,0,1:],label='Est1')
#plt.plot(pos_est[1,1,1:],pos_est[1,0,1:],label='Est2')
plt.plot(longitude[0,1:],latitude[0,1:],label='True')
plt.plot(kal_pos[1,1:],kal_pos[0,1:],label="Kalman")
plt.ylim([57.05,57.09])
plt.xlim([9.95,10.08])
plt.legend()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Position of the vessel")

print(kal_pos[1,-1]-longitude[0,-1])
print(kal_pos[0,-1]-latitude[0,-1])

d_pos = np.sqrt(((kal_pos[1,size-1]-longitude[0,size-1])*60000)**2+((kal_pos[0,size-1]-latitude[0,size-1])*111000)**2)
print("diff pos: ", d_pos)

plt.figure()
plt.plot(state_cov[0,0,:],label="State cov 1")
plt.plot(state_cov[1,1,:],label="State cov 2")
plt.plot(meas_cov[0,0,:],label="Meas cov 1")
plt.plot(meas_cov[1,1,:],label="Meas cov 2")
plt.plot(meas_cov[2,2,:],label="Meas cov 3")
plt.plot(meas_cov[3,3,:],label="Meas cov 4")
plt.legend()
plt.xlabel("Sample")
plt.ylabel("Variance")
plt.title("Variance over time")


plt.figure()
plt.subplot(1,2,1)
plt.plot(longitude[0,1:],label="True")
plt.plot(kal_pos[1,1:],label="Kalman")
plt.xlabel("Sample")
plt.ylabel("Longitude")
plt.subplot(1,2,2)
plt.plot(latitude[0,1:],label='True')
plt.plot(kal_pos[0,1:],label="Kalman")
plt.xlabel("Sample")
plt.ylabel("Latitude")
plt.suptitle("Latitude and logitude over time")

plt.figure()
plt.plot(heading[0,:])
plt.xlabel("Sample")
plt.ylabel("Heading")
plt.title("Heading over time")


plt.figure()
plt.plot(xbr_radius[0,:])

plt.show()