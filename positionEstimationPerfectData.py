import numpy as np 
import skimage.io
import pandas as pd
import matplotlib.pyplot as plt
import skimage.feature
import math


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
Make measurement on image with a given radius with only bouyes present
"""
def make_measurement_perfect_data(img,max_radius):
    blobs = skimage.feature.blob_dog(img,min_sigma=5,max_sigma=20,threshold=1)
    centers = [(blob[1], blob[0]) for blob in blobs]
    meassurements = []
    for center in centers:
        distance, bearing = read_blob_distance_bearing(center,max_radius)
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

# Setting up the necessary data 
allData = pd.read_csv("dataset.csv")
size = 65
data = allData.loc[0:size-1,['timestamp','enc_data','enc_max_range','latitude','longitude','heading','target_data','max_radius']]

timestamp = np.zeros([1,size])
enc_radius = np.zeros([1,size])
longitude = np.zeros([1,size])
latitude = np.zeros([1,size])
heading = np.zeros([1,size])
target_radius = np.zeros([1,size])
enc_data = np.zeros([500,500,4,size])
target_data = np.zeros([500,500,size])

for i, row in data.iterrows():
    timestamp[:,i] = row['timestamp']
    enc_radius[:,i] = row['enc_max_range']
    longitude[:,i] = row['longitude']
    latitude[:,i] = row['latitude']
    heading[:,i] = row['heading']
    target_radius[:,i] = row['max_radius']
    enc_path = row['enc_data']
    enc_data[:,:,:,i] = skimage.io.imread("enc_data/"+enc_path[17:])
    target_path = row['target_data']
    target = np.load("/home/christian/targetData/"+target_path[29:])
    target_data[:,:,i] = target[:,:,1]

'''
plt.figure()
plt.imshow(enc_data[:,:,1,0])
plt.scatter(250,250)
'''
bouyes_pos_true =[[57+3.408/60,10+3.352/60],[57+3.257/60,10+3.384/60]] #Found on OpenSeaMap

#err = np.zeros([2,2,2,size])
measurements = np.zeros([2,2,size])
bouyes_pos_est = np.zeros([2,2,size])
pos_est = np.zeros([2,2,size])

#This is a simulation/film where it is plotted on what is happening
plt.figure()
plt.subplot(1,2,1)
plt.subplot(1,2,2)
plt.ion() # Turn on interactivity
plt.show()
for i in range(size):
    measurement = make_measurement_perfect_data(enc_data[:,:,1,i],enc_radius[:,i][0])
    measurements[:,:,i] = measurement
    for j in range(len(measurement)):
        est_pos = calculate_my_position(bouyes_pos_true[j][0],bouyes_pos_true[j][1],measurement[j][1],measurement[j][0])
        #est_pos1 = calculate_my_position(bouyes_pos_true[1][0],bouyes_pos_true[1][1],measurement[j][1],measurement[j][0])
        #err[j,0,0,i] = est_pos0[0]-latitude[0,i]
        #err[j,0,1,i] = est_pos0[1]-longitude[0,i]
        #err[j,1,0,i] = est_pos1[0]-latitude[0,i]
        #err[j,1,1,i] = est_pos1[1]-longitude[0,i]
        #err[j,0,0,i] = est_pos0[0] - bouyes_pos_true[0][0]
        #err[j,0,1,i] = est_pos0[1] - bouyes_pos_true[0][1]
        #err[j,1,0,i] = est_pos0[0] - bouyes_pos_true[1][0]
        #err[j,1,1,i] = est_pos0[1] - bouyes_pos_true[1][1]
        #bouyes_pos_est[j,:,i] = est_pos0
        pos_est[j,:,i] = est_pos
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(enc_data[:,:,:,i].astype(int))
    plt.scatter(250,250)
    plt.subplot(1,2,2)
    plt.imshow(target_data[:,:,i])
    plt.scatter(250,250)
    plt.draw()
    plt.pause(0.005)
plt.ioff() # Turns of interactivity

''' Error plots
plt.figure()
plt.subplot(2,2,1)
plt.plot(err[0,0,0,:])

plt.subplot(2,2,2)
plt.plot(err[0,0,1,:])

plt.subplot(2,2,3)
plt.plot(err[0,1,0,:])

plt.subplot(2,2,4)
plt.plot(err[0,1,1,:])
    
plt.figure()
plt.subplot(2,2,1)
plt.plot(err[1,0,0,:])

plt.subplot(2,2,2)
plt.plot(err[1,0,1,:])

plt.subplot(2,2,3)
plt.plot(err[1,1,0,:])

plt.subplot(2,2,4)
plt.plot(err[1,1,1,:])
'''

plt.figure()
plt.plot(pos_est[0,1,:],pos_est[0,0,:],label='Est1')
plt.plot(pos_est[1,1,:],pos_est[1,0,:],label='Est2')
plt.plot(longitude[0,:],latitude[0,:],label='True')
plt.ylim([57.03,57.07])
plt.xlim([10.04,10.08])
plt.legend()

''' Plots of measurements and the estimated position of the bouyes together with the variance in the estimate

plt.figure()
plt.subplot(2,2,1)
plt.plot(measurements[0,0,:])
plt.subplot(2,2,2)
plt.plot(measurements[0,1,:])
plt.subplot(2,2,3)
plt.plot(measurements[1,0,:])
plt.subplot(2,2,4)
plt.plot(measurements[1,1,:])

# Try making a test with ECEF frame
# Try changing how the projection is done
plt.figure()
plt.plot(bouyes_pos_est[0,1,:],bouyes_pos_est[0,0,:])
plt.plot(bouyes_pos_est[1,1,:],bouyes_pos_est[1,0,:])
plt.ylim([57.03,57.07])
plt.xlim([10.04,10.08])

print("Variances of estimates")
print(np.var(bouyes_pos_est[0,1,:]),np.var(bouyes_pos_est[0,0,:]))
print(np.var(bouyes_pos_est[1,1,:]),np.var(bouyes_pos_est[1,0,:]))
'''



plt.show()

