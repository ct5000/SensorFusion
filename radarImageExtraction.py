import numpy as np 
import skimage.io
import pandas as pd
import matplotlib.pyplot as plt
import skimage.feature
import math
import UKF_simple
import cv2

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
def make_measurement(img,max_radius):
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


def find_true_buoys(latitude, longitude, true_buoy_pos,measurement):
    true_buoy = False
    meas_pos = calculate_destination_point(latitude,longitude,measurement[1],measurement[0])
    return true_buoy


allData = pd.read_csv("dataset.csv")
size = 65
data = allData.loc[0:size-1,['timestamp','image_data','latitude','longitude','xbr_max_range']]
speed_data_all = pd.read_csv("velocity.csv")
speed_data = speed_data_all.loc[0:5*size-1,['timestamp','sog','cog']]

timestamp = np.zeros([1,size])
radar_im = np.zeros([500,500,size])
xbr_radius = np.zeros([1,size])
longitude = np.zeros([1,size])
latitude = np.zeros([1,size])
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

for i, row in speed_data.iterrows():
    if row['sog'] > 0:
        sog_data[:,i] = row['sog']
        cog_data[:,i] = row['cog']
        vel_timestamp[:,i] = row['timestamp']

bouyes_pos_true =[[57+3.408/60,10+3.352/60],[57+3.257/60,10+3.384/60]] #Found on OpenSeaMap


meas = make_measurement(radar_im[:,:,0],xbr_radius[0,0])
print(len(meas))

quit()
plt.figure()
plt.subplot(1,2,1)
plt.subplot(1,2,2)
plt.ion() # Turn on interactivity
plt.show()

for i in range(size):
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(radar_im[:,:,i])
    plt.subplot(1,2,2)


    plt.draw()
    plt.pause(0.05)



