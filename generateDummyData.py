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
    v = np.array([blob_center[0]-250,-(blob_center[1]-250)])
    distance = (math.sqrt((v[0])**2+(v[1])**2)/250)*max_radius
    bearing = math.atan2(v[1],v[0])
    return distance, convert_to_navigation_angle(bearing)

def make_measurement_perfect_data(img,max_radius):
    blobs = skimage.feature.blob_dog(img,min_sigma=5,max_sigma=20,threshold=1)
    centers = [(blob[1], blob[0]) for blob in blobs]
    meassurements = []
    for center in centers:
        distance, bearing = read_blob_distance_bearing(center,max_radius)
        meassurements.append([distance,bearing])
    return meassurements


'''
Calculates position of a point from a distance and a bearing
'''
'''
def calculate_destination_point(latitude, longitude, bearing, distance):
    R = 6378137  # Radius of the Earth in meters
    b = 6356752.3142 

    # Convert latitude and longitude to radians
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)

    # Convert bearing to radians
    bearing_rad = math.radians(bearing)

    # Calculate the change in latitude
    delta_lat = (2*b*math.pi) / 360
    lat_change = distance*math.cos(-bearing_rad)/delta_lat

    # Calculate the change in longitude
    e2 = (R**2 - b**2)/(R**2)
    delta_lon = (math.pi * R * math.cos(lat_rad)) / (180 * math.sqrt(1 - e2 * math.sin(lat_rad)**2))
    lon_change = -distance*math.sin(-bearing_rad)/delta_lon

    # Calculate the new latitude and longitude
    new_latitude = latitude + lat_change
    new_longitude = longitude + lon_change

    return new_latitude, new_longitude
'''

"""
Converts from geodetic coordinates to ECEF. Takes in latitude and longitude in radians
"""
def geodeticToECEF(latitude,longitude,h):
    # Defining parameters
    a = 6378137
    f  = 1 / 298.257223563
    b = a * (1-f)
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
    b = a * (1-f)
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
    # Defining parameters
    #a = 6378137
    #f  = 1 / 298.257223563
    #b = a * (1-f)
    #e = math.sqrt(f*(2-f))
    #RN = a / (math.pow(1-e**2*(math.sin(latitude)),0.5))

    # Converting to radians
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)
    bearing_rad = math.radians(bearing)

    # Converting to ECEF
    x ,y , z = geodeticToECEF(lat_rad,lon_rad,0)
    lat_test, lon_test, h_test = ECEFtoGeodetic(x,y,z)

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

square = np.ones([3,3])*255
n = 498
ims = np.zeros([500,500,n])

for i in range(n):
    im = np.zeros([500,500])
    square_center = np.array([i+1, 200])
    im[99:102,i:i+3] = square
    ims[:,:,i] = im


latitude = 57
longitude = 10

est_pos = np.zeros([2,n])

for i in range(n):
    measurement = make_measurement_perfect_data(ims[:,:,i],500)
    lat,lon = calculate_destination_point(latitude,longitude,measurement[0][1],measurement[0][0])
    est_pos[:,i] = np.array([lat,lon])

print(est_pos[0,:10])
plt.figure()
plt.subplot(1,2,1)
plt.plot(est_pos[0,:])

plt.subplot(1,2,2)
plt.plot(est_pos[1,:])

plt.figure()
plt.plot(est_pos[1,:],est_pos[0,:])
plt.ylim(56.99,57.01)

plt.show()