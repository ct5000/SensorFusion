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
bouye_meas = make_measurement_perfect_data(enc_data[:,:,1,0],enc_radius[0,0])

bouyes_pos = []
bouyes_pos.append(calculate_destination_point(latitude[:,0][0],longitude[:,0][0],bouye_meas[0][1],bouye_meas[0][0]))
bouyes_pos.append(calculate_destination_point(latitude[:,0][0],longitude[:,0][0],bouye_meas[1][1],bouye_meas[1][0]))
print(bouyes_pos)

bouyes_pos_true =[[57+3.408/60,10+3.352/60],[57+3.257/60,10+3.384/60]]
print(bouyes_pos_true)

lat, lon, h = ECEFtoGeodetic(-2430601.828,-4702442.703,3546587.358)
print(math.degrees(lat),math.degrees(lon),h)

err = np.zeros([2,2,2,size])
measurements = np.zeros([2,2,size])


#This is a simulation/film where it is plotted on what is happening
plt.figure()
plt.subplot(1,2,1)
plt.subplot(1,2,2)
plt.ion()
plt.show()
for i in range(size):
    measurement = make_measurement_perfect_data(enc_data[:,:,1,i],enc_radius[:,i][0])
    measurements[:,:,i] = measurement
    for j in range(len(measurement)):
        #est_pos0 = calculate_destination_point(bouyes_pos_true[0][0],bouyes_pos_true[0][1],(180-measurement[j][1])%360,measurement[j][0])
        #est_pos1 = calculate_destination_point(bouyes_pos_true[1][0],bouyes_pos_true[1][1],(180-measurement[j][1])%360,measurement[j][0])
        est_pos0 = calculate_destination_point(latitude[0,i],longitude[0,i],measurement[j][1],measurement[j][0])
        #print(est_pos0[0]-latitude[0,i],est_pos0[1]-longitude[:,i][0])
        #print(est_pos1[0]-latitude[:,i][0],est_pos1[1]-longitude[:,i][0])
        #print(est_pos[0]-latitude[:,i][0],est_pos[1]-longitude[:,i][0])
        #err[j,0,0,i] = est_pos0[0]-latitude[0,i]
        #err[j,0,1,i] = est_pos0[1]-longitude[0,i]
        #err[j,1,0,i] = est_pos1[0]-latitude[0,i]
        #err[j,1,1,i] = est_pos1[1]-longitude[0,i]
        err[j,0,0,i] = est_pos0[0] - bouyes_pos_true[0][0]
        err[j,0,1,i] = est_pos0[1] - bouyes_pos_true[0][1]
        err[j,1,0,i] = est_pos0[0] - bouyes_pos_true[1][0]
        err[j,1,1,i] = est_pos0[1] - bouyes_pos_true[1][1]

    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(enc_data[:,:,:,i].astype(int))
    plt.scatter(250,250)
    plt.subplot(1,2,2)
    plt.imshow(target_data[:,:,i])
    plt.scatter(250,250)
    plt.draw()
    plt.pause(0.005)

plt.ioff()

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
plt.plot(err[1,1,0,:])

plt.subplot(2,2,2)
plt.plot(err[1,1,1,:])

plt.subplot(2,2,3)
plt.plot(err[1,1,0,:])

plt.subplot(2,2,4)
plt.plot(err[1,1,1,:])


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



plt.show()
