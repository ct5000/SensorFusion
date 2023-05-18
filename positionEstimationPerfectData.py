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



def calculate_destination_point(latitude, longitude, bearing, distance):
    R = 6371  # Radius of the Earth in kilometers

    # Convert latitude and longitude to radians
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)

    # Convert bearing to radians
    bearing_rad = math.radians(bearing)

    # Convert distance to kilometers
    distance_km = distance / 1000

    # Calculate the change in latitude
    lat_change = math.asin(math.sin(lat_rad) * math.cos(distance_km / R) +
                          math.cos(lat_rad) * math.sin(distance_km / R) * math.cos(bearing_rad))

    # Calculate the change in longitude
    lon_change = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_km / R) * math.cos(lat_rad),
                                      math.cos(distance_km / R) - math.sin(lat_rad) * math.sin(lat_change))

    # Convert back to degrees
    lat_change_deg = math.degrees(lat_change)
    lon_change_deg = math.degrees(lon_change)

    # Calculate the new latitude and longitude
    new_latitude = latitude + lat_change_deg
    new_longitude = longitude + lon_change_deg

    return new_latitude, new_longitude

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


plt.figure()
plt.imshow(enc_data[:,:,1,0])

bouye_meas = make_measurement_perfect_data(enc_data[:,:,1,0],enc_radius[:,0][0])
print(bouye_meas)
bouyes_pos = []
calculate_destination_point(latitude[:,0][0],longitude[:,0][0],bouye_meas[0][1],bouye_meas[0][0])
new_lat1, new_long1 = calculate_destination_point(latitude[:,0][0],longitude[:,0][0],bouye_meas[0][1],bouye_meas[0][0])
bouyes_pos.append([new_lat1,new_long1])
new_lat2, new_long2 = calculate_destination_point(latitude[:,0][0],longitude[:,0][0],bouye_meas[1][1],bouye_meas[1][0])
bouyes_pos.append([new_lat2,new_long2])
print(bouyes_pos)
''' This is a simulation/film where it is plotted on what is happening
plt.figure()
plt.subplot(1,2,1)
plt.subplot(1,2,2)
plt.ion()
plt.show()
for i in range(size):
    meas = make_measurement_perfect_data(target_data[:,:,i],target_radius[:,i][0])
    print(i)
    print(meas)
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(enc_data[:,:,:,i])
    plt.scatter(250,250)
    plt.subplot(1,2,2)
    plt.imshow(target_data[:,:,i])
    plt.scatter(250,250)
    plt.draw()
    plt.pause(0.5)

'''
    

plt.show()
