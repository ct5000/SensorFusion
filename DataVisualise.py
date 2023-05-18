import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import skimage.io
from time import sleep


data = pd.read_csv("dataset.csv")
print(data.head(1))

print(data.info())

latitude = data["latitude"]
longitude = data["longitude"]
enc_data = data["enc_data"] #Fixed
time = data["timestamp"]

target_data = data["target_data"] #Fixed
img_data = data["image_data"] #Fixed
lidar = data["lidar1"] #fixed

wradar0 = data["wradar_group0_id0"] #Fixed
wradar1 = data["wradar_group0_id1"] #Fixed
print(type(wradar0))

time_np = np.array(time.array)
time_diff = time_np[1:] - time_np[:-1]
print(np.mean(time_diff))

enc_img = data[["enc_data","image_data"]]

print(data.iloc[1,2])
'''
i = 0
for d in wradar1:
    
    if (not isinstance(d,float)):
        truePath = "./" + d[17:]
        i += 1
        cmd = "scp " + truePath + " ./radar1"
        #print(cmd)
        os.system(cmd)
        #if (i>5):
        #    break
'''   

'''
radar_im = skimage.io.imread('imageData/XBR_1657868305.6786375.jpg')
enc_im = skimage.io.imread('enc_data/ENC_1657868305.6786375_maxrange1346.0_width_500.png')




with open('radar0/1657868305.567_group0_radar0.pkl','rb') as f:
    datas = pickle.load(f)
    print(datas)
    print(type(datas))
 

plt.figure()
plt.imshow(radar_im)
plt.scatter(radar_im.shape[0]/2,radar_im.shape[1]/2)


enc_im = skimage.io.imread('enc_data/ENC_1657864803.1399536_maxrange488.0_width_500.png')


plt.figure()
plt.imshow(enc_im)
plt.scatter(enc_im.shape[0]/2,enc_im.shape[1]/2)

radar_im = skimage.io.imread('imageData/XBR_1657864801.4728134.jpg')
plt.figure()
plt.imshow(radar_im)

radar_im = skimage.io.imread('imageData/XBR_1657864803.1399536.jpg')
plt.figure()
plt.imshow(radar_im)

radar_im = skimage.io.imread('imageData/XBR_1657864804.8184116.jpg')
plt.figure()
plt.imshow(radar_im)

radar_im = skimage.io.imread('imageData/XBR_1657864806.4949012.jpg')
plt.figure()
plt.imshow(radar_im)
'''


'''
plt.figure()
plt.subplot(1,2,1)
plt.subplot(1,2,2)
plt.ion()
plt.show()
for i in range(9867):
    plt.clf()
    d = data.iloc[i,2]    
    plt.subplot(1,2,1)
    curr_path = "enc_data" + d[16:]
    enc_im = skimage.io.imread(curr_path)
    
    plt.imshow(enc_im)
    #plt.draw()
    
    d = data.iloc[i,9]
    plt.subplot(1,2,2)
    curr_path = "imageData" + d[16:]
    radar_im = skimage.io.imread(curr_path)
    
    plt.imshow(radar_im)
    plt.draw()
    plt.pause(0.01)

'''


niceData = np.load("/home/christian/targetData/1657864801.4728134_targets.npy")
encData = skimage.io.imread("enc_data/ENC_1657864801.4728134_maxrange488.0_width_500.png")
radarData = skimage.io.imread("imageData/XBR_1657864801.4728134.jpg")

plt.figure()
plt.imshow(niceData[:,:,1])
plt.scatter(radarData.shape[0]/2,radarData.shape[1]/2)

plt.figure()
plt.imshow(niceData[:,:,0])
plt.scatter(radarData.shape[0]/2,radarData.shape[1]/2)
plt.figure()
plt.imshow(niceData[:,:,2])
plt.scatter(radarData.shape[0]/2,radarData.shape[1]/2)
plt.figure()
plt.imshow(niceData[:,:,3])
plt.scatter(radarData.shape[0]/2,radarData.shape[1]/2)
plt.figure()
plt.imshow(niceData[:,:,4])
plt.scatter(radarData.shape[0]/2,radarData.shape[1]/2)

plt.figure()
plt.imshow(encData)
plt.scatter(radarData.shape[0]/2,radarData.shape[1]/2)

plt.figure()
plt.imshow(radarData)
plt.scatter(radarData.shape[0]/2,radarData.shape[1]/2)

plt.show()
