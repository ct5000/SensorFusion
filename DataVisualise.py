import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np



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

wradar0 = data["wradar_group0_id0"] 
wradar1 = data["wradar_group0_id1"] 
print(type(wradar0))

time_np = np.array(time.array)
time_diff = time_np[1:] - time_np[:-1]
print(np.mean(time_diff))


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
        
        

    

plt.plot(longitude,latitude)





plt.show()