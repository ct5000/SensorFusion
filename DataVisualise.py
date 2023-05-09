import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np



data = pd.read_csv("dataset.csv")
print(data.head(1))

print(data.info())

latitude = data["latitude"]
longitude = data["longitude"]
enc_data = data["enc_data"]
time = data["timestamp"]

time_np = np.array(time.array)
time_diff = time_np[1:] - time_np[:-1]
print(np.mean(time_diff))

'''
i = 0
for d in enc_data:
    truePath = "./ENC_imgs" + d[16:]
    i += 1
    cmd = "scp " + truePath + " ./enc_data"
    
    os.system(cmd)
'''
    

plt.plot(longitude,latitude)





plt.show()