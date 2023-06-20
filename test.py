import math
import numpy as np
import skimage.io 
import matplotlib.pyplot as plt

data_ex = np.load("/home/christian/targetData/1657864801.4728134_targets.npy")
plt.figure()
plt.imshow(data_ex[:,:,3]+data_ex[:,:,1])
plt.show()

quit()

skimage.io.imsave("extractedBuoy.png",data_ex[:,:,1])

