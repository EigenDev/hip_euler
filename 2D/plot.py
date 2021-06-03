import h5py
import numpy as np
import matplotlib.pyplot as plt


with h5py.File("sod_test2d.h5", "r+") as hf:
    rho = np.array(hf["rho"][:])
    vel = np.array(hf["v1"][:])
    pre = np.array(hf["pre"][:])

rho = rho.reshape(128, 128)
    
x = np.linspace(-1.0, 1.0, rho.shape[1])
y = np.linspace(-1.0, 1.0, rho.shape[0])

xx, yy = np.meshgrid(x, y)

plt.pcolormesh(xx, yy, rho, shading='auto')
#plt.plot(x, rho[0])
plt.show()