import h5py
import numpy as np
import matplotlib.pyplot as plt


with h5py.File("sod_test2d.h5", "r+") as hf:
    rho = np.array(hf["rho"][:])
    vel = np.array(hf["v1"][:])
    pre = np.array(hf["pre"][:])
    sim_info  = hf['sim_info']
    nx        = sim_info.attrs['nx']
    ny        = sim_info.attrs['ny']

rho = rho.reshape(nx, ny)
    
x = np.linspace(-1.0, 1.0, rho.shape[1])
y = np.linspace(-1.0, 1.0, rho.shape[0])

xx, yy = np.meshgrid(x, y)

plt.pcolormesh(xx, yy, rho, shading='auto')
plt.savefig('sod.png')
plt.show()