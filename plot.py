import h5py
import numpy as np
import matplotlib.pyplot as plt


with h5py.File("sod_test.h5", "r+") as hf:
    rho = np.array(hf["rho"][:])
    vel = np.array(hf["vel"][:])
    pre = np.array(hf["pre"][:])
    
x = np.linspace(0.0, 1.0, rho.size)
plt.plot(x, rho)
plt.show()