import numpy as np
import time
from nemesispy.radtran.calc_mmw import calc_mmw

NITER = 1000000

ID = np.array([1,2,3,4,5,6])
VMR = np.array([0.5,0.1,0.1,0.1,0.1,0.1])
ISO = np.array([0,0,0,0])

start = time.time()
for i in range(NITER):
    mmw = calc_mmw(ID, VMR)
end = time.time()

print('runtime=',(end-start)/NITER)
