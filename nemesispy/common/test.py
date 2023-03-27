import numpy as np
from nemesispy.common.calc_trig_fast import disc_weights_2tp
import matplotlib.pyplot as plt

phase = 45
nmu = 2
daymin = -90
daymax = 90

dtr = np.pi/180
import time
s = time.time()
output_day_zen,output_day_wt,output_night_zen,output_night_wt,\
    output_day_lat,output_day_lon,output_night_lat,output_night_lon\
    = disc_weights_2tp(phase, nmu, daymin, daymax)
e = time.time()
print('day')
print(np.cos(dtr*np.array(output_day_zen)))
print(output_day_wt)
print(sum(output_day_wt))

print('night')
print(np.cos(dtr*np.array(output_night_zen)))
print(output_night_wt)
print(sum(output_night_wt))

plt.scatter(output_day_lon,output_day_lat,color='red')
plt.scatter(output_night_lon,output_night_lat,color='blue')
plt.xlim(0,360)
plt.ylim(0,90)
plt.savefig('test')
s = time.time()
output_day_zen,output_day_wt,output_night_zen,output_night_wt,\
    output_day_lat,output_day_lon,output_night_lat,output_night_lon\
    = disc_weights_2tp(phase, nmu, daymin, daymax)
e = time.time()
print('time = ', e-s)