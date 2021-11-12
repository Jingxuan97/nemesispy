import numpy as np
import os
import matplotlib.pyplot as plt
from nemesispy.radtran.readk import read_kta, read_kls
# a = np.arange(1,10)
# b = np.arange(1,10)**2
# plt.plot(a,b)
# plt.show()
from pathlib import Path
main_path = Path(__file__).parents[1]
print(main_path)

lowres_files = ['{}/data/ktables/h2o'.format(main_path),
         '{}/data/ktables/co2'.format(main_path),
         '{}/data/ktables/co'.format(main_path),
         '{}/data/ktables/ch4'.format(main_path)]

aeriel_files = ['{}/data/ktables/H2O_Katy_ARIEL_test'.format(main_path),
          '{}/data/ktables/CO2_Katy_ARIEL_test'.format(main_path),
          '{}/data/ktables/CO_Katy_ARIEL_test'.format(main_path),
          '{}/data/ktables/CH4_Katy_ARIEL_test'.format(main_path)]

hires_files = ['{}/data/ktables/H2O_Katy_R1000'.format(main_path),
         '{}/data/ktables/CO2_Katy_R1000'.format(main_path),
          '{}/data/ktables/CO_Katy_R1000'.format(main_path),
          '{}/data/ktables/CH4_Katy_R1000'.format(main_path)]


gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid, k_w_g_p_t\
    = read_kta(lowres_files[0])

print(gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid, k_w_g_p_t)

x =  np.array([0.003436, 0.018014, 0.043883, 0.080442, 0.126834, 0.181973,
       0.244567, 0.313147, 0.386107, 0.461737, 0.538263, 0.613893,
       0.686853, 0.755433, 0.818027, 0.873166, 0.919558, 0.956117,
       0.981986, 0.996564], dtype="float32")
y =  np.array([0.003436, 0.018014, 0.043883, 0.080442, 0.126834, 0.181973,
       0.244567, 0.313147, 0.386107, 0.461737, 0.538263, 0.613893,
       0.686853, 0.755433, 0.818027, 0.873166, 0.919558, 0.956117,
       0.981986, 0.996564], dtype="float32")