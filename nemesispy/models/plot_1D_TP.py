# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from nemesispy.models.TP_profiles import TP_Guillot, TP_Line
fig, axs = plt.subplots(
    nrows=1,ncols=2,
    sharex=True,sharey=True,
    figsize=[10,4],
    dpi=600
    )
fig.supxlabel('Temperature [k]')
fig.supylabel('Pressure [bar]')

NLAYER = 100
P_grid = np.geomspace(10e8,1,NLAYER) # pressure in pa
T_eq = 1450
# k_IR = 1e-3
g = 50
# gamma = 4
T_int = 100
f = 0.25
c = ['k','r','b','y']
l = ['-','--',':']
ic = 0
il = 0
for k_IR in np.geomspace(1e-4,1e0,3):
    
    for gamma in np.geomspace(1e-3,1e1,3):
        # for T_int in np.linspace(10,3000,10):
        x = TP_Guillot(P=P_grid,g_plt=g,T_eq=T_eq,k_IR=k_IR,gamma=gamma,
                f=f,T_int=T_int)
        axs[0].plot(x,P_grid/1e5,linewidth=0.5,color = c[ic],
                    linestyle=l[il],
                    label=r'$\kappa_{th}$'+'={:.0e}\n'.format(k_IR)+r'$\gamma$'+'={:.0e}'.format(gamma))
        print(il)
        il = (il + 1)%3

        

    ic += 1
    # plt.plot(x,P_grid/1e5,linewidth=2,color='k')

axs[0].semilogy()
#axs[0].set_ylim(1e-3,1e2)
axs[0].invert_yaxis()
axs[0].set_xlim(1000,6000)
axs[0].set_title('2-stream')
axs[0].legend(ncol=3,fontsize='xx-small',loc='upper right')




k_IR = 1e-2
gamma1 = 1e-2
# gamma2 = 0.1
# alpha = 0.5
beta = 1
T_int = 100

c = ['k','r','b','y']
l = ['-','--',':']
ic = 0
il = 0

for alpha in [0.1,0.5,0.9]:
    for gamma2 in [1e-3,1,10]:
        x = TP_Line(P=P_grid,g_plt=g,T_eq=T_eq,k_IR=k_IR,gamma1=gamma1,gamma2=gamma2,
                alpha=alpha,beta=beta,T_int=T_int)
        axs[1].plot(x,P_grid/1e5,linewidth=0.5,color=c[ic],linestyle=l[il],
                    label=r'$\alpha$'+'={:.0e}'.format(alpha)+'\n'+'$\gamma_2$={:.0e}'.format(gamma2))
        il = (il + 1)%3
    ic += 1

axs[1].set_title('3-stream')
axs[1].legend(ncol=3,fontsize='xx-small',loc='upper right')
plt.savefig('1D_TP_profiles_Guillot.pdf',dpi=400)
