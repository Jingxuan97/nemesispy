import sys
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
import numpy as np
import matplotlib.pyplot as plt

from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,pvmap,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,hvmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,hvmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,vmrmap_mod_new)


"""
# plot TP profiles from all 64*32 = 2048 grid points
for ilon in range(nlon):
    for ilat in range(nlat):
        plt.plot(tmap_mod[ilon,ilat,:],pv/1e5,lw=0.1,color='#FF00FF')

# plt.ylim(1e-3,20)
plt.xlabel('Temperature [K]',size='x-large')
plt.ylabel('Pressure [bar]',size='x-large')
plt.semilogy()
plt.tick_params(length=10,width=1,labelsize='large',which='major')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('TP_all.pdf',format='pdf',dpi=400)
plt.show()
"""

# plot a few phases

# plot equitorial pressure against longitude contour plot

# plot averaged-over-lattitude pressure against longitude contour plot
