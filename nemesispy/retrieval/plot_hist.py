import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('./chains3/fix-post_equal_weights.dat',unpack=True)
# print(data)
data = data.T
vmr = data[:,-5:-1]

D = vmr

plt.style.use('_mpl-gallery')

fig, ax = plt.subplots()
print(data.shape)
print(vmr.shape)
# print(vmr)
position = [0,0.25,0.5,0.75]
vp = ax.violinplot(D, position, widths=0.1,
        showmeans=True, showmedians=False, showextrema=False)
for p in position:
    ax.axvline(p,color='black')
ax.set_ylabel(r'log $X$')
ax.set_xlabel('Orbital phase')
plt.tight_layout()
plt.show()