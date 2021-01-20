import os
from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Relevant parameters
E0 = 2454731.68039
P = 2.204735471
a = 0.0379 * 149597870.7
Rp = 1.431 * 71492.0
Rs = 2. * 696342.0
inc = 81.1
b = a/Rs * np.cos(np.radians(inc))
Td = P/np.pi * np.arcsin( np.sqrt( (Rs + Rp)**2 - b**2)/a )

# Relevant arrays
flux = []
time = []
noise = []

ff = []
tt = []
nn = []

# Go over files
files = os.listdir('./data/')
files = [filename for filename in files if filename.endswith('.fits')]

for filename in files:

	hdul = fits.open('./data/'+filename)
	data = hdul[1].data
	time = np.concatenate((time, data.field('TIME') + 2454833.0))
	flux = np.concatenate((flux, data.field('PDCSAP_FLUX')))
	noise = np.concatenate((noise, data.field('PDCSAP_FLUX_ERR')))

# Clean NaN flux values
are_finite = np.squeeze( np.argwhere( np.isnan(flux) == False ))
flux = flux[are_finite]
time = time[are_finite]
noise = noise[are_finite]

# Raw data
#plt.plot(time, flux, 'b.')
#plt.show()



# Average around mean time and normalize
N = np.arange(1, 1000, 1)
flag = np.zeros((N.size))
for ii in N:
	Tmid = E0 + ii*P
	isthere = np.squeeze(np.argwhere( (time > Tmid - 3*Td) & (time < Tmid + 3*Td) ))
	if isthere.size>0:
		flag[ii-1] = 1

tr = np.squeeze(np.argwhere(flag==1)) + 1
for it in tr:
	Tmid = E0 + it*P
	idx = np.squeeze(np.argwhere( (time > Tmid - 3*Td) & (time < Tmid + 3*Td) ))

	ff = np.concatenate((ff, flux[idx]/np.median(flux[idx])))
	tt = np.concatenate((tt, time[idx] - Tmid))
	nn = np.concatenate((nn, noise[idx]))

# Averaged and normalized data
#plt.plot(tt, ff, 'b.')
#plt.show()



# Bin the data set in 2min
tmin = np.min(tt)
tmax = np.max(tt)
fac = 2. / (24. * 60.)

nbins = np.int( np.floor( (tmax - tmin)/fac ))
ff_bin = np.zeros((nbins))
tt_bin = np.zeros((nbins))
nn_bin = np.zeros((nbins))
for i in range(nbins):
	ins = np.squeeze( np.argwhere( (tt >= tmin + i*fac) & (tt < tmin +(i+1)*fac) ))
	tt_bin[i] = tmin + i*fac
	ff_bin[i] = np.median(ff[ins])
	if ins.size == 1:
		nn_bin[i] = 0.0
	else:
		nn_bin[i] = np.std(ff[ins])/np.sqrt(ins.size)

# Binned data
#plt.plot(tt_bin, ff_bin, 'b.')
#plt.show()


# Remove the remaining baseline
baseline = np.squeeze(np.argwhere( (tt_bin < -Td/2.0) | (tt_bin > Td/2.0) ))
try:
    detf = np.polyfit(tt_bin[baseline], ff_bin[baseline], 2)
    det_func = detf[0]*tt_bin*tt_bin + detf[1]*tt_bin + detf[2]
    ff_detrend = ff_bin / det_func
except:
    print('Could not fit baseline')

# Save detrended data in a DataFrane
df = pd.DataFrame({'time':[tt_bin], 'flux':[ff_detrend], 'noise':[nn_bin]})
df.to_pickle('binned_data.pkl')

# Save plots
fig, ax = plt.subplots(1, 2, figsize=(16,6))
ax[0].plot(time, flux, 'k.', label='Raw data')
ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)
ax[0].legend()
ax[1].plot(tt_bin, ff_detrend, 'b.', label='Clean data')
ax[1].set_xlabel('time')
ax[1].set_ylabel('flux')
ax[1].legend()

fig.savefig('Raw2Clean_Data.png')
#plt.show()
