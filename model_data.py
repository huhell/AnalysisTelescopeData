import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyAstronomy.modelSuite import forTrans as ft
import mc3

# Relevant parameters
period = 2.204735471
epoch = 2454731.68039

# Load and read the data
df = pd.read_pickle('binned_data.pkl')
time = df.time.item() + epoch
flux = df.flux.item()
noise = df.noise.item()

# Define the model
# The parameters to estimate are:
# - Inclination: the inclination of the planetary orbital plane
# - up, um: some stellar parameters (I do not go in details on those here)
# - a_Rs: the ratio between the star-planet orbital distance and the stellar radius
# - Rp_Rs: the ratio between the planetary and stellar radii
def mymodel(params_to_fit, time, period, epoch):

	inclination, up, um, a_Rs, Rp_Rs = params_to_fit # parameters to estimate

	ma = ft.MandelAgolLC(orbit="circular", ld="quad")
	ma["b"] = 0.0
	ma["per"] = period
	ma["T0"] = epoch
	ma["i"] = inclination
	ma["linLimb"] = (up + um)/2.
	ma["quadLimb"] = (up - um)/2.
	ma["a"] = a_Rs
	ma["p"] = Rp_Rs

	return(ma.evaluate(time))


# Estimation of the best parameters using Bayesian statistics (Markov Chain Monte Carlo algorithm)

# Initial guess
params_init = np.array([86., .5, -.2, 4., .068])

# Bounds, priors, and stepsizes for each parameters
pmin = np.array([70., 0., -1., 3., .05])
pmax = np.array([90., 2., 1., 6., .09])
prior = np.array([0., 0., 0., 0., 0.])
priorlow = np.array([0., 0., 0., 0., 0.])
priorup = np.array([0., 0., 0., 0., 0.])
pstep = np.array([0.01, 1e-2, 1e-2, 1e-2, 1e-3])
pnames = np.array(['Inclination', 'u_p', 'u_m', 'a_Rs', 'Rp_Rs'])

# Choose the random-walk algorithm for the MCMC, and configuration
sampler = 'demc'

nsamples = 5e5
nchains = 6
ncpu = 6
burnin = 30000
thinning = 1

kickoff = 'normal'
hsize = 10
fgamma = 0.9

# Convergence
grtest = True # Gelman-Rubin test

# Outputs
savefile = 'DEMC_output.npz'
plots = False

# Run the MCMC
mc3_output = mc3.sample(data=flux, uncert=noise, func=mymodel, params=params_init, indparams=(time, period, epoch),
        wlike=False, pmin=pmin, pmax=pmax, pstep=pstep, pnames=pnames, fgamma=fgamma,
        prior=prior, priorlow=priorlow, priorup=priorup,
        sampler=sampler, nsamples=nsamples, nchains=nchains,
        ncpu=ncpu, burnin=burnin, thinning=thinning,
        hsize = hsize, kickoff=kickoff, grtest=grtest,
        plots=plots, savefile=savefile)


# Load best parameters
trace = np.load('DEMC_output.npz')
bestp = trace['bestp']
best_model = mymodel(bestp, time, period, epoch)

fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios':[3,1]})
ax[0].plot(time-epoch, flux, 'b.', label='data')
ax[0].plot(time-epoch, best_model, 'r', label='best fit')
ax[0].legend()
ax[0].set_ylabel('flux')
ax[0].tick_params(labelbottom=False)
ax[1].plot(time-epoch, (flux - best_model)*1e6, 'k.', label='residuals')
ax[1].legend()
ax[1].set_ylabel('residuals')
ax[1].set_xlabel('time')

fig.savefig('bestfit.png')
plt.show()


# Save parameter estimates
z = trace['posterior']
stat = open('parameter_estimates.txt', 'w')
stat.write('Best-fit parameters:' + os.linesep)
for ip in range(len(pnames)):
	stat.write(pnames[ip] + ' =  %.4f' % bestp[ip] + os.linesep)
stat.write('-----------------------------------------------------------------------------------------------' + os.linesep)
stat.write('Parameter estimates:' + os.linesep)
for ip in range(len(pnames)):
	stat.write(pnames[ip] + ' = %.4f' % np.quantile(z[:,ip], .5) + ' + %.4f' % (np.quantile(z[:,ip], .84) - np.quantile(z[:,ip], .5)) + 
                ' - %.4f' % (np.quantile(z[:,ip], .5) - np.quantile(z[:,ip], .16)) + os.linesep)
