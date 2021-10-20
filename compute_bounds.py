import numpy as np
from scipy.special import erf

def compute_expecation_bound(gammas):
	""" computes the expectation bound for the provided gamma values """

	y_vals = np.zeros(gammas.shape)
	a_vals = np.zeros(gammas.shape)
	sigma = 1
	for i in range(len(gammas)):
		mu = gammas[i]
		yargs = np.linspace(0,16*80,8*16*3200)
		f_vals = mu*(erf((yargs-mu)/sigma)+1)-sigma*np.exp(-(yargs-mu)**2/sigma**2)/np.sqrt(np.pi)
		y_vals[i] = yargs[np.argmin(f_vals**2)]
		a_vals[i] = 0.5*(1+erf((y_vals[i]-mu)/sigma))

def compute_var_bound(gammas):
	""" computes the variance bound for the provided gamma values """

	return 1./(1 + gammas**2)