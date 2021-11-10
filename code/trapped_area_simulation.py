import numpy as np
from scipy.integrate import odeint
import time
import matplotlib.pyplot as plt
from itertools import product
import multiprocessing
import os

def forward(x, phases, jvals, kvals, factors, mu=0):
	"""
	Computes velocity at a given position given a specification of the streamfunction in terms of wave vectors, phases and factors.

	Args:
	jvals : 2D array with x-component of wave vectors associated with the streamfunction
	kvals : 2D array with y-component of wave vectors associated with the streamfunction
	phases : phases for each fourier mode
	factors: 2D array with coefficient for each fourier component of the streamfunction
	mu : scalar value of mean velocity magnitude

	"""
	covals = [-factors*(2*np.pi*kvals)*np.sin(2*np.pi*(jvals*x[0]+kvals*x[1])+phases), factors*(2*np.pi*jvals)*np.sin(2*np.pi*(jvals*x[0]+kvals*x[1])+phases)]
	return np.sum(covals, axis=(1,2))+mu*np.asarray([1,0])

def generate_trajectory(arg_tuple, tmax=600):
	"""
	Given an argument tuple, generates a streamfunction satisfying specifications and integrates a trajectory
	starting at the origin over a time tmax. 

	arg_tuple values:
	jvals : 2D array with x-component of wave vectors associated with the streamfunction
	kvals : 2D array with y-component of wave vectors associated with the streamfunction
	factors : 2D array with coefficient for each fourier component of the streamfunction 
	mu : scalar value of mean velocity magnitude
	sigma : scalar value of sqrt(<v_x^2+v_y^2>) for the velocity ensemble
	seed : random seed
	tmax : maximum simulation time

	"""
	jvals,kvals,factors,mu,sigma,seed = arg_tuple
	x0 = np.zeros(2)
	np.random.seed(seed)
	phases = 2*np.pi*np.random.rand(*factors.shape)
	f = lambda x,t: 2*forward(x,phases,jvals,kvals,factors,mu=mu)/sigma
	t = np.linspace(0,100,tmax)
	return odeint(f,x0,t)

def simulate(nphases=151, Lmax=5, klambd=10, runs=1000, gammas=np.logspace(3,-1,base=0.1), dir_name='../data/yvals'):
	"""
	This function runs a sampling program for each value of gamma in a range. For each value of gamma a number
	of independent trajectories are simulated from streamfunctions drawn from a random-phase approximation of
	an RBF ensemble. The trajectories are saved in files of the form dir_name/yvals%i/yvals%j.npy for the ith
	gamma value and jth experimental run.

	Args:
	----
	nphases : number of Fourier components in each dimension.
	Lmax : maximum Fourier wavelength scale parameter  
	klambd : correlation length for the RBF power spectal density
	runs : number of independent trajectories simulated for each gamma value
	gammas : array of values of gamma that area simulated
	dir_name : name of top-level directory for storing experimental runs.

	"""

	jvals, kvals = np.meshgrid(np.arange(-(nphases//2), nphases//2+1)/Lmax, np.arange(-(nphases//2), nphases//2+1)/Lmax)
	factors = np.exp(-(jvals**2+kvals**2)/(klambd**2))*((kvals**2 + jvals**2) <= np.max(kvals)**2)*(kvals**2+jvals**2>0)
	sigma = np.sqrt(np.sum((factors*(2*np.pi*kvals))**2))

	for i in range(len(gammas)):
		mu = gammas[i]*sigma
		pool = multiprocessing.Pool(None)
		yvals_multi = pool.map(generate_trajectory, [(jvals,kvals,factors,mu,sigma,j) for j in range(runs)])
		
		# save data
		if not os.path.exists(dir_name + '/yvals%d' %i):
				os.makedirs(dir_name + '/yvals%d' %i)
		for j in range(len(yvals_multi)):
			np.save(dir_name + '/yvals%d/yvals%d.npy' %(i,j), yvals_multi[j])

def min_dist(yvals):
	"""
	Helper function for compute_areas. Takes a trajectory yvals and outputs the minimum distance
	of any point in the second half of the trajectory to first point.

	"""
	time_dists = np.linalg.norm(yvals, axis=-1)
	min_dist = np.min(time_dists[:,time_dists.shape[1]//2:],axis=-1)
	return min_dist


def compute_areas(dirname='../data/yvals', data_fname='new_data' runs=1000, gammas=np.logspace(3,-1,base=0.1)):
	"""
	Computes trapped area estimate from trajectory data using a heuristic clasiification of trapped and nontrapped trajectories.
	The heuristic is based on discontinuities in the observed displacements for each trajectory.

	Args:
	----
	dirname : top-level directory with stored trajectory data. The jth run for the ith gamma value should be stored as
		dirname/yvals%i/yvals%j.npy
	data_fname : filename to store trapped area data
	runs : number of runs per gamma value
	gammas : gamma values simulated

	"""
	for i in range(len(gammas)):
		yval_list.append(np.stack([np.load(dirname + '/yvals%d/yvals%d.npy' %(i,j)) for j in range(runs)],axis=0))

	dists_list = [np.linalg.norm(yval[:,-1,:]-yval[:,0,:],axis=-1) for yval in yval_list] # computes distance to origin of each point on trajectory
	min_dists = [min_dist(yval) for yval in yval_list] # minimum distance out of points in second half o trajectory
	my_sorts = [np.sort(dist) for dist in min_dists]
	my_args = [np.argsort(dist) for dist in min_dists]
	# computes trapped area threshold by looking for biggest discontinuity in cumulative distribution of minimum displacements. This quantity exhibits
	# boundary effects for certain values of gamma so the tails of the distribution are cut off at one end or the other depending on whether gamma > 0.05
	thresh_vals = np.asarray([(np.argmax(my_sorts[i][15:]/my_sorts[i][14:-1])+15)/len(my_sorts[i]) if gammas[i] <= 0.05 else (np.argmax(my_sorts[i][1:-30]/my_sorts[i][:-31]))/len(my_sorts[i]) for i in range(len(my_sorts))])
	errbars = np.sqrt(thresh_vals*(1 - thresh_vals)/1000) # bernoulli error for trapped area
	np.save(data_fname, np.concatenate([gammas, thresh_vals, errbars],axis=0))

if __name__ == "__main__":
	"""
	Runs a full simulation and computes the trapped area
	"""
	simulate()
	compute_areas()

