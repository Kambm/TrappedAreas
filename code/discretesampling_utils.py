import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, RK45, solve_ivp
from scipy.linalg import block_diag, eigh
from itertools import product
import time

class GPSampler():
	""" Gaussian process velocity sampling object """

	def __init__(self, x0=np.zeros(2),sigma=1,lambd=1,mu=np.zeros(2)):
		self.xvals = None
		self.vvals = None
		self.sigma = sigma
		self.lambd = lambd
		self.mu = mu

	def sample_v(self, x):
		if self.xvals is None:
			self.xvals = x[None,:]
		else:
			self.xvals = np.vstack([self.xvals,x[None,:]])

		t0 = time.time()
		C = flow_cov_gauss(self.xvals, sigma=self.sigma, lambd=self.lambd)
		t1 = time.time()
		if self.vvals is None:
			v = np.random.multivariate_normal(self.mu, C)
			self.vvals = v[None,:]
		else:
			v = gcond(self.vvals.flatten(), np.concatenate([self.mu for i in range(self.xvals.shape[0])]),C, singular=True)
			test = np.random.multivariate_normal(np.concatenate([self.mu for i in range(self.xvals.shape[0])]), C)
			self.vvals = np.vstack([self.vvals,v[None,:]])
		return v

def update_cov_gauss(xvals,sigma=1,lambd=1):
	"""
	Gets the cholesky factor U of the rank-2 update U @ U.T to the covariance matrix.
	"""

	C_new = flow_cov_gauss(xvals,sigma=sigma,lambd=lambd) #np.zeros((xvals.shape[0],xvals.shape[0]))
	C_new[:-2,:-2] = 0
	C_new[-2:,-2:] -= np.eye(2)

	U, L, V = np.linalg.svd(C_new, hermitian=True)

	return U[:,:4] @ np.diag(L[:4]), V[:4,:]

def phi_flow_cov_gauss(x, sigma=1, lambd=1, interleave=True):
	"""
	Takes in an array of 2d positions [[x1_0, x1_1], [x2_0, x2_1], ...] and returns a covariance matrix for [phi, v_0, v_1] at each position, where state vector is
	structured as [phi(x1), v_0(x1), v_1(x1), phi(x2), v_0(x2), v_1(x2), ...].

	The value of the covariance is taken to be the covariance associated with a gaussian streamfunction distributed with the covariance
	G(x,y) = lambd**2 sigma**2 exp(-|x-y|**2/(2*lambd**2)) / 2, ensuring that Var(|v|**2) = sigma**2. The two-point v covariance is then
	given by <v_i(x) v_j(y)> = (-1)**(i+j) partial_xi partial_yj G(x,y) which ensures that div(v) = 0, and the covariance <phi(x) v_i(y)>
	is (-1)**i partial_yi G(x,y).

	"""
	Q = np.exp(-((x[:,0]-x[:,0,None])**2 + (x[:,1]-x[:,1,None])**2)/(2*lambd**2))
	# phi-phi covariance
	P = lambd**2*sigma**2*Q/2

	# phi-velocity covariance
	P0 = Q*(sigma**2)*(x[:,1]-x[:,1,None])/2
	P1 = -Q*(sigma**2)*(x[:,0]-x[:,0,None])/2

	# velocity-velocity covariance
	C_00 = Q*(sigma**2)*(1 - (x[:,1]-x[:,1,None])**2/(lambd**2))/2
	C_11 = Q*(sigma**2)*(1 - (x[:,0]-x[:,0,None])**2/(lambd**2))/2
	C_01 = Q*(sigma**2)*(x[:,0]-x[:,0,None])*(x[:,1]-x[:,1,None])/(2*lambd**2)

	n = x.shape[0]
	C = np.zeros((3*n, 3*n))
	if interleave:
		C[::3,::3] = P
		C[::3,1::3] = P0.T
		C[::3,2::3] = P1.T
		C[1::3,::3] = P0
		C[2::3,::3] = P1
		C[1::3,1::3] = C_00
		C[2::3,2::3] = C_11
		C[1::3,2::3] = C_01
		C[2::3,1::3] = C_01
	else:
		C[:n,:n] = P
		C[:n,n:2*n] = P0.T
		C[:n,2*n:] = P1.T
		C[n:2*n, :n] = P0
		C[2*n:,:n] = P1
		C[n:2*n, n:2*n] = C_00
		C[2*n:, 2*n:] = C_11
		C[n:2*n,2*n:] = C_01
		C[2*n:,n:2*n] = C_01

	return C


def flow_cov_gauss(x, sigma=1, lambd=1, interleave=True):
	"""
	Takes in an array of 2d positions [[x1_0, x1_1], [x2_0, x2_1], ...] and returns a covariance matrix for [v_0, v_1] at each position, where state vector is
	structured as [v_0(x1), v_1(x1), v_0(x2), v_1(x2), ...].

	The value of the covariance is taken to be the covariance associated with a gaussian streamfunction distributed with the covariance
	G(x,y) = lambd**2 sigma**2 exp(-|x-y|**2/(2*lambd**2)) / 2, ensuring that Var(|v|**2) = sigma**2. The two-point v covariance is then
	given by <v_i(x) v_j(y)> = (-1)**(i+j) partial_xi partial_yj G(x,y) which ensures that div(v) = 0.

	"""
	# print(type(np))
	Q = np.exp(-((x[:,0]-x[:,0,None])**2 + (x[:,1]-x[:,1,None])**2)/(2*lambd**2))
	C_00 = Q*(sigma**2)*(1 - (x[:,1]-x[:,1,None])**2/(lambd**2))/2
	C_11 = Q*(sigma**2)*(1 - (x[:,0]-x[:,0,None])**2/(lambd**2))/2
	C_01 = Q*(sigma**2)*(x[:,0]-x[:,0,None])*(x[:,1]-x[:,1,None])/(2*lambd**2)

	n = x.shape[0]
	C = np.zeros((2*n, 2*n))
	if interleave:
		C[::2,::2] = C_00
		C[1::2,1::2] = C_11
		C[::2,1::2] = C_01
		C[1::2,::2] = C_01
	else:
		C[:n,:n] = C_00
		C[n:,n:] = C_11
		C[:n,n:] = C_01
		C[n:,:n] = C_01

	return C

def phi_cov_gauss(x, sigma=1, lambd=1):
	"""
	Takes in an array of 2d positions [[x1_0, x1_1], [x2_0, x2_1], ...] and returns a covariance matrix for [v_0, v_1] at each position, where state vector is
	structured as [v_0(x1), v_1(x1), v_0(x2), v_1(x2), ...].

	The value of the covariance is taken to be the covariance associated with a gaussian streamfunction distributed with the covariance
	G(x,y) = lambd**2 sigma**2 exp(-|x-y|**2/(2*lambd**2)) / 2, ensuring that Var(|v|**2) = sigma**2.

	"""

	return (lambd**2*sigma**2)*np.exp(-((x[:,0]-x[:,0,None])**2 + (x[:,1]-x[:,1,None])**2)/(2*lambd**2))/2

def gcond(cond_vals, mu, covar, singular=False):
	"""
	Given an n x n covariance matrix covar, an n-dim mean mu, and a list of m samples, samples the remaining n-m variables
	from the gaussian distribution defined by provided the mean and covariance matrix, conditioned on the previous samples.

	"""

	n = mu.shape[0]
	m = cond_vals.shape[0]
	invcov = np.linalg.pinv(covar) if singular else np.linalg.inv(covar)
	Axx = invcov[m-n:, m-n:]
	Acx = invcov[m-n:,:m]
	Sxx = np.linalg.pinv(Axx) if singular else np.linalg.inv(Axx)
	mu_2 = -Sxx @ Acx @ (cond_vals-mu[:m])

	return np.random.multivariate_normal(mu[m-n:]+mu_2, Sxx)


def generate_flowfield():
	x = np.linspace(0,15,num=150)
	y = np.linspace(0,15,num=150)
	X,Y = np.meshgrid(x,y)
	xvals = np.stack([X.flatten(), Y.flatten()],axis=1)

	C = phi_cov_gauss(xvals)
	phi_vals = np.random.multivariate_normal(np.zeros(xvals.shape[0]), C)
	plt.imshow(phi_vals.reshape(X.shape))
	plt.show()
	np.save('data/phi_vals.npy', phi_vals.reshape(X.shape))


if __name__ == "__main__":
	# generates data for figure 1
	generate_flowfield()
