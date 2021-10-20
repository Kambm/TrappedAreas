import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	# directory of base flow
	phi = np.load('data/phi_vals.npy')
	vy,vx = np.gradient(phi)
	sigma = np.sqrt(np.mean(vy**2+vx**2))

	X,Y = np.meshgrid(np.linspace(0,1,num=phi.shape[0]),np.linspace(0,1,num=phi.shape[0]))
	dx = 1/(phi.shape[0]-1)
	fig, ax = plt.subplots()

	mu = 0.1*sigma
	q = ax.contour(phi+mu*Y/dx, levels=35)
	fig2, ax2 = plt.subplots(2,1)
	for item in q.collections:
		for line in item.get_segments():
			if np.max(line[:,0]) >= 98 and np.min(line[:,0]) <= 0:
				ax2[0].plot(line[:,0], line[:,1], color='lightblue', linewidth=1, zorder=0)
			else:
				ax2[0].plot(line[:,0], line[:,1], color='green', linewidth=1, zorder=0)
	vy, vx = np.gradient(phi+mu*Y/dx)
	Xl, Yl = np.meshgrid(np.arange(100),np.arange(100))
	ax2[0].quiver(Xl[::4,::4], Yl[::4,::4],-vy[::4,::4],vx[::4,::4],color='black', zorder=1)
	ax2[0].axis('off')
	ax2[0].set_ylim(Xl[0,Xl.shape[1]//4],Xl[0,3*Xl.shape[1]//4])

	mu = 0.7*sigma
	q = ax.contour(phi+mu*Y/dx, levels=90)
	for item in q.collections:
		for line in item.get_segments():
			if np.max(line[:,0]) >= 98 and np.min(line[:,0]) <= 0:
				ax2[1].plot(line[:,0], line[:,1], color='lightblue', linewidth=1, zorder=0)
			else:
				ax2[1].plot(line[:,0], line[:,1], color='green', linewidth=1, zorder=0)
	vy, vx = np.gradient(phi+mu*Y/dx)
	Xl, Yl = np.meshgrid(np.arange(100),np.arange(100))
	ax2[1].quiver(Xl[::4,::4], Yl[::4,::4],-vy[::4,::4],vx[::4,::4],color='black', zorder=1)
	ax2[1].axis('off')
	ax2[1].set_ylim(Xl[0,Xl.shape[1]//4],Xl[0,3*Xl.shape[1]//4])

	plt.close(fig)
	plt.show()