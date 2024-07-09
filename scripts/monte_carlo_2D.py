import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import stats
from matplotlib import cm
from scipy.integrate import odeint
from scipy.interpolate import griddata


def monte_carlo_sample(N):
	t = np.arange(0, 1.01, 0.01)
	mu = np.array([np.pi, np.pi])
	cov = np.array([0.2, 0.3])
	samples = np.random.normal(loc=mu, scale=cov, size=(N,2))

	plot_kernel_estimate(samples)

	prop_samples = np.zeros((N, samples.shape[1]))

	pbar = tqdm(total=N)
	for i in range(samples.shape[0]):
		vals = odeint(func, samples[i,:], t)
		prop_samples[i,:] = vals[-1,:]
		pbar.update(1) 
	pbar.close()

	return prop_samples

def func(x, t):
	return [dynamics1(x), dynamics2(x)]

def dynamics1(x):
	return 1.0 * np.sin(x[1]) + \
			0.2 * np.cos(x[0])

def dynamics2(x):
	return 0.5 * np.sin(x[0]) + \
			1.0 * np.cos(x[1])

def plot_kernel_estimate(samples):
	"""
	Plots the gaussian_kde estimate given samples
	"""	
	kde = stats.gaussian_kde(samples.T)

	nLevel = [np.linspace(0, 2*np.pi, 100),np.linspace(0, 2*np.pi, 100)]
	coordinates = np.array(np.meshgrid(*nLevel)).T.reshape(-1, 2)
	z = kde(np.transpose(coordinates))

	dx = 2 * np.pi / 1024
	xq,yq = np.meshgrid(np.arange(0,2*np.pi,dx),
						np.arange(0,2*np.pi,dx))
	zq = griddata(coordinates, z, (xq, yq), method='cubic')

	inferno = plt.get_cmap('inferno')
	vals = inferno(zq)[:,:,:]
	fig = plt.figure(frameon=False)
	plt.axis('off')
	plt.imshow(vals, interpolation='nearest')
	plt.show()


particles = monte_carlo_sample(N=5000)
plot_kernel_estimate(particles)