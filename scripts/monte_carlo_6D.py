import sparselib
import numpy as np
import matplotlib
import s3dlib.surface as s3d
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import stats
from scipy import interpolate
from matplotlib import cm, colorbar
from scipy.integrate import odeint
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from matplotlib.ticker import LinearLocator


omega = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]) + \
            np.random.normal(scale=0.01, size=(6,))
K = np.array([[1.0, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.1, 1.0, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.1, 1.0, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 1.0, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 1.0, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.1, 1.0]])

def normalize(ys, N):
    return np.squeeze(ys / (2 * np.pi * np.sum(ys) / N))

def dynamics1(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return omega[0] + K[0,1] * np.sin(x[1] - x[0]) \
                    + K[0,2] * np.sin(x[2] - x[0]) \
                    + K[0,3] * np.sin(x[3] - x[0]) \
                    + K[0,4] * np.sin(x[4] - x[0]) \
                    + K[0,5] * np.sin(x[5] - x[0])

def dynamics2(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return omega[1] + K[1,0] * np.sin(x[0] - x[1]) \
                    + K[1,2] * np.sin(x[2] - x[1]) \
                    + K[1,3] * np.sin(x[3] - x[1]) \
                    + K[1,4] * np.sin(x[4] - x[1]) \
                    + K[1,5] * np.sin(x[5] - x[1])

def dynamics3(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return omega[2] + K[2,0] * np.sin(x[0] - x[2]) \
                    + K[2,1] * np.sin(x[1] - x[2]) \
                    + K[2,3] * np.sin(x[3] - x[2]) \
                    + K[2,4] * np.sin(x[4] - x[2]) \
                    + K[2,5] * np.sin(x[5] - x[2])

def dynamics4(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return omega[3] + K[3,0] * np.sin(x[0] - x[3]) \
                    + K[3,1] * np.sin(x[1] - x[3]) \
                    + K[3,2] * np.sin(x[2] - x[3]) \
                    + K[3,4] * np.sin(x[4] - x[3]) \
                    + K[3,5] * np.sin(x[5] - x[3])

def dynamics5(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return omega[4] + K[4,0] * np.sin(x[0] - x[4]) \
                    + K[4,1] * np.sin(x[1] - x[4]) \
                    + K[4,2] * np.sin(x[2] - x[4]) \
                    + K[4,3] * np.sin(x[3] - x[4]) \
                    + K[4,5] * np.sin(x[5] - x[4])

def dynamics6(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return omega[5] + K[5,0] * np.sin(x[0] - x[5]) \
                    + K[5,1] * np.sin(x[1] - x[5]) \
                    + K[5,2] * np.sin(x[2] - x[5]) \
                    + K[5,3] * np.sin(x[3] - x[5]) \
                    + K[5,4] * np.sin(x[4] - x[5])

def func(x, t):
    return [dynamics1(x), dynamics2(x), dynamics3(x),
            dynamics4(x), dynamics5(x), dynamics6(x)]

def monte_carlo_sample(N):
    # mu = np.array([0, 0, 0, 0, 0, 0])
    # mu = np.linspace(0, 2*np.pi, 6)
    mu = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
    # cov = np.sqrt(np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3]))
    cov = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    samples = np.random.normal(loc=mu, scale=cov, size=(N,6))
    return samples % (2 * np.pi)

def kernel_estimate(samples):
    """
    Plots the gaussian_kde estimate given samples
    """ 
    kde = stats.gaussian_kde(samples.T)
    coords = np.linspace(0, 2*np.pi, 100)
    z = kde(np.transpose(coords))
    return normalize(z, 100)

def propagate(samples, dt):
    t = np.arange(0, dt, dt/10)
    prop_samples = np.zeros((samples.shape[0], samples.shape[1]))

    for i in range(samples.shape[0]):
        vals = odeint(func, samples[i,:], t)
        prop_samples[i,:] = vals[-1,:]

    return prop_samples % (2 * np.pi)

# Propoagation time parameters
total_time = 0.5
M = 10
t = 0
dt = total_time / M
ts = np.linspace(0, total_time, M+1)

# Sample using Monte Carlo
samples = monte_carlo_sample(N=100000)

# Results
t1 = kernel_estimate(samples[:,0])
t2 = kernel_estimate(samples[:,1])
t3 = kernel_estimate(samples[:,2])
t4 = kernel_estimate(samples[:,3])
t5 = kernel_estimate(samples[:,4])
t6 = kernel_estimate(samples[:,5])

N = 100
x = np.linspace(0, 2*np.pi, N)
xs, ys = np.meshgrid(x, ts)

print("Computing coordinates wrt time ...")
pbar = tqdm(total=M)
for i in range(M):
    t += dt
    samples = propagate(samples, dt)
    
    # Marginalize along each dimension
    t1s = kernel_estimate(samples[:,0])
    t2s = kernel_estimate(samples[:,1])
    t3s = kernel_estimate(samples[:,2])
    t4s = kernel_estimate(samples[:,3])
    t5s = kernel_estimate(samples[:,4])
    t6s = kernel_estimate(samples[:,5])

    t1 = np.vstack((t1, t1s))
    t2 = np.vstack((t2, t2s))
    t3 = np.vstack((t3, t3s))
    t4 = np.vstack((t4, t4s))
    t5 = np.vstack((t5, t5s))
    t6 = np.vstack((t6, t6s))
    pbar.update(1)
pbar.close()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_xlabel('Frequency')
ax1.set_ylabel('Time [s]')
ax1.set_zlabel('Probability Density')
ax1.set_title('Harmonic Oscillation Frequency vs Time of First Mode')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Time [s]')
ax2.set_zlabel('Probability Density')
ax2.set_title('Harmonic Oscillation Frequency vs Time of Second Mode')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.set_xlabel('Frequency')
ax3.set_ylabel('Time [s]')
ax3.set_zlabel('Probability Density')
ax3.set_title('Harmonic Oscillation Frequency vs Time of Third Mode')

fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.set_xlabel('Frequency')
ax4.set_ylabel('Time [s]')
ax4.set_zlabel('Probability Density')
ax4.set_title('Harmonic Oscillation Frequency vs Time of Fourth Mode')

fig5 = plt.figure()
ax5 = fig5.add_subplot(111, projection='3d')
ax5.set_xlabel('Frequency')
ax5.set_ylabel('Time [s]')
ax5.set_zlabel('Probability Density')
ax5.set_title('Harmonic Oscillation Frequency vs Time of Fifth Mode')

fig6 = plt.figure()
ax6 = fig6.add_subplot(111, projection='3d')
ax6.set_xlabel('Frequency')
ax6.set_ylabel('Time [s]')
ax6.set_zlabel('Probability Density')
ax6.set_title('Harmonic Oscillation Frequency vs Time of Sixth Mode')

xnew, ynew = np.mgrid[0:2*np.pi:100j, 0:total_time:100j]
tck = interpolate.bisplrep(xs, ys, t1, s=1)
t1 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t2, s=1)
t2 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t3, s=1)
t3 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t4, s=1)
t4 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t5, s=1)
t5 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

tck = interpolate.bisplrep(xs, ys, t6, s=1)
t6 = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

# ax1.plot_surface(xnew, ynew, t1.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
# ax2.plot_surface(xnew, ynew, t2.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
# ax3.plot_surface(xnew, ynew, t3.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
# ax4.plot_surface(xnew, ynew, t4.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
# ax5.plot_surface(xnew, ynew, t5.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
# ax6.plot_surface(xnew, ynew, t6.clip(min=0), rstride=1, cstride=1, alpha=None, antialiased=True)
# plt.show()

np.savez('t1_monte_carlo.npz', x=xnew, y=ynew, t1=t1)
