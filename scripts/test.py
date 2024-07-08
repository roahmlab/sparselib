import sparselib
import numpy as np

class SolverParams():
	max_level: int = 6
	dim: int = 1
	domain: np.ndarray = np.array(0, 2*np.pi)
	funcs: list = []

def dynamics(x):
    """
    Dynamics for x
    
    :param      x:    Coordinates
    :type       x:    np.array
    
    :returns:   vector field at coords
    :rtype:     np.array
    """
    return -np.sin(2 * x)

def uniform_uncertainty(x):
    """
    Initial Uniform uncertainty, independent of dimension
    
    :param      x:     Coordinates
    :type       x:     np.array
    :param      mu:    Optional mean
    :type       mu:    np.array

    :returns:   initial uncertainty at coords
    :rtype:     np.array
    """
    return np.sqrt(np.ones_like(x[:,0]) * 1 / (2 * np.pi))


params = SolverParams()
params.funcs = [uniform_uncertainty, dynamics]

container = sparselib.GridContainer(params)