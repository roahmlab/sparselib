Funcs:
	- Variables:
		- dynamics: list(function)
		- uncertainty: function

Domain:
	- Variables
		- max_level: (int)
		- domain: (Dx2 array)
		- dim: int

SparseGrid:
	- Methods:
		- init:
			- Inputs: domain, max level
			- Outputs: N/A
		- fit:
			- Inputs: func
			- Outputs: N/A
		- eval:
			- Inputs: coords (NxD array)
			- Outputs: vals (N array)

GridContainer:
	- Methods:
		- fit:
			- Inputs: Funcs class
			- Output: N/A

SolverParams:
	- max_level
	- dimension
	- domain
	- dynamics function(s)
	- uncertainty function
	- funcs: [uncertainty, dynamics]

SpectralDiscretization:
	- Methods:
		- init:
			- Inputs: SolverParams
			- Outputs: N/A
		- solve:
			- Inputs: time (float)
					  coords (NxD array)
			- Output: vals (NxD array)


solverParams --> SpectralDiscretization
SpectralDiscretization():
	1) Constructs GridContainer
	2) Calls gridContainer.fit()
SpectralDiscretization.solve(t, xs):
	1) Runs propagation script



FIXES:
	- Fix issue with basis_der always multiplying 1j * k[d] for each
		dim even though der is not over each dim
	- Similar fix for basis function