from spgrid import SparseGrid

class GridContainer():
	def __init__(self, params):
		# Sparse grid for uncertainty
		self.grids = [SparseGrid(params.domain, params.max_level, params.dim) for _ in range(params.dim + 1)]

	def fit(self):
		[grid.fit(f) for grid, f in zip(self.grids, params.funcs)]
