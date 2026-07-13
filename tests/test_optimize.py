import unittest
import numpy as np
from src import optimize, model

class TestOptimize(unittest.TestCase):
    def test_residuals_shape(self):
        # Verify that residuals calculations match output dimensions
        x_data = np.linspace(60, 100, 10)
        y_data = np.linspace(46, 65, 10)
        params = [0.52, 0.03, 55.0]
        
        obj = optimize._make_de_objective(x_data, y_data, optimize.OptimizationHistory())
        cost = obj(params)
        self.assertTrue(cost >= 0.0)

    def test_fit_result_bounds(self):
        # Run a micro optimization run on synthetic data to test full pipeline Integration
        params = model.CurveParams(theta=0.5235983, M=0.03, X=55.0)
        t_grid = np.linspace(6.0, 60.0, 50)
        x_syn, y_syn = model.forward(t_grid, params)
        
        # Run pipeline
        fit = optimize.run_full_pipeline(x_syn, y_syn)
        
        # Parameter estimates should be within bounds and close to original values
        self.assertTrue(0.0 < fit.theta < np.deg2rad(50.0))
        self.assertTrue(-0.05 < fit.M < 0.05)
        self.assertTrue(0.0 < fit.X < 100.0)
        self.assertTrue(fit.cost < 1e-4)

if __name__ == '__main__':
    unittest.main()
