import unittest
import numpy as np
from src import model

class TestModel(unittest.TestCase):
    def setUp(self):
        self.params = model.CurveParams(
            theta=0.5235983,  # ~30 degrees
            M=0.03,
            X=55.0
        )

    def test_forward_shapes(self):
        x_val, y_val = model.forward(6.0, self.params)
        self.assertTrue(isinstance(x_val, (float, np.float64)))
        self.assertTrue(isinstance(y_val, (float, np.float64)))

        t_arr = np.linspace(6.0, 60.0, 100)
        x_arr, y_arr = model.forward(t_arr, self.params)
        self.assertEqual(len(x_arr), 100)
        self.assertEqual(len(y_arr), 100)

    def test_inverse_rotate_round_trip(self):
        t_val = 15.0
        x_pt = t_val * np.cos(self.params.theta) + self.params.X
        y_pt = 42.0 + t_val * np.sin(self.params.theta)

        u_rec, v_rec = model.inverse_rotate(np.array([x_pt]), np.array([y_pt]), self.params.theta, self.params.X)
        self.assertAlmostEqual(u_rec[0], t_val, places=6)
        self.assertAlmostEqual(v_rec[0], 0.0, places=6)

    def test_recovered_t_large_value(self):
        t_large = 100.0
        x_pt = t_large * np.cos(self.params.theta) + self.params.X
        y_pt = 42.0 + t_large * np.sin(self.params.theta)
        
        t_rec = model.recovered_t(np.array([x_pt]), np.array([y_pt]), self.params.theta, self.params.X)
        self.assertAlmostEqual(t_rec[0], t_large, places=6)

if __name__ == '__main__':
    unittest.main()
