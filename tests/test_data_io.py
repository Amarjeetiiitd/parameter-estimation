import unittest
import pandas as pd
import numpy as np
from src import data_io

class TestDataIO(unittest.TestCase):
    def test_load_dataset(self):
        try:
            df = data_io.load_dataset("data/xy_data.csv")
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertIn("x", df.columns)
            self.assertIn("y", df.columns)
            self.assertEqual(len(df), 1500)
        except FileNotFoundError:
            self.skipTest("xy_data.csv not found (running in separate test context)")

if __name__ == '__main__':
    unittest.main()
