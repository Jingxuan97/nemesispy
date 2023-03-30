import unittest
import numpy as np

from nemesispy.common.get_gas_info import get_gas_name,get_gas_id

class TestGetGasName(unittest.TestCase):

    def test_invalid_gas_id(self):
        invalid_id = [-1,0,1444]
        for id in invalid_id:
            with self.assertRaises(Exception):
                name = get_gas_id(id)

class TestGetGasId(unittest.TestCase):
    def test_invalid_gas_name(self):
        invalid_name = ['h2o','CH11','ABC']
        for name in invalid_name:
            with self.assertRaises(Exception):
                id = get_gas_name(name)

if __name__ == "__main__":
    unittest.main()