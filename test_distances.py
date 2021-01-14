import unittest
from distances import calc_wmd, calc_ict

class DistanceTestCase(unittest.TestCase):
    """ Test for 'distances.py'. """

    def test_wmd(self):
        """Is the wmd calculated correctly?"""
        model_embedding = {(1, 4): 1,
                           (5, 1): 1}
        real_embedding = {(5, 1): 1}
        self.assertAlmostEqual(calc_wmd(model_embedding, real_embedding), 2.5)

    def test_ict(self): 
        """Is the ict calculated correctly?"""
        model_embedding = {(1, 4): 1,
                           (5, 1): 1}
        real_embedding = {(5, 1): 1}
        self.assertAlmostEqual(calc_ict(model_embedding, real_embedding, 1), 2.5)

if __name__ == '__main__':
    unittest.main()