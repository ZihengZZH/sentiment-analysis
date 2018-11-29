import unittest
import random
import src.sign_test as sign

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.sign_test_test
'''

class SignTestTest(unittest.TestCase):
    # method called to prepare the test fixture
    def setUp(self):
        pass
    
    # method called immediately after the test method has been called and the result recorded
    def tearDown(self):
        pass

    def test_get_p_value(self):
        # assert sign.get_p_value(50, 50, 100) < 0.0001
        assert sign.get_p_value(3, 11, 0) == .029
        

if __name__ == "__main__":
    unittest.main()