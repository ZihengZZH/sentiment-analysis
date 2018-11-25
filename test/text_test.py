import unittest
from src.text import read_data_from_file
from src.text import read_tag_from_file

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.text_test
'''

class TextReadTest(unittest.TestCase):
    # method called to prepare the test fixture
    def setUp(self):
        self.number_reviews = 1000
    # method called immediately after the test method has been called and the result recorded
    def tearDown(self):
        pass

    def test_read_data_from_file(self):
        reviews_neg = read_data_from_file('neg')
        reviews_pos = read_data_from_file('pos')
        assert len(reviews_neg) == self.number_reviews
        assert len(reviews_pos) == self.number_reviews
        
    def test_rea_tag_from_file(self):
        tags_neg = read_tag_from_file('neg')
        tags_pos = read_tag_from_file('pos')
        assert len(tags_neg) == self.number_reviews
        assert len(tags_pos) == self.number_reviews

if __name__ == "__main__":
    unittest.main()