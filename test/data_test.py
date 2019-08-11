import unittest
from src.utils.data import load_data_from_file
from src.utils.data import load_data_tag_from_file

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
        reviews_neg = load_data_from_file('neg')
        reviews_pos = load_data_from_file('pos')
        assert len(reviews_neg) == self.number_reviews
        assert len(reviews_pos) == self.number_reviews

    def test_read_data_tag_from_file(self):
        tags_neg = load_data_tag_from_file('neg')
        tags_pos = load_data_tag_from_file('pos')
        assert isinstance(tags_neg[0], dict)
        assert isinstance(tags_pos[0], dict)
        assert len(tags_neg) == self.number_reviews
        assert len(tags_pos) == self.number_reviews


if __name__ == "__main__":
    unittest.main()

'''
The .tag file is formatted with the Penn Treebank tagset
CC      Coordinating conjuction
CD      Cardinal number
DT      Determiner
EX      Existential there
FW      Foreign word
IN      preposition or subordinating conjuction
JJ      Adjective
JJR     Adjective, comparative
JJS     Adjective, superlative
LS      List item marker
MD      Modal
NN      Noun, singular or mass
NNS     Noun, plural
NNP     Proper noun, singular
NNPS    Proper noun, plural
PDT     Predeterminer
POS     Possessive ending
PRP     Personal pronoun
PRP$    Possessive pronoun
RB      Adverb
RBR     Adverb, comparative
RBS     Adverb, superlative
RP      Particle
SYM     Symbol
TO      to
UH      Interjection
VB      Verb, base form
VBD     Verb, past tense
VBG     Verb, gerund or present participle
VBN     Verb, past participle
VBP     Verb, non-3rd person singular present
VBZ     Verb, 3rd person singular present
WDT     Wh-determiner
WP      Wh-pronoun
WP$     Possessive wh-pronoun
WRB     Wh-adverb
ref:{https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html}
'''
