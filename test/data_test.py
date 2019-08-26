import unittest
from src.utils.data import load_data


class TextReadTest(unittest.TestCase):
    def test_load_data(self):
        load_data('cam_data')
        load_data('imdb_data')
        load_data('twitter')
        load_data('douban')

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
