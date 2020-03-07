import unittest
from src.embeddings.pretrain import loadPretrain


class loadPretrainTest(unittest.TestCase):
    def test_load_embeddings(self):
        loader = loadPretrain()
        loader.load_embeddings()


if __name__ == '__main__':
    unittest.main()