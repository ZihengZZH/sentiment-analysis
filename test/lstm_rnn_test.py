import unittest
from src.utils.data import load_data
from src.embeddings.lstm_rnn import LSTMRNN


class LSTMRNNTest(unittest.TestCase):
    def test_build_model(self):
        X_train, y_train, X_test, _ = load_data('cam_data')
        lstm = LSTMRNN(X_train, X_test, 'cam_data')
        lstm.build_model()
        lstm.train_model(y_train)


if __name__ == '__main__':
    unittest.main()