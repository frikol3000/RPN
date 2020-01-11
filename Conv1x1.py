import numpy as np

class Conv1x1:
    def __init__(self, in_shape, out_shape):
        self.inp = in_shape
        self.out = out_shape
        self.filters = np.random.uniform(size=(self.out, self.inp, 1, 1))

    def forward(self, X):
        h, w, c_i = X.shape
        c_o = self.filters.shape[0]
        X = X.reshape(c_i, h * w)
        K = self.filters.reshape(c_o, c_i)
        Y = np.dot(K, X)  # Matrix multiplication in the fully connected layer
        return Y.reshape(h, w, c_o)[0][0]



