import numpy as np

class Conv1x1:
    def __init__(self, in_shape, out_shape):
        self.inp = in_shape
        self.out = out_shape
        self.filters = np.random.normal(0.0, 0.1, size=(1, 1, self.inp, self.out))

    def forward(self, X):
        output = np.zeros((1, self.out))

        for i in range(self.out):
            output[:, i] = np.sum((np.multiply(self.filters[:, :, :, i], X)))
            # print(self.filters[:,:,:,i])

        return output[0]

    def backpropagate(self, d_L, learning_rate, index):
        self.filters[:, :, :, index] -= (learning_rate * d_L)
        pass


    def mutate(self, mutation_rate):
        self.filters += np.random.normal(0.0, mutation_rate, (self.filters.shape))



