import numpy as np

class Conv3x3:
    def __init__(self, num_of_filters):
        self.num_of_filters = num_of_filters
        self.filters = np.random.uniform(size=(3, 3, self.num_of_filters))/9


    def forward(self, region):

        output = (np.sum(region * self.filters, axis=(0,1)).reshape((1, 1, 512)))

        return output