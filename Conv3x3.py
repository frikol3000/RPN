import numpy as np

class Conv3x3:
    def __init__(self, num_of_filters):
        self.num_of_filters = num_of_filters
        self.filters = np.random.normal(0.0, 0.1, size=(3, 3, self.num_of_filters))
        self.__last_input = None

    def region_extraction(self, feature_map):

        h, w, _ = feature_map.shape

        for i in range(h-2):
            for j in range(w-2):
                yield i, j, feature_map[i:i+3, j:j+3, :]

    def forward(self, feature):

        self.__last_input = feature

        h, w, d = feature.shape

        output = np.zeros((h-2, w-2, d))

        for i, j, region in self.region_extraction(feature):
            output[i, j, :] = (np.sum(region * self.filters, axis=(0,1)).reshape((1, 1, 512)))

        return output

    def backpropagate(self, d_L_out, learning_rate):

        d_L_d_filters = np.zeros(self.filters.shape)

        for i, j, im_region in self.region_extraction(self.__last_input):
            for f in range(self.num_of_filters):
                d_L_d_filters[:, :, f] += d_L_out[:, :, f] * im_region[:, :, f]

        self.filters -= learning_rate * d_L_d_filters
        pass

    def mutate(self, mutation_rate):
        #print(self.filters[0][0][0])
        self.filters += np.random.normal(0.0, mutation_rate, (self.filters.shape))
        #print(self.filters[0][0][0])