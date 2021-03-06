import numpy as np
import random
import matplotlib.pyplot as plt

class Dataset(object):
    def __init__(self, class_num = 2, data_num = 200, seed = 4, center=None):
        '''
        :param class_num: class number
        :param data_num: data number for each class
        :param seed: random seed
        :param center: cluster center of data
        '''
        self.class_num = class_num
        self.data_num = data_num
        self.data = np.zeros((data_num * class_num, 2), dtype = np.float32)
        if center == None:
            # self.centers = [(0, 0), (0, 1), (1, 1), (1, 0)]
            self.centers = [(0, 0), (1, 0), (0.5, 0.866)]
            # self.centers = [(0, 1), (0, 0.9), (0, 1.1), (0.1, 1.1)]
        else:
            self.centers = center
        self.colors = ['red', 'green', 'blue']

        random.seed(seed)

    def generate(self):
        '''
        generate data
        :return: None
        '''
        for k in range(self.class_num):
            mu_x = self.centers[k][0]
            mu_y = self.centers[k][1]
            sigma = random.random() * 0.4
            for i in range(self.data_num):
                self.data[k * self.data_num + i][0] = np.random.normal(mu_x, sigma)
                self.data[k * self.data_num + i][1] = np.random.normal(mu_y, sigma)

    def show(self):
        '''
        show the distribution of data
        :return: None
        '''
        plt.figure()
        for i in range(self.class_num):
            x = self.data[i * self.data_num: (i + 1) * self.data_num - 1, 0]
            y = self.data[i * self.data_num: (i + 1) * self.data_num - 1, 1]
            plt.scatter(x, y, s = 10, c = self.colors[i])
        plt.show()

if __name__ == '__main__':
    dataset = Dataset()
    dataset.generate()
    dataset.show()