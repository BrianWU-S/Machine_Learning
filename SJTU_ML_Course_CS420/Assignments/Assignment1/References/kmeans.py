from sklearn.cluster import KMeans
from dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from math import exp

def dis(cor1, cor2):
    '''
    distance function
    :param cor1: first point
    :param cor2: second point
    :return: distance
    '''
    return (cor1[0] - cor2[0])**2 + (cor1[1] - cor2[1])**2

class Center(object):
    '''
    cluster center
    '''
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
        self.data = None
        self.cnt = 0

class kmeans(object):
    def __init__(self, n_clusters = 1, verbose = 2):
        self.model = KMeans(
            n_clusters = n_clusters,
            verbose = verbose)
        self.n_clusters = n_clusters
        self.dataset = Dataset(class_num = 3)
        self.dataset.generate()
        self.data = self.dataset.data
        self.centers = []
        self.rho = []

    def get_density(self):
        '''
        get density of each data point
        :return: None
        '''
        b = 0.0
        for i in self.data:
            b += dis(self.data[0], i)

        for i in self.data:
            v = 0.0
            for j in self.data:
                v += dis(i, j) / b
            self.rho.append(exp(-v))

    def RPCL(self):
        # init
        self.get_density()
        self.random_center()
        self.show('report/demo/RPCL_1')
        alpha = 0.04
        beta = 0.04

        # training weight vectors
        while True:
            for idx, i in enumerate(self.data):
                distances = []
                for c in self.centers:
                    distances.append(dis(i, (c.x, c.y)))

                winner_idx = distances.index(max(distances))
                delta_x = alpha * self.rho[idx] * (i[0] - self.centers[winner_idx].x)
                delta_y = alpha * self.rho[idx] * (i[1] - self.centers[winner_idx].y)
                self.centers[winner_idx].x += delta_x
                self.centers[winner_idx].y += delta_y
                distances[winner_idx] = -1

                rival_idx = distances.index(max(distances))
                delta_x = beta * self.rho[idx] * (i[0] - self.centers[rival_idx].x)
                delta_y = beta * self.rho[idx] * (i[1] - self.centers[rival_idx].y)
                self.centers[rival_idx].x -= delta_x
                self.centers[rival_idx].y -= delta_y

            if (delta_x + delta_y) < 0.01:
                break

        # assign each data
        for i in self.data:
            tmp = 999
            res = None
            for idx, c in enumerate(self.centers):
                if dis(i, (c.x, c.y)) < tmp:
                    tmp = dis(i, (c.x, c.y))
                    res = idx
            self.centers[res].cnt += 1

        # remove extra centers
        eta = 100
        kill_list = []
        for idx, c in enumerate(self.centers):
            print(c.cnt)
            if c.cnt < eta:
                kill_list.append(idx)

        for i in kill_list:
            self.centers[i] = None

        self.show('report/demo/RPCL_2')

    def random_center(self):
        '''
        generate centers randomly
        :return: initial centers
        '''
        for i in range(8):
            self.centers.append(Center(x=np.random.normal(0.5, 0.2),
                                       y=np.random.normal(0.5, 0.2)))

    def show(self, filename = None):
        '''
        show results
        :param filename: file to save
        :return: None
        '''
        plt.figure()

        plt.scatter(self.data[:, 0], self.data[:, 1], s=15)
        for c in self.centers:
            if c != None:
                plt.scatter(c.x, c.y, c='yellow', s=25)

        if filename != None:
            plt.savefig(filename)
        plt.show()

if __name__ == '__main__':
    km = kmeans()
    km.RPCL()