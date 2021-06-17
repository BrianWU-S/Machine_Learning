from math import exp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def distance(x1, x2):
    return (x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2


def dataset_construct():
    X_ps, Y_ps = make_blobs(n_samples=[100, 100, 100], centers=[[0, 0], [0, 1], [1, -1]], cluster_std=0.2)
    plt.scatter(X_ps[:, 0], X_ps[:, 1], c=[["lightgreen", "tomato", "lightblue"][k] for k in Y_ps], alpha=0.4)
    plt.show()
    return X_ps, Y_ps


class DataCenter(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.data = None
        self.count = 0


class MyKmeans(object):
    def __init__(self, init_centers=10):
        self.init_centers = init_centers
        self.data, self.Y = dataset_construct()
        self.centers = []
        self.rho = []
        self.alpha_c = 0.05
        self.alpha_r = 0.05
        self.theta_1 = 0.01
        self.theta_2 = 45

    def initialization(self):
        for i in range(self.init_centers):
            self.centers.append(
                DataCenter(x=np.random.normal(loc=0.5, scale=0.5), y=np.random.normal(loc=0., scale=0.5)))

    def get_density(self):
        b = 0.
        for i in self.data:
            b += distance(self.data[0], i)
        for i in self.data:
            v = 0.0
            for j in self.data:
                v += distance(i, j) / b
            self.rho.append(exp(-v))

    def visualize(self, fp=None, color="red"):
        plt.figure()
        plt.scatter(self.data[:, 0], self.data[:, 1])
        for c in self.centers:
            if c != None:
                plt.scatter(c.x, c.y, c=color, s=25)
        if fp == None:
            fp = 'images/tmp.png'
        plt.savefig(fp)
        plt.show()

    def RPCL(self):
        self.get_density()
        self.initialization()
        self.visualize(fp="images/RPCL_init.png", color='black')
        dx = 0
        dy = 0
        while True:
            for idx, i in enumerate(self.data):
                dists = []
                for c in self.centers:
                    dists.append(distance(i, (c.x, c.y)))
                c_idx = np.argmax(dists)  # winner index
                self.centers[c_idx].x += self.alpha_c * self.rho[idx] * (i[0] - self.centers[c_idx].x)
                self.centers[c_idx].y += self.alpha_c * self.rho[idx] * (i[1] - self.centers[c_idx].y)

                dists[c_idx] = -1000
                r_idx = np.argmax(dists)  # second winner index
                dx = self.alpha_r * self.rho[idx] * (i[0] - self.centers[r_idx].x)
                dy = self.alpha_r * self.rho[idx] * (i[1] - self.centers[r_idx].y)
                self.centers[r_idx].x -= dx
                self.centers[r_idx].y -= dy
            if (dx + dy) < self.theta_1:
                break
        # assign each point to nearest center
        for i in self.data:
            d_max = 100000
            index = None
            for idx, c in enumerate(self.centers):
                d = distance(i, (c.x, c.y))
                if d < d_max:
                    d_max = d
                    index = idx
            self.centers[index].count += 1
        # remove extra centers
        remove_list = []
        for idx, c in enumerate(self.centers):
            if c.count < self.theta_2:
                remove_list.append(idx)
                self.centers[idx] = None
        print("Removed centers:\n", remove_list)
        self.visualize(fp="images/RPCL.png", color='red')


if __name__ == "__main__":
    # X_ps, Y_ps = dataset_construct()
    model = MyKmeans(init_centers=10)
    model.RPCL()
