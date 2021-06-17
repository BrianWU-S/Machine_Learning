import numpy as np
from sklearn.datasets import make_blobs
from sklearn import mixture
from scipy import linalg
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt

np.random.seed(1)

n_samples = 100
color_iter = itertools.cycle(['red', 'blue', 'yellow', 'green'])
color_iter_more = itertools.cycle(['red', 'blue', 'yellow', 'green', 'orange', 'pink', 'purple'])


def plot_results_gmm(gmm,X,Y_,means,index,title):
    splot=plt.subplot(1,2,1+index)
    for i,(mean,color) in enumerate(zip(means,color_iter_more)):
        if gmm.covariance_type=='full':
            covariances=clf.covariances_[i][:2,:2]
        elif gmm.covariance_type=='tied':
            covariances=gmm.covariances_[:2,:2]
        elif gmm.covariance_type=='diag':
            covariances=np.diag(gmm.covariances_[i][:2])
        elif gmm.covariance_type=='spherical':
            covariances=np.eye(gmm.means_.shape[1])*gmm.covariances_[i]
        v,w=np.linalg.eigh(covariances)
        v=2.*np.sqrt(2.)*np.sqrt(v)
        u=w[0]/linalg.norm(w[0])
        if not np.any(Y_==i):
            continue
        plt.scatter(X[Y_==i,0],X[Y_==i,1],0.8,color=color)
        angle=np.arctan(u[1]/u[0])
        angle=180.*angle/np.pi


if __name__ == '__main__':
    C1 = np.array([[0., -0.1], [0.7, 4]])
    C2 = np.array([[-1, 0.1], [0.3, -0.2]])
    X1 = np.r_[np.dot(np.random.randn(n_samples, 2), C1), 0.7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
    X2 = np.r_[np.dot(np.random.randn(n_samples, 2), C2), 0.4 * np.random.randn(n_samples, 2) + np.array([1, 2])]
    X0 = np.r_[X1, X2]
    X, y = make_blobs(n_samples=50, centers=3, random_state=1234)
    lowest_AIC = np.infty
    AIC_list = []
    n_components_range = range(1, 9)
    covariance_type_list = ['spherical', 'tied', 'diag', 'full']
    for covariance_type in covariance_type_list:
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type)
            gmm.fit(X)
            AIC_list.append(gmm.aic(X))
            if AIC_list[-1] < lowest_AIC:
                lowest_AIC = AIC_list[-1]
                best_model = gmm
    AIC_list = np.array(AIC_list)
    clf = best_model
    bars = []
    spl=plt.subplot(1,2,1)
    for i,(cv_type,color) in enumerate(zip(covariance_type_list,color_iter)):
        xpos=np.array(n_components_range)+0.2*(i-2)
        bars.append(plt.bar(xpos,AIC_list[i*len(n_components_range):(i+1)*len(n_components_range)],color=color))
    plt.xticks(n_components_range)
    plt.ylim([AIC_list.min()*1.01-0.01*AIC_list.max(),AIC_list.max()])
    plt.title("AIC score for each model")
    xpos=np.mod(AIC_list.argmin(),len(n_components_range))+0.65+0.2*np.floor(AIC_list.argmin()/len(n_components_range))
    plt.text(xpos,AIC_list.min()*0.97+0.03*AIC_list.max(),'*',fontsize=14)
    spl.set_xlabel("Number of components")
    spl.legend([b[0] for b in bars],covariance_type_list)

    plot_results_gmm(clf,X,clf.predict(X),clf.means_,1,"Gaussian Mixture of AIC")
    plt.show()
