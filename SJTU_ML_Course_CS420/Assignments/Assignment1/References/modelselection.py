import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import make_blobs

from sklearn import mixture

color_iter = ['navy', 'c', 'cornflowerblue', 'gold'
                              ,'red','pink','green']


def plot_results_dpgmm(X, Y_, means, covariances,index, title):
    splot = plt.subplot(1,2, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        '''
        if gmm.covariance_type == 'full':
            covariances = clf.covariances_[i][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[i][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[i]
            '''
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    #plt.xlim(-9., 5.)
    #plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


def plot_results_gmm(gmm,X, Y_, means, index, title):
    splot = plt.subplot(1,2, 1 + index)
    for i, (mean,  color) in enumerate(zip(
            means,  color_iter)):

        if gmm.covariance_type == 'full':
            covariances = clf.covariances_[i][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[i][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[i]

        v, w = np.linalg.eigh(covariances)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    #plt.xlim(-9., 5.)
    #plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)
C1 = np.array([[0., -0.1], [0.7, .4]])
C2 = np.array([[-1,0.1],[0.3,-0.2]])
X1 = np.r_[np.dot(np.random.randn(n_samples, 2), C1),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
X2=np.r_[np.dot(np.random.randn(n_samples, 2), C2),
          .4 * np.random.randn(n_samples, 2) + np.array([1,2])]
X0=np.r_[X1,X2]
#Generate random sample
X, y = make_blobs(n_samples=50, centers=3, n_features=2,random_state=0)
# Fit a Gaussian mixture with AIC
lowest_aic = np.infty
aic = []
n_components_range = range(1, 9)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        aic.append(gmm.aic(X))
        if aic[-1] < lowest_aic:
            lowest_aic = aic[-1]
            best_gmm1 = gmm

aic = np.array(aic)
color_iter2 = itertools.cycle(['navy', 'pink', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm1
bars = []

# Plot the BIC scores
spl = plt.subplot(1,2, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter2)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, aic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([aic.min() * 1.01 - .01 * aic.max(), aic.max()])
plt.title('AIC score per model')
xpos = np.mod(aic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(aic.argmin() / len(n_components_range))
plt.text(xpos, aic.min() * 0.97 + .03 * aic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)

#plot the aic winner
plot_results_gmm(clf,X, clf.predict(X), clf.means_,1,
             'Gaussian Mixture of AIC')
plt.show()

# Fit a Gaussian mixture with BIC
lowest_bic = np.infty
bic = []
n_components_range = range(1, 9)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm2 = gmm

bic = np.array(bic)
color_iter2 = itertools.cycle(['navy', 'pink', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm2
bars = []

# Plot the BIC scores
spl = plt.subplot(1,2, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter2)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)

#plot the bic winner
plot_results_gmm(clf,X, clf.predict(X), clf.means_,1,
             'Gaussian Mixture of BIC')
plt.show()
# Fit a Dirichlet process Gaussian mixture using five components
dpgmm = mixture.BayesianGaussianMixture(n_components=8,
                                        covariance_type='full').fit(X)
plot_results_dpgmm(X, dpgmm.predict(X), dpgmm.means_,dpgmm.covariances_,0,
             'VBEM')

plt.show()