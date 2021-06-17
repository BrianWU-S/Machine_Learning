import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt


def data_generation(sample_size=100, output_dim=10, latent_dim=5, covariance=0.1, mu=0., A_scale=10.):
    assert covariance >= 0, "covariance need to be >=0 "
    # randomly sample latent variable y from Gaussian distribution
    y_t = np.random.normal(loc=0., scale=1.0, size=(sample_size, latent_dim))
    print("The shape of y:", y_t.shape)
    # randomly sample error variable e from Gaussian distribution
    e_t = np.random.normal(loc=0., scale=np.sqrt(covariance), size=(sample_size, output_dim))
    print("The shape of e:", e_t.shape)
    # randomly generate A
    A = np.random.normal(loc=0., scale=0.5, size=(latent_dim, output_dim)) * A_scale
    print("The shape of A:", A.shape)
    # construct xt
    x_t = np.add(np.add(np.dot(y_t, A), e_t), np.ones_like(e_t) * mu)
    print("The shape of X:", x_t.shape)
    return x_t


def test_model(n_components):
    fa_egi = FactorAnalyzer(n_factors=n_components, rotation=None)
    fa_egi.fit(X)
    ev, v = fa_egi.get_eigenvalues()
    print("Eigenvalues", ev)
    print("Plotting eigenvalues")
    plotting(ev=ev)
    fa_model = FactorAnalysis(n_components=n_components, random_state=1234)
    fa_model.fit(X)
    score = fa_model.score(X)  # Compute the average log-likelihood of the samples
    params = fa_model.get_params()
    print("Log Likelihood:", score)
    print("Params:", np.shape(params))


def AIC(log_prob, num_params):
    return log_prob - num_params


def BIC(log_prob, num_params, N):
    return log_prob - 0.5 * np.log(N) * num_params


def FA_model(M):
    nc_list = np.arange(1, M)
    aic_max = -10000000
    bic_max = -10000000
    best_m_aic = 0
    best_m_bic = 0
    for n_components in nc_list:
        fa_model = FactorAnalysis(n_components=n_components)
        fa_model.fit(X)
        score = fa_model.score(X)*X.shape[0]  # Compute the average log-likelihood of the samples
        print("n_components:", n_components, "Log Likelihood:", score)
        num_params = latent_dim * n_components + 1
        aic_score = AIC(log_prob=score, num_params=num_params)
        bic_score = BIC(log_prob=score, num_params=num_params, N=sample_size)
        if aic_score > aic_max:
            aic_max = aic_score
            best_m_aic = n_components
        if bic_score > bic_max:
            bic_max = bic_score
            best_m_bic = n_components
        print("AIC score:", aic_score, "BIC score:", bic_score, "\n")
    print("Best AIC score:", aic_max, "Best m:", best_m_aic)
    print("Best BIC score:", bic_max, "Best m:", best_m_bic)


def plotting(ev):
    # Create scree plot using matplotlib
    plt.scatter(range(1, X.shape[1] + 1), ev)
    plt.plot(range(1, X.shape[1] + 1), ev)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    sample_size = 1000      # AIC 可以找到： sample_size=500, output_dim=10, latent_dim=3
    output_dim = 18
    latent_dim = 5
    covariance = 0.1
    mu = 0
    A_scale = 3
    M = output_dim
    
    X = data_generation(sample_size=sample_size, output_dim=output_dim, latent_dim=latent_dim, covariance=covariance,
                        mu=mu, A_scale=A_scale)
    test_model(n_components=latent_dim)
    FA_model(M)
