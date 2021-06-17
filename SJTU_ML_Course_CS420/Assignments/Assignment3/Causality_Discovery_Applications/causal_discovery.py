import numpy as np
import pandas as pd
import graphviz
import lingam
from sklearn.linear_model import LassoCV
import lightgbm as lgb


np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)


def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine='dot')
    names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=f'{coef:.2f}')
    return d


if __name__ == '__main__':
    print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])
    X = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original',
                    delim_whitespace=True, header=None,
                    names=['mpg', 'cylinders', 'displacement',
                           'horsepower', 'weight', 'acceleration',
                           'model year', 'origin', 'car name'])
    X.dropna(inplace=True)
    X.drop(['model year', 'origin', 'car name'], axis=1, inplace=True)
    print(X.shape)
    print(X.head())
    # causal discovery
    model = lingam.DirectLiNGAM()
    model.fit(X)
    labels = [f'{i}. {col}' for i, col in enumerate(X.columns)]
    d=make_graph(model.adjacency_matrix_, labels)
    # train the model
    target = 0  # mpg
    features = [i for i in range(X.shape[1]) if i != target]
    reg = lgb.LGBMRegressor(random_state=0)
    reg.fit(X.iloc[:, features], X.iloc[:, target])
    # identification of feature influence on model

    ce = lingam.CausalEffect(model)
    effects = ce.estimate_effects_on_prediction(X, target, reg)

    df_effects = pd.DataFrame()
    df_effects['feature'] = X.columns
    df_effects['effect_plus'] = effects[:, 0]
    df_effects['effect_minus'] = effects[:, 1]
    print(df_effects)
    max_index = np.unravel_index(np.argmax(effects), effects.shape)
    print(X.columns[max_index[0]])
    
    print("Training process finished. Congratulations, sir!")