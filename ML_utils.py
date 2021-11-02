from sklearn.linear_model import LinearRegression
from mlinsights.mlmodel import PiecewiseRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import r2_score
import pickle
import numpy as np
import pandas as pd

#from yellowbrick.regressor import ResidualsPlot, PredictionError


def fit_polynomial_regression_model(degree, X_train, Y_train):
    """
    Creates a polynomial regression model for the given degree
    :param degree: degree of the polynomial
    :param X_train: input features for training
    :param Y_train: target values for training
    :return: poly_model: The polynomial regression model
    """

    poly_features = PolynomialFeatures(degree=degree)
    # transform the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)

    # fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, Y_train)

    return poly_model


def fit_piecewise_polynomial_regression_model(degree, X_train, Y_train, nsplits=4):
    """
    Creates a piecewise polynomial regression model for the given degree
    :param degree: degree of the polynomial
    :param X_train: features for training
    :param Y_train: the 'true' output (expected values)
    :param nsplits: number of splits of the training data set
    :return:
    """
    samplesize = int(np.floor(X_train.shape[0]/nsplits))
    poly_features = PolynomialFeatures(degree=degree)
    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)

    # fit the transformed features to Linear Regression
    poly_model = PiecewiseRegressor(verbose=True, binner=DecisionTreeRegressor(min_samples_leaf=samplesize))
    poly_model.fit(X_train_poly, Y_train)

    return poly_model


def create_ann_model(num_neurons, num_layers):
    """
    Creates an object of the class sklearn.neural_network.MLPRegressor
    :param num_neurons: number of neurons in each hidden layer
    :param num_layers: number of hidden layers
    :return: the model object
    """
    hidden_structure = tuple(np.repeat(num_neurons, num_layers))
    ann_model = MLPRegressor(hidden_structure, random_state=8)
    return ann_model


def fit_ann_model(num_layers, num_neurons, X_train, Y_train):
    """
    train a multilayer perceptron (MLP)
    :param num_layers: number of hidden layers
    :param X_train: training data (input)
    :param Y_train: training data (target values)
    :param num_neurons: number of neurons per layer
    :return: ann_model: The model
    """
    ann_model = create_ann_model(num_neurons, num_layers)
    ann_model.fit(X_train, Y_train)

    return ann_model


def prediction_rmse(model, X, Y):
    """
    :param model: model on which model.predict(X) can be called
    :param X: input
    :param Y: output (the truth)
    :return: rmse: root mean squared error
    """
    # predicting
    y_predicted = model.predict(X)
    rmse = np.sqrt(mean_squared_error(Y, y_predicted))
    return rmse


def prediction_rmse_poly(model, X, Y, degree):
    """
    wrapper for prediction_rmse for polynomial models
    :param model:
    :param X:
    :param Y:
    :return:
    """
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    y_predicted = model.predict(X_poly)
    rmse = np.sqrt(mean_squared_error(Y, y_predicted))
    return rmse


def prediction_rmse_ensemble(model_list, X, Y):
    """

    :param model_list: list of models on which predict can be called
    :param X: input
    :param Y: target output (the truth)
    :return: rmse: the rmse of the ensemble predictions and the target y values
    """
    y_predicted = ensemble_predict(model_list, X)
    rmse = np.sqrt(mean_squared_error(Y, y_predicted))
    return rmse


def ensemble_predict(model_list, x, return_std=False):
    y = []
    for model in model_list:
        y.append(model.predict(x))
    y_predicted = np.nanmean(np.array(y), axis=0)
    if return_std:
        y_std = np.nanstd(np.array(y), axis=0)
        return y_predicted, y_std
    else:
        return y_predicted


def ensemble_predict_poly(model_list, x, degree, return_std=False):
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x)
    y = []
    for model in model_list:
        y.append(model.predict(x_poly))
    y_predicted = np.nanmean(np.array(y), axis=0)
    if return_std:
        y_std = np.nanstd(np.array(y), axis=0)
        return y_predicted, y_std
    else:
        return y_predicted


class EnsembleEstimator:
    def __init__(self, models):
        self.intercepts_ = [i.intercepts_ for i in models]
        self.models = models
        self._estimator_type = 'regressor'
    def fit(self):
        return NotImplementedError, "on my list"

    def predict(self, X):
        return ensemble_predict(self.models, X)

    def score(self, X, y, sample_weight=None):
        score = []
        for model in self.models:
            score.append(model.score(X, y, sample_weight))
        return np.nanmean(score)


def rmse_learning_curve(model, x, y, n_splits):
    """
    gives mean and standard deviation of the rmse as a function of training size
    :param model: model object on which "fit" and "predict" can be called
    :param x: training data (input)
    :param y: training data (target values)
    :param n_splits: number of splits for cross-validation
    """
    train_sizes_abs, train_scores, test_scores = learning_curve(model, x, y, shuffle=False, cv=n_splits,
                                                            scoring='neg_mean_squared_error')
    train_scores = np.sqrt(-train_scores)
    test_scores = np.sqrt(-test_scores)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    return train_sizes_abs, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std


def rmse_train_validation(model, X_train, Y_train, X_val, Y_val):
    rmse_training = prediction_rmse(model, X_train, Y_train)
    rmse_validation = prediction_rmse(model, X_val, Y_val)
    print("Model performance for the training set")
    print("-------------------------------------------")
    print("RMSE of training set is {}".format(rmse_training))

    print("\n")

    print("Model performance for the test set")
    print("-------------------------------------------")
    print("RMSE of test set is {}".format(rmse_validation))

def r2(x, y):
    return r2_score(x,y)

def save_to_file(model, filename):
    """
    save a ML model or scaler object to a file
    :param model: model or scaler to be saved using pickle
    :param filename: name of the file to be written
    :return:
    """
    pickle.dump(model, open(filename, 'wb'))
    print(f'saved object to file {filename}')


def load_from_file(filename):
    model = pickle.load(open(filename, 'rb'))
    return model


def sigmoid(x):
    """
    based on code from https://www.kaggle.com/aleksandradeis/regression-addressing-extreme-rare-cases
    https://en.wikipedia.org/wiki/Sigmoid_function
    Args:
        x: x values for which the sigmoid function is computed

    """
    return 1 / (1 + np.exp(-x))


def gaussian(x, mu, sig):
    """
    based on code from
    https://stackoverflow.com/questions/14873203/plotting-of-1-dimensional-gaussian-distribution-function
    Args:
        x: x values for which the Gaussian is computed
        mu: mean
        sig: standard deviation

    Returns:

    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def relevance_ka(x):
    """
    based on code from https://www.kaggle.com/aleksandradeis/regression-addressing-extreme-rare-cases
    see paper: https://www.researchgate.net/publication/220699419_Utility-Based_Regression
    use the sigmoid function to create the relevance function, so that relevance function
    has values close to 1 when the target variable is greater than 0.6
    Args:
        x: the x values for which the relevance should be returned

    """
    x = np.array(x)
    return sigmoid((x-0.5) * 15)


def relevance_w(x):
    """

    Args:
        x: the x values for which the relevance should be returned

    """
    x = np.array(x)
    return 1 - gaussian(x, 0.48, 0.07) - 0.15*gaussian(x, 0.25, 0.1) + 0.1*gaussian(x, 0.58, 0.05)


def get_synth_cases(D, target, o=200, k=3, categorical_col = []):
    '''
    based on code from https://www.kaggle.com/aleksandradeis/regression-addressing-extreme-rare-cases
    Function to generate the new cases.
    Args:
        D - pd.DataFrame with the initial data
        target - string name of the target column in the dataset
        o - oversampling rate
        k - number of nearest neighbors to use for the generation
        categorical_col - list of categorical column names
    Returns:
        new_cases - pd.DataFrame containing new generated cases
    '''

    new_cases = pd.DataFrame(columns = D.columns) # initialize the list of new cases
    ng = o // 100 # the number of new cases to generate
    for index, case in D.iterrows():
        # find k nearest neighbors of the case
        knn = KNeighborsRegressor(n_neighbors = k+1) # k+1 because the case is the nearest neighbor to itself
        knn.fit(D.drop(columns = [target]).values, D[[target]])
        neighbors = knn.kneighbors(case.drop(labels = [target]).values.reshape(1, -1), return_distance=False).reshape(-1)
        neighbors = np.delete(neighbors, np.where(neighbors == index))
        for i in range(0, ng):
            # randomly choose one of the neighbors
            x = D.iloc[neighbors[np.random.randint(k)]]
            attr = {}
            for a in D.columns:
                # skip target column
                if a == target:
                    continue
                if a in categorical_col:
                    # if categorical then choose randomly one of values
                    if np.random.randint(2) == 0:
                        attr[a] = case[a]
                    else:
                        attr[a] = x[a]
                else:
                    # if continious column
                    diff = case[a] - x[a]
                    attr[a] = case[a] + np.random.randint(2) * diff
            # decide the target column
            new = np.array(list(attr.values()))
            d1 = cosine_similarity(new.reshape(1, -1), case.drop(labels = [target]).values.reshape(1, -1))[0][0]
            d2 = cosine_similarity(new.reshape(1, -1), x.drop(labels = [target]).values.reshape(1, -1))[0][0]
            attr[target] = (d2 * case[target] + d1 * x[target]) / (d1 + d2)

            # append the result
            new_cases = new_cases.append(attr,ignore_index = True)

    return new_cases


def smoter(D, target, relevance_function, th=0.9, o=200, u=100, k=3, categorical_col=[]):
    '''
    based on code from https://www.kaggle.com/aleksandradeis/regression-addressing-extreme-rare-cases
    smoter = synthetic minority oversampling technique (for regression problems)
    The implementation of SmoteR algorithm:
    https://core.ac.uk/download/pdf/29202178.pdf
    Args:
        D - pd.DataFrame - the initial dataset
        target - the name of the target column in the dataset
        th - relevance threshold
        o - oversampling rate
        u - undersampling rate
        k - the number of nearest neighbors
    Returns:
        new_D - the resulting new dataset
    '''
    # median of the target variable
    y_bar = D[target].median()

    # find rare cases where target less than median
    rareL = D[(relevance_function(D[target]) > th) & (D[target] > y_bar)]
    # generate rare cases for rareL
    new_casesL = get_synth_cases(rareL, target, o, k , categorical_col)

    # find rare cases where target greater than median
    rareH = D[(relevance_function(D[target]) > th) & (D[target] < y_bar)]
    # generate rare cases for rareH
    new_casesH = get_synth_cases(rareH, target, o, k , categorical_col)

    new_cases = pd.concat([new_casesL, new_casesH], axis=0)

    # undersample norm cases
    norm_cases = D[relevance_function(D[target]) <= th]
    # get the number of norm cases
    nr_norm = int(len(norm_cases) * u / 100)

    norm_cases = norm_cases.sample(min(len(D[relevance_function(D[target]) <= th]), nr_norm))

    # get the resulting dataset
    new_D = pd.concat([new_cases, norm_cases], axis=0)

    return new_D


def smoter_x_y(X_data, Y_data, relevance_function=relevance_ka, **kwargs):
    """
    Perform synthetic minority oversampling technique for regression tasks
    Args:
        X_data: nxm features
        Y_data: nx1 targets

    Returns: X_data_new, Y_data_new (containing synthetic samples)

    """
    smoter_in = pd.DataFrame(np.hstack((X_data, Y_data[:, np.newaxis])),
                             columns=['x'+str(i) for i in np.arange(X_data.shape[1])] + ['y'])
    print('synthetic minority oversampling technique for regression...')
    smoter_out = smoter(smoter_in, 'y', relevance_function, **kwargs)
    return np.array(smoter_out)[:, :X_data.shape[1]], np.array(smoter_out)[:, -1]


def get_scaler(x, scaler='standard'):
    assert scaler in ['standard', 'robust', 'minmax'], f"scaler must be 'robust', 'standard' or 'minmax'," \
                                                       f" but is {scaler}."
    if scaler == 'standard':
        scaler_obj = StandardScaler()
    elif scaler == 'robust':
        scaler_obj = RobustScaler()
    elif scaler == 'minmax':
        scaler_obj = MinMaxScaler()

    scaler_obj.fit(x)
    return scaler_obj


def split_train_test(x, y, test_size=0.1, fixed_test_index=[], shuffle=False):
    """
    Use sklearn library to split into training and testing data set
    :param x: array containing predictors (input variables)
    :param y: array containing predictands (target variables)
    :param test_size: if float, describes the fraction of data which will be contained in the test set. Standard value
    is set to 10%
    :return: x_train, x_test, y_train, y_test
    """
    print(f'running split_train_test with fixed_test_index of length {len(fixed_test_index)}')
    if len(fixed_test_index) > 0:
        train_index = [j for i, j in enumerate(np.arange(len(y))) if i not in fixed_test_index]
        x_test_fixed = x[fixed_test_index, :]
        y_test_fixed = y[fixed_test_index]
        x = x[train_index, :]
        y = y[train_index]
        test_size = test_size - len(fixed_test_index)/len(y)
        print(f'test size is now {test_size}')
    if test_size > 0:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=shuffle, random_state=42)
        y_test = np.hstack((y_test, y_test_fixed))
        x_test = np.vstack((x_test, x_test_fixed))
    return x_train, x_test, y_train, y_test


def split_train_test_manually(x, y, start_index=28992, test_size=0.1):
    # hard coded: start index for test set

    test_index = np.concatenate((np.arange(start_index, start_index+0.5*test_size*len(y), dtype=int),
                                np.arange(round(len(y)-0.5*test_size*len(y)), len(y), dtype=int)))
    train_index = ~np.isin(np.arange(len(y)), test_index)
    x_train = x[train_index, :]
    x_test = x[test_index, :]
    y_train = y[train_index, ]
    y_test = y[test_index, ]
    return x_train, x_test, y_train, y_test


def kfold_cv(x, n_splits, shuffle=False):
    """
    utility for k-fold cross-validation
    :param x: input dataset
    :param n_splits: number of splits
    :return:
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=42)
    return kf.split(x)


def kfold_cv_ann(n_splits, max_n_layers, max_n_neurons, x, y, **kwargs):
    """

    :param n_splits: number of splits for k-fold cross-validation
    :param max_n_layers: maximum number of hidden layers in the neural network
    :param max_n_neurons: maximum number of neurons in each layer
    :param x: features for training/ validation
    :param y: target values corresponding to the features
    :param kwargs:
     y_thresh: Threshold to apply to validation data if more weight should be put e.g. on the prediction of high values
     shuffle: boolean
    :return:
    """
    shuffle = kwargs['shuffle'] if 'shuffle' in kwargs else False
    rmses_train = np.zeros((n_splits, max_n_layers, max_n_neurons))
    rmses_val = np.zeros((n_splits, max_n_layers, max_n_neurons))
    kfsplits = kfold_cv(x, n_splits=n_splits, shuffle=shuffle)

    s = 0
    for train_index, val_index in kfsplits:
        x_train, x_val = x[train_index, :], x[val_index, :]
        y_train, y_val = y[train_index,], y[val_index,]
        if 'y_thresh' in kwargs:
            # apply threshold to validation data
            index_y = y_val > kwargs['y_thresh']
            x_val = x_val[index_y, :]
            y_val = y_val[index_y,]
        for n_l in range(max_n_layers):
            for n in range(max_n_neurons):
                print(f'k={s + 1}, number of layers: {n_l + 1}, number of neurons: {n + 1}...')
                model = fit_ann_model(num_layers=n_l + 1, num_neurons=n + 1, X_train=x_train, Y_train=y_train)
                rmses_train[s, n_l, n] = prediction_rmse(model, x_train, y_train)
                if len(y_val) > 1:
                    rmses_val[s, n_l, n] = prediction_rmse(model, x_val, y_val)
                else:
                    rmses_val[s, n_l, n] = np.nan
                print(f'RMSE training set: {rmses_train[s, n_l, n]}, validation set: {rmses_val[s, n_l, n]}')
        s += 1
    return rmses_train, rmses_val
