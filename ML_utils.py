from sklearn.linear_model import LinearRegression
from mlinsights.mlmodel import PiecewiseRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
import pickle

import numpy as np


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


def fit_piecewise_polynomial_regression_model(degree, X_train, Y_train, X_val, Y_val, nsplits=4):
    """
    Creates a piecewise polynomial regression model for the given degree
    :param degree:
    :param X_train:
    :param Y_train:
    :param X_val:
    :param Y_val:
    :param nsplits:
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
    hidden_structure = tuple(np.repeat(num_neurons, num_layers))
    ann_model = MLPRegressor(hidden_structure)
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


def ensemble_predict(model_list, X):
    y = []
    for model in model_list:
        y.append(model.predict(X))
    y_predicted = np.nanmean(np.array(y), axis=0)
    return y_predicted


def rmse_learning_curve(model, x, y, n_splits):
    """
    gives mean and standard deviation of the rmse as a function of training size
    :param model: model object on which "fit" and "predict" can be called
    :param x: training data (input)
    :param y: training data (target values)
    :param n_splits: number of splits for cross-validation
    """
    train_sizes_abs, train_scores, test_scores = learning_curve(model, x, y, shuffle=True, cv=n_splits,
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


def get_scaler(x):
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler


def split_train_test(x, y, test_size=0.1):
    """
    Use sklearn library to split into training and testing data set
    :param x: array containing predictors (input variables)
    :param y: array containing predictands (target variables)
    :param test_size: if float, describes the fraction of data which will be contained in the test set. Standard value
    is set to 10%
    :return: x_train, x_test, y_train, y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def kfold_cv(x, n_splits):
    """
    utility for k-fold cross-validation
    :param x: input dataset
    :param n_splits: number of splits
    :return:
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return kf.split(x)


def kfold_cv_ann(n_splits, max_n_layers, max_n_neurons, x, y):
    rmses_train = np.zeros((n_splits, max_n_layers, max_n_neurons))
    rmses_val = np.zeros((n_splits, max_n_layers, max_n_neurons))
    kfsplits = kfold_cv(x, n_splits=n_splits)

    s = 0
    for train_index, val_index in kfsplits:
        x_train, x_val = x[train_index, :], x[val_index, :]
        y_train, y_val = y[train_index,], y[val_index,]
        for n_l in range(max_n_layers):
            for n in range(max_n_neurons):
                print(f'k={s + 1}, number of layers: {n_l + 1}, number of neurons: {n + 1}...')
                model = fit_ann_model(num_layers=n_l + 1, num_neurons=n + 1, X_train=x_train, Y_train=y_train)
                rmses_train[s, n_l, n] = prediction_rmse(model, x_train, y_train)
                rmses_val[s, n_l, n] = prediction_rmse(model, x_val, y_val)
                print(f'RMSE training set: {rmses_train[s, n_l, n]}, validation set: {rmses_val[s, n_l, n]}')
        s += 1
    return rmses_train, rmses_val
