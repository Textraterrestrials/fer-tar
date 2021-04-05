import pandas as pd


def load_data(path_to_data_folder='/content/gdrive/MyDrive/data'):
    """
    :param path_to_data_folder: (str) path to the folder contain train, dev and test data. If not specified the default
     path is used: /content/gdrive/MyDrive/data
    :return: (tuple) X_train, X_dev, X_test, y_train, y_dev, y_test
    """
    X_train = pd.read_csv(path_to_data_folder + '/Training/subtaskA_data_all.csv', index_col=0)
    X_dev = pd.read_csv(path_to_data_folder + '/Dev/subtaskA_dev_data.csv', index_col=0)
    X_test = pd.read_csv(path_to_data_folder + '/Test/subtaskA_test_data.csv', index_col=0)

    y_train = pd.read_csv(path_to_data_folder + '/Training/subtaskA_answers_all.csv', index_col=0, header=None)
    y_dev = pd.read_csv(path_to_data_folder + '/Dev/subtaskA_gold_answers.csv', index_col=0, header=None)
    y_test = pd.read_csv(path_to_data_folder + '/Test/subtaskA_gold_answers.csv', index_col=0, header=None)

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def melt_columns(X, y):
    """
    :param X: (pandas.dataframe) a pandas dataframe containing pairs of sentences
    :param y: (numpy.array) labels that determine which sentence makes sense
    :return: (tuple) pandas dataframe containing only single sentences and a numpy array of 0s and 1s where 1 means
    that a corresponding sentence makes sense and 0 means otherwise.
    """
    X_melted = pd.melt(X)['value']
    X_melted.columns = ['sentence']
    y_melted = 1 - y
    y_melted = y_melted.append(y)

    return X_melted, y_melted
