import numpy as np
import pandas as pd
import torch


def load_data2(
        data_folder='/content/gdrive/MyDrive/data',
        training_x='/subtaskA_data_all.csv',
        training_y='/subtaskA_answers_all.csv'):
    X_train = pd.read_csv(f'{data_folder}/Training/{training_x}', index_col=0)
    X_dev = pd.read_csv(f'{data_folder}/Dev/subtaskA_dev_data.csv', index_col=0)
    X_test = pd.read_csv(f'{data_folder}/Test/subtaskA_test_data.csv', index_col=0)

    y_train = pd.read_csv(f'{data_folder}/Training/{training_y}', index_col=0, header=None, dtype=np.float32)
    y_dev = pd.read_csv(f'{data_folder}/Dev/subtaskA_gold_answers.csv', index_col=0, header=None, dtype=np.float32)
    y_test = pd.read_csv(f'{data_folder}/Test/subtaskA_gold_answers.csv', index_col=0, header=None, dtype=np.float32)

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def load_data(path_to_data_folder='/content/gdrive/MyDrive/data'):
    """
    :param path_to_data_folder: (str) path to the folder contain train, dev and test data. If not specified the default
     path is used: /content/gdrive/MyDrive/data
    :return: (tuple) X_train, X_dev, X_test, y_train, y_dev, y_test
    """
    X_train = pd.read_csv(path_to_data_folder + '/Training/subtaskA_data_all.csv', index_col=0)
    X_dev = pd.read_csv(path_to_data_folder + '/Dev/subtaskA_dev_data.csv', index_col=0)
    X_test = pd.read_csv(path_to_data_folder + '/Test/subtaskA_test_data.csv', index_col=0)

    y_train = pd.read_csv(path_to_data_folder + '/Training/subtaskA_answers_all.csv', index_col=0, header=None,
                          dtype=np.float32)
    y_dev = pd.read_csv(path_to_data_folder + '/Dev/subtaskA_gold_answers.csv', index_col=0, header=None,
                        dtype=np.float32)
    y_test = pd.read_csv(path_to_data_folder + '/Test/subtaskA_gold_answers.csv', index_col=0, header=None,
                         dtype=np.float32)

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


x_df_to_numpy = lambda x: np.array(x[['sent0', 'sent1']])


class ComVEDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, x_transforms=[], y_transforms=[], lazy=False):
        """
        Creates a dataset from the given data. If given data is in pandas.DataFrame objects,
        they are transformed into numpy arrays before performing other given transformations.

        :param x: (Union(numpy.ndarray, pandas.DataFrame)) array containing pairs of sentences
        :param y: (Union(numpy.ndarray, pandas.DataFrame)) array containing indexes of incorrect sentences
        :param x_transforms: (iterable[callable]) functions to transform a single pair of sentences.
        Default is empty list
        :param y_transforms: (iterable[callable]) functions to transform the index. Default is empty list
        :param lazy: (boolean) flag indicating if the transformations should be done lazy. Default is False
        """
        super(ComVEDataset, self).__init__()
        assert len(x) == len(y), "Length of x and y must match."

        if isinstance(x, pd.DataFrame):
            x = x_df_to_numpy(x)
        if isinstance(y, pd.DataFrame):
            y = np.array(y)

        if lazy is False:
            for x_transform in x_transforms:
                x = [x_transform(x_) for x_ in x]
            for y_transform in y_transforms:
                y = [y_transform(y_) for y_ in y]

        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        self.lazy = lazy
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        if self.lazy is True:
            for x_transform in self.x_transforms:
                x = x_transform(x)
            for y_transform in self.y_transforms:
                y = y_transform(y)

        return x, y

    @classmethod
    def from_csv(cls, path_to_x, path_to_y, x_transforms=[], y_transforms=[], lazy=False):
        """
        :param path_to_x: (str) path to the csv file from which pairs of sentences are read
        :param path_to_y: (str) path to the csv file from which labels of sentences are read
        :param x_transforms: (iterable[callable]) functions to transform a single pair of sentences.
            Default is empty list
        :param y_transforms: (iterable[callable]) functions to transform the index. Default is empty list
        :param lazy: (boolean) flag indicating if the transformations should be done lazy. Default is False
        :return: (ComVEDataset) dataset read from the given files
        """
        x = pd.read_csv(path_to_x, index_col=0)
        y = pd.read_csv(path_to_y, index_col=0, header=None, dtype=np.float32)
        return cls(x, y, x_transforms, y_transforms, lazy)

    @classmethod
    def from_data_folder(cls, path_to_data_folder='/content/gdrive/MyDrive/data',
                         x_transforms=[], y_transforms=[], lazy=False):
        """
        Load train, dev and test data into (ComVEDataset) datasets from given directory.
        Check examples.py

        :param path_to_data_folder: (str) path to the folder contain train, dev and test data. If not specified the
        default
            path is used: /content/gdrive/MyDrive/data
        :param x_transforms: (iterable[callable]) functions to transform a single pair of sentences.
            Default is empty list
        :param y_transforms: (iterable[callable]) functions to transform the index. Default is empty list
        :param lazy: (boolean) flag indicating if the transformations should be done lazy. Default is False
        :return: (tuple[ComVEDataset]): train dataset, dev dataset, test dataset
        """
        X_train, X_dev, X_test, y_train, y_dev, y_test = load_data(path_to_data_folder)
        return (
            cls(X_train, y_train, x_transforms, y_transforms, lazy),
            cls(X_dev, y_dev, x_transforms, y_transforms, lazy),
            cls(X_test, y_test, x_transforms, y_transforms, lazy)
        )

    @classmethod
    def gpt2_lm_plain(
            cls,
            data_folder='/content/gdrive/MyDrive/data',
            training_x='/GPT2_data_final_log.csv',
            training_y='/GPT2_answers_final_log.csv',
            x_transforms=[],
            y_transforms=[],
            lazy=False,
            aug_size=10000):
        X_train_lm, X_dev, X_test, y_train_lm, y_dev, y_test = load_data2(data_folder, training_x, training_y)
        X_train_plain, X_dev, X_test, y_train_plain, y_dev, y_test = load_data2(data_folder)

        X_train_plain = X_train_plain[['sent0', 'sent1']]

        X_train_lm = X_train_lm.head(aug_size)
        y_train_lm = y_train_lm.head(aug_size)

        X_train = pd.concat([X_train_plain, X_train_lm], axis=0)
        y_train = pd.concat([y_train_plain, y_train_lm], axis=0)

        return (
            cls(X_train, y_train, x_transforms, y_transforms, lazy),
            cls(X_dev, y_dev, x_transforms, y_transforms, lazy),
            cls(X_test, y_test, x_transforms, y_transforms, lazy)
        )

    @classmethod
    def gpt2_lm(
            cls,
            data_folder='/content/gdrive/MyDrive/data',
            training_x='/GPT2_data_final.csv',
            training_y='/GPT2_answers_final.csv',
            x_transforms=[],
            y_transforms=[],
            lazy=False):
        X_train, X_dev, X_test, y_train, y_dev, y_test = load_data2(data_folder, training_x, training_y)
        return (
            cls(X_train, y_train, x_transforms, y_transforms, lazy),
            cls(X_dev, y_dev, x_transforms, y_transforms, lazy),
            cls(X_test, y_test, x_transforms, y_transforms, lazy)
        )

    @classmethod
    def gpt2(
            cls,
            data_folder='/content/gdrive/MyDrive/data',
            training_x='/GPT2_subtaskA_data.csv',
            training_y='/GPT2_answers.csv',
            x_transforms=[],
            y_transforms=[],
            lazy=False):
        X_train, X_dev, X_test, y_train, y_dev, y_test = load_data2(data_folder, training_x, training_y)
        return (
            cls(X_train, y_train, x_transforms, y_transforms, lazy),
            cls(X_dev, y_dev, x_transforms, y_transforms, lazy),
            cls(X_test, y_test, x_transforms, y_transforms, lazy)
        )

    @classmethod
    def gpt2_final(
            cls,
            data_folder='/content/gdrive/MyDrive/data',
            training_x='/GPT2_data_final.csv',
            training_y='/GPT2_answers_final.csv',
            x_transforms=[],
            y_transforms=[],
            lazy=False):
        X_train, X_dev, X_test, y_train, y_dev, y_test = load_data2(data_folder, training_x, training_y)
        return (
            cls(X_train, y_train, x_transforms, y_transforms, lazy),
            cls(X_dev, y_dev, x_transforms, y_transforms, lazy),
            cls(X_test, y_test, x_transforms, y_transforms, lazy)
        )

    @classmethod
    def gpt2_final(
            cls,
            data_folder='/content/gdrive/MyDrive/data',
            training_x='/GPT2_data_final_log.csv',
            training_y='/GPT2_answers_final_log.csv',
            x_transforms=[],
            y_transforms=[],
            lazy=False):
        X_train, X_dev, X_test, y_train, y_dev, y_test = load_data2(data_folder, training_x, training_y)
        return (
            cls(X_train, y_train, x_transforms, y_transforms, lazy),
            cls(X_dev, y_dev, x_transforms, y_transforms, lazy),
            cls(X_test, y_test, x_transforms, y_transforms, lazy)
        )

    @classmethod
    def back_translation(
            cls,
            data_folder='/content/gdrive/MyDrive/data',
            training_x='backtranslation_data3.csv',
            training_y='backtranslation_labels.csv',
            x_transforms=[],
            y_transforms=[],
            lazy=False):
        X_train_lm, X_dev, X_test, y_train_lm, y_dev, y_test = load_data2(data_folder, training_x, training_y)
        X_train_plain, X_dev, X_test, y_train_plain, y_dev, y_test = load_data2(data_folder)

        X_train_plain = X_train_plain[['sent0', 'sent1']]

        # X_train_lm = X_train_lm.head(aug_size)
        # y_train_lm = y_train_lm.head(aug_size)

        X_train = pd.concat([X_train_plain, X_train_lm], axis=0)
        y_train = pd.concat([y_train_plain, y_train_lm], axis=0)

        return (
            cls(X_train, y_train, x_transforms, y_transforms, lazy),
            cls(X_dev, y_dev, x_transforms, y_transforms, lazy),
            cls(X_test, y_test, x_transforms, y_transforms, lazy)
        )
