import pandas as pd
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

import time
# def create_augmented_datasets(data, augmenters, augment_func):
#     for augmenter in augmenters:


def augment_A(A_df, augmenter):
    """
    Function that augments subtask A data
    :param A_df: subtask A data
    :param augmenter: augmenter used to augment data
    :return: pandas DataFrame with augmented data
    """

    s1 = [augmenter.augment(sent) for sent in A_df['sent0'].values]
    s2 = [augmenter.augment(sent) for sent in A_df['sent1'].values]

    return pd.DataFrame({'sent0': s1, 'sent1': s2})

def augment_data(data, aug_func, augmenters, dataset_folder_path, n_datasets, start_index=0):
    for aug in augmenters:
        for i in range(start_index, n_datasets + start_index):
            df = aug_func(data, aug)
            df.to_csv(f'{dataset_folder_path}/{aug.name}_{i}.csv', index_label='id')
            print(f'Done with {i}. {aug.name}')


if __name__ == '__main__':
    A = pd.read_csv('subtaskA_data_all.csv', index_col=0)
    augs = [naw.ContextualWordEmbsAug(model_path='bert-base-cased', action="substitute", aug_min=1, aug_max=1, top_k=30, temperature=0.2)]

    augment_data(A, augment_A, augs, '/media/maculjak/31879BCA74C36D17/data', 5)
