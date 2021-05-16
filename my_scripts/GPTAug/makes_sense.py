import pandas as pd
import numpy as np

if __name__ == '__main__':
    sentences = pd.read_csv('../../subtaskA_data_all.csv', index_col=0)
    labels = pd.read_csv('../../subtaskA_answers_all.csv', index_col=0, header=None)

    sentences['label'] = labels
    sentences['sent0'], sentences['sent1'] = np.where(sentences['label'] == 1, (sentences['sent1'], sentences['sent0']),
                                                 (sentences['sent0'], sentences['sent1']))

    sentences['sents'] = sentences['sent1']

    sentences.pop('sent0')
    sentences.pop('sent1')
    sentences.pop('label')


    sentences.to_csv('only_sense.csv', sep=';')