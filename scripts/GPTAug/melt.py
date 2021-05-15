from scripts.data import melt_columns
import pandas as pd


if __name__ == '__main__':
    sentences = pd.read_csv('data/subtaskA_data_all.csv', index_col=0)
    labels = pd.read_csv('data/subtaskA_answers_all.csv', index_col=0, header=None)
    labels.columns = ['label']
    X, y = melt_columns(sentences[['sent0', 'sent1']], labels['label'])
    X = pd.DataFrame(X)


    X.insert(1, 'label', list(y))
    y.to_csv('A_labels_melted.csv', index_label='id', header=['label'])
    X.to_csv('A_sentences_melted.csv', index_label='id', header=['sentence', 'label'])
