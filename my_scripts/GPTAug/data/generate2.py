"""
@uthor: Prakhar
"""
import os
import argparse

import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


def choose_from_top_k_top_n(probs, k=0, p=0.9):
    ind = np.argpartition(probs, -k)[-k:]
    top_prob = probs[ind]
    top_prob = {i: top_prob[idx] for idx, i in enumerate(ind)}
    sorted_top_prob = {k: v for k, v in sorted(top_prob.items(), key=lambda item: item[1], reverse=True)}

    t = 0
    f = []
    pr = []
    for k, v in sorted_top_prob.items():
        t += v
        f.append(k)
        pr.append(v)
        if t >= p:
            break
    top_prob = pr / np.sum(pr)
    token_id = np.random.choice(f, 1, p=top_prob)

    return int(token_id)


def generate(tokenizer, model, sentences, label):
    s = []
    with torch.no_grad():
        for idx in tqdm(range(sentences)):
            finished = False
            model.to('cuda:0')

            cur_ids = torch.tensor(tokenizer.encode(label)).unsqueeze(0).to('cpu')

            for i in range(100):
                outputs = model(cur_ids.cuda(), labels=cur_ids.cuda())
                loss, logits = outputs[:2]

                softmax_logits = torch.softmax(logits[0, -1], dim=0)

                if i < 5:
                    n = 10
                else:
                    n = 5

                next_token_id = choose_from_top_k_top_n(softmax_logits.to('cpu').numpy())  # top-k-top-n sampling
                cur_ids = torch.cat([cur_ids.to('cuda'), torch.ones((1, 1)).long().to('cuda') * next_token_id], dim=1)

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    finished = True
                    break

            if finished:
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                print(output_text.split(' '))
                s.append(' '.join(output_text.split(' ')[2:-2]))
            else:
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                print(output_text.split(' '))
                s.append(' '.join(output_text.split(' ')[2:-2]))

    pd.DataFrame({'sentence': s}).to_csv('GPT2_medium_aug_sense', index_label='id')


def load_models(model_name):
    """
    Summary:
        Loading the trained model
    """
    print('Loading Trained GPT-2 Modell')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    model_path = model_name
    model.load_state_dict(torch.load(model_path))
    return tokenizer, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for inferencing Text Augmentation model')

    parser.add_argument('--model_name', default='D:\\git\\fer-tar\\mymodel.pt.pt', type=str, action='store', help='Name of the model file')
    parser.add_argument('--sentences', type=int, default=5000, action='store', help='Number of sentences in outputs')
    parser.add_argument('--label', type=str, action='store', help='Label for which to produce text')
    args = parser.parse_args()

    SENTENCES = args.sentences
    MODEL_NAME = args.model_name
    LABEL = '0'

    TOKENIZER, MODEL = load_models(MODEL_NAME)

    generate(TOKENIZER, MODEL, SENTENCES, LABEL)