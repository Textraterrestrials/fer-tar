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


def choose_from_top_k_top_n(probs, k=100, p=0.9):
    ind = np.argpartition(probs, -k)[-k:]
    top_prob = probs[ind]
    top_prob = {i: top_prob[idx] for idx, i in enumerate(ind)}
    sorted_top_prob = {k: v for k, v in sorted(top_prob.items(), key=lambda item: item[1], reverse=False)}

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


def generate(tokenizer, model, sentences):
    s = []
    with torch.no_grad():
        for idx in tqdm(range(sentences)):
            k = 50
            p = 0.9
            finished = False
            model.to('cuda:0')
            cur_ids = torch.tensor(tokenizer.encode('S:')).unsqueeze(0).to('cpu')

            for i in (range(100)):
                outputs = model(cur_ids.cuda(), labels=cur_ids.cuda())
                loss, logits = outputs[:2]

                softmax_logits = torch.softmax(logits[0, -1], dim=0)

                if i < 5:
                    n = 10
                else:
                    n = 5

                next_token_id = choose_from_top_k_top_n(softmax_logits.to('cpu').numpy(), k=k,
                                                        p=p)  # top-k-top-n sampling
                cur_ids = torch.cat([cur_ids.to('cuda'), torch.ones((1, 1)).long().to('cuda') * next_token_id], dim=1)

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    finished = True
                    break
                elif next_token_id in tokenizer.encode('/') or next_token_id in tokenizer.encode('./'):
                    k = 100
                    p = 0.9

            if finished:
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                s.append(output_text[2:-13])
            else:
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                # s.append(' '.join(output_text.split(' ')[1:-1]))
                s.append(output_text[2:-13])
            if idx % 1000 == 0:
                pd.DataFrame({'sentence': s}).to_csv('GPT2_medium_aug_sense4', index_label='id')

        print(s)
        pd.DataFrame({'sentence': s}).to_csv('GPT2_medium_aug_sense4', index_label='id')


def load_model(model_name):
    """
    Summary:
        Loading the trained model
    """
    print('Loading Trained GPT-2 Model')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

    # model_path = model_name
    model_path = '/content/gdrive/MyDrive/fer-tar/gpt_medium_double_aug.pt'
    model.load_state_dict(torch.load(model_path))
    return tokenizer, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for inferencing Text Augmentation model')

    parser.add_argument('--model_name', default='double_aug.pt.pt', type=str, action='store',
                        help='Name of the model file')
    parser.add_argument('--sentences', type=int, default=5, action='store', help='Number of sentences in outputs')
    parser.add_argument('--label', type=str, action='store', help='Label for which to produce text')
    args = parser.parse_args()

    SENTENCES = args.sentences
    MODEL_NAME = args.model_name
    LABEL = '0'

    TOKENIZER, MODEL = load_models(MODEL_NAME)
    MODEL.cuda()
    TOKENIZER.add_special_tokens({'sep_token': '<SEP>'})
    print(TOKENIZER.__len__())
    MODEL.resize_token_embeddings(len(TOKENIZER))
    generate(TOKENIZER, MODEL, SENTENCES)
