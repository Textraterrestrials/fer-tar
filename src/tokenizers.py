from transformers import AutoTokenizer

from src.models import MODELS


class __SentencePairTokenizer:
    def __call__(self, sentences_pair):
        sent0_tokenized = self.tokenize_sent(sentences_pair[0])
        sent1_tokenized = self.tokenize_sent(sentences_pair[1])
        return sent0_tokenized, sent1_tokenized

    def tokenize_sent(self, sentence):
        pass


class TransformerTokenizer(__SentencePairTokenizer):
    def __init__(self, name='bert', **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(MODELS[name])
        self.max_length = kwargs.get('max_length', 32)

    def tokenize_sent(self, sentence):
        dictionary = self.tokenizer(
            sentence,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True
        )
        for k, v in dictionary.items():
            dictionary[k] = v.reshape(-1)
        return dictionary
