from transformers import BertTokenizer, AlbertTokenizer, ElectraTokenizer, RobertaTokenizer


class SentencePairEncoder:
    def __call__(self, sentences):
        sent0 = self.encode(sentences[0])
        sent1 = self.encode(sentences[1])
        return sent0, sent1

    def encode(self, sentence):
        pass


class TransformerEncoder(SentencePairEncoder):
    def __init__(self, tokenizer):
        super(TransformerEncoder, self).__init__()
        self.tokenizer = tokenizer
        # print('tokenizer is None:', tokenizer is None)

    def encode(self, sentence):
        dictionary = self.tokenizer(
            sentence,  # Sentences to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=32,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        # print('dictionary is None:', dictionary is None)
        for k, v in dictionary.items():
            dictionary[k] = v.reshape(-1)
        return dictionary


class BertEncoder(SentencePairEncoder):
    def __init__(self, name='bert-base-uncased'):
        super(BertEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(name)

    def __call__(self, sentence_pair):
        sent0_dict = self.encode(sentence_pair[0])
        sent1_dict = self.encode(sentence_pair[1])
        return sent0_dict, sent1_dict

    def encode(self, sentence):
        dictionary = self.tokenizer(
            sentence,  # Sentences to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=32,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        for k, v in dictionary.items():
            dictionary[k] = v.reshape(-1)
        return dictionary


class AlbertEncoder(TransformerEncoder):
    def __init__(self, name='albert-base-v1'):
        super(AlbertEncoder, self).__init__(AlbertTokenizer.from_pretrained(name))


class ElectraEncoder(TransformerEncoder):
    def __init__(self, name='google/electra-small-discriminator'):
        super(ElectraEncoder, self).__init__(ElectraTokenizer.from_pretrained(name))


class RobertaEncoder(TransformerEncoder):
    def __init__(self, name='roberta-base'):
        super(RobertaEncoder, self).__init__(RobertaTokenizer.from_pretrained(name))
