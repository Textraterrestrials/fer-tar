import torch


class SentencePairEncoder:

    def __call__(self, sentences):
        sent0 = self.encode(sentences[0])
        sent1 = self.encode(sentences[1])
        return torch.vstack([sent0, sent1])

    def encode(self, sentence):
        pass

class FakeEncoder(SentencePairEncoder):

    def encode(self, sentence):
        return torch.randn(5)
