import re
from collections import Counter


class Vocabulary:
    def __init__(self, freq_threshold = 5):
        self.pad = 0
        self.sos = 1
        self.eos = 2
        self.unk = 3
        self.stoi = {
            '<PAD>': self.pad,
            '<SOS>': self.sos,
            '<EOS>': self.eos,
            '<UNK>': self.unk
        }
        self.itos = {
            self.pad: '<PAD>',
            self.sos: '<SOS>',
            self.eos: '<EOS>',
            self.unk: '<UNK>'
        }
        self.idx = 4
        self.freq_threshold = freq_threshold

    def tokenize(self, string):
        return re.findall(r'\w+', string.lower())

    def build_vocabulary(self, captions):
        words = []
        for sentence in captions:
            words.extend(self.tokenize(sentence))

        counter = Counter(words)

        for k, v in counter.items():
            if v >= self.freq_threshold:
                self.stoi[k] = self.idx
                self.itos[self.idx] = k
                self.idx += 1
            else:
                continue
        

    def numericalize(self, sentence:str):
        words = self.tokenize(sentence)
        tokenized = [self.sos]
        for word in words:
            if word in self.stoi:
                tokenized.append(self.stoi[word])
            else:
                tokenized.append(self.unk)
        tokenized.append(self.eos)

        return tokenized

    def __len__(self):
        return len(self.stoi)