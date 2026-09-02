import re

class Vocabulary:
    def __init__(self):
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
        self.freq_threshold = 4

    def tokenize(self, string):
        return re.findall(r'\w+', string.lower())

    def build_vocabulary(self, captions):
        for sentence in captions:
            words = self.tokenize(sentence)

            for word in words:
                if word in self.stoi:
                    continue
                else:
                    self.stoi[word] = self.freq_threshold
                    self.itos[self.freq_threshold] = word
                    self.freq_threshold += 1

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