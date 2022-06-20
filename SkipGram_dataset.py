import torch
import collections
import numpy as np

class SkipGram_Dataset(torch.utils.data.Dataset):
    def __init__(self, contexts, target, corpus, power = 0.75, sample_size = 5):
        self.target = target
        self.contexts = np.empty((0, len(contexts[0]) + sample_size), np.int32)
        self.labels = np.empty((len(contexts), len(contexts[0]) + sample_size), np.int32)
        sampler = UnigramSampler(corpus, power, sample_size)

        for idx, context in enumerate(contexts):
            negative_sample = sampler.get_negative_sample(target[idx])

            self.contexts = np.append(self.contexts, np.append(context, negative_sample).reshape(1, len(context) + sample_size), axis=0)

            for j in range(len(context)):
                self.labels[idx][j] = 1

    def __len__(self):
        return len(self.contexts)
        
    def __getitem__(self, idx):
        return (self.contexts[idx], self.target[idx]), self.labels[idx]


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1


        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):

        negative_sample = np.zeros(self.sample_size, dtype=np.int32)
        p = self.word_p.copy()
        target_idx = target
        p[target_idx] = 0
        p /= p.sum()
        negative_sample[:] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)

        return negative_sample