from multiprocessing import reduction
from torch import nn
import torch
import torch.nn.functional as F

class co_matrix_add_SkipGram(nn.Module):
    
    def __init__(self, vocab_size, hidden_size, co_matrix):
        super(co_matrix_add_SkipGram, self).__init__()
        self.input_layer = nn.Linear(vocab_size, hidden_size, bias = False)
        self.target_layer = nn.Linear(vocab_size, hidden_size, bias = False)
        self.co_matrix = nn.Embedding.from_pretrained(co_matrix, freeze=True)
        self.vocab_size = vocab_size

    def forward(self, data):

        contexts, target = data
        contexts, target = contexts.type(torch.long), target.type(torch.long)

        target_vector = F.one_hot(target, num_classes=self.vocab_size) + self.co_matrix(target)
        target_out = self.target_layer(target_vector)

        contexts_out = []
        
        for i in range(len(contexts[0])):

            contexts_vector = F.one_hot(contexts[:,i], num_classes = self.vocab_size) + self.co_matrix(contexts[:,i])
            contexts_out.append(self.input_layer(contexts_vector))

        return (contexts_out, target_out)


class BCE_loss_func(nn.Module):
    def __init__(self):
        super(BCE_loss_func, self).__init__()
        self.sig = nn.Sigmoid()
        self.loss_func = nn.BCELoss(reduction=None)

    def forward(self, data, label):
        contexts, target = data
        label = label.type(torch.float)

        loss = torch.zeros(len(contexts))

        for i, context in enumerate(contexts):            
            k = torch.mul(context, target)
            out = torch.sum(k, 1)
            out = self.sig(out)
            loss += self.loss_func(out, label[:, i])    

        return loss.mean()