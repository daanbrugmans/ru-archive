import torch.nn as nn

class nwp_rnn(nn.Module):
    def __init__(self, config):
        super(nwp_rnn, self).__init__()
        embed = config['embed']
        rnn = config['rnn']
        lin = config['lin']

        self.is_cuda = config['cuda']
        self.max_len = config['max_len']
        
        self.embed = nn.Embedding(num_embeddings = embed['n_embeddings'], 
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])

        if rnn['type'].lower() == 'srn':
            self.RNN = nn.RNN(input_size = rnn['in_size'], 
                          hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['n_layers'],
                          batch_first = rnn['batch_first'],
                          dropout = rnn['dropout'])
        if rnn['type'].lower() == 'lstm':
            self.RNN = nn.LSTM(input_size = rnn['in_size'], 
                          hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['n_layers'],
                          batch_first = rnn['batch_first'],
                          dropout = rnn['dropout'])

        self.linear = nn.Sequential(nn.Linear(rnn['hidden_size'], 
                                              lin['hidden_size']
                                              ), 
                                    nn.Tanh(), 
                                    nn.Linear(lin['hidden_size'],
                                              embed['n_embeddings']
                                              )
                                    )

    def forward(self, input, sent_lens):
        targs      = nn.functional.pad(input[:,1:], [0,1]).long()    # create the targets by shifting the input left
        embeddings = self.embed(input.long())

        x       = nn.utils.rnn.pack_padded_sequence(embeddings, sent_lens, batch_first = True, enforce_sorted = False)
        x, hx   = self.RNN(x)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)

        out = self.linear(x)  
        
        return out, targs
