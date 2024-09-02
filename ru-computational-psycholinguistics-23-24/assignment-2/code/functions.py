import pickle
from collections import defaultdict
import numpy as np
#import pickle5 as pickle

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, loc):
    with open(str(loc) + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL) 

def word_2_index(batch, batch_size, w_dict):
    # filter words that do not occur in the dictionary, these will become coded as 0s
    batch = [[word if word in w_dict else '<oov>' for word in sent] for sent in batch]
    # get the maximum sentence length for this batch
    max_sent_len = max([len(x) for x in batch])
    index_batch  = np.zeros([batch_size, max_sent_len]) 
    # keep track of the unpadded sentence lengths. 
    lengths = []
    # load the indices for the words from the dictionary
    for i, words in enumerate(batch):
        lengths.append(len(words))
        for j, word in enumerate(words):
            index_batch[i][j] = w_dict[word]
    return index_batch, lengths

def index_2_word(dictionary, indices):
    rev_dictionary = defaultdict(str)
    for x, y in dictionary.items():
        rev_dictionary[y] = x
    sentences = [[rev_dictionary[int(i)] for i in ind] for ind in indices]   
    return(sentences)

def get_rnn_config(dict_size, cuda):
           
    config_rnn = {'embed':{'n_embeddings': dict_size, 'embedding_dim': 400, 
                'sparse': False, 'padding_idx': 0
                }, 
            'max_len': 52,
            'rnn':{'in_size': 400, 'hidden_size': 500, 'n_layers': 1, 
                'batch_first': True, 'bidirectional': False, 'dropout': 0,
                'type': 'lstm'
                }, 
            'lin':{'hidden_size': 400},
            'cuda': cuda
            }
    return config_rnn