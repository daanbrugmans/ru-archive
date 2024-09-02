
import torch
import copy
from functions import word_2_index

def read_sentences(items_loc, item_set, index_dict):
    
    sent = []
    with open(items_loc + item_set + '.txt', encoding='utf-8') as file:
        for line in file:
            # split the sentence into tokens
            sent.append(['<s>'] + line.lower().split() + ['</s>'])

    # Convert text to indices. Then identify sentences with <oov> items that are not padding, i.e., a 0 that is not trailing
    sent, sentlengths = word_2_index(sent, len(sent), index_dict)  
    oov_sent = []
    for this_sent in list(sent):
        while this_sent[-1]==0:
            this_sent = this_sent[:-1]
        oov_sent += [0 in this_sent]
    
    return sent, oov_sent, sentlengths

# Function definitions for producing the surprisals and RNN gradients for the test sentences
def calc_surprisal(items_loc, item_set, index_dict, nwp_model):

    sent, oov_sent, sentlengths = read_sentences(items_loc, item_set, index_dict)
    
    # get the predictions and targets for this sentence
    predictions, targets = nwp_model(torch.FloatTensor(sent), sentlengths)

    # convert the predictions to surprisal (negative log softmax)
    surprisal = -torch.log_softmax(predictions, dim = 2).squeeze()
    # extract only the surpisal ratings for the target words. Set to None if sentence contains <oov>
    surprisal = surprisal.gather(-1, targets.unsqueeze(-1)).squeeze()
    # remove any padding applied by word_2_index and remove end-of-sentence prediction
    surprisal = surprisal.data.numpy()
    surprisal[oov_sent] = None
    surprisal = [s[:l-2] for s, l in zip(surprisal, sentlengths)]    

    return surprisal

def one_grad(model, loss_fun, sentence, sentlength):
    # Return the gradient at each word
    predictions, target = model.forward(torch.FloatTensor([list(sentence)]), [sentlength]) 

    # Repeatedly process the sentence, incrementally from word 1 up to word w. 
    # The gradients on word w are the sentence-prefix gradients for words 1 to w minus the gradients for 1 to w-1
    grads = [0.0]
    for w in range(1, sentlength):
        loss = loss_fun(predictions[0][:w].view(-1, predictions[0][:w].size(-1)), target[0][:w].view(-1)[:w])
        loss.backward(retain_graph=True)
        these_grads = model.RNN.weight_hh_l0.grad     # size of grads is hidden_size x hidden_size for SRN, (4 x hidden_size) x hidden_size for LSTM (because 4 units per LSTM cell)
        grads += [copy.deepcopy(these_grads)]
        # zero the gradients from previous run
        for param in model.parameters():
            param.grad = None
    
    return [float(sum(sum(abs(grads[n+1] - grads[n])))) for n in range(sentlength-2)] 

def calc_gradient(items_loc, item_set, index_dict, nwp_model):
    # Calculates gradients on recurrent units. 

    sent, oov_sent, sentlengths = read_sentences(items_loc, item_set, index_dict)

    # get the predictions and targets for this sentence, and backpropagate the error
    loss_fun = torch.nn.CrossEntropyLoss(ignore_index= 0, reduction='sum')
    grad = []
    for s in range(len(sent)):
        if oov_sent[s]:
            grad += [[None] * (sentlengths[s]-2)]     # subtract 2 from sentlengths because no gradient for <s> and </s> will be reported
        
        else:
            these_grads = one_grad(nwp_model, loss_fun, sent[s], sentlengths[s])
            grad += [these_grads]

    return grad
