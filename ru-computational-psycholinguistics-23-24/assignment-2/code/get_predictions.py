import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from functions import *
from nwp_rnn import nwp_rnn
from calc_preds import *
    
def get_predictions(item_sets: list[str], language: str, measure: str, cuda: bool):
    home = 'C:/Users/Daan/Documents/Projecten/ru-computational-psycholinguistics-23-24/assignment-2/code'

    # item_sets = ['stimuli']
    # language  = 'en'     # Change to 'nl' when testing Dutch sentences
    # measure   = 'surp'   # Change to 'grad' to get gradient measures

    epochs = ['10000', '30000', '100000', '300000', '1000000', '3000000', '10000000', '30000000', 'epoch1']
    # cuda   = True

    model_loc = home + '/trained_models/nwp_model_lstm_{}_{}' 
    items_loc = home + '/items/'
    dict_loc  = home + '/vocab/train_indices_{}' 
    pred_loc  = home + '/predictions/' + measure + '_{}.csv'
    
    # load the dictionary of indexes and create a reverse lookup dictionary so
    # we can look up target words by their index
    index_dict = load_obj(dict_loc.format(language))
    dict_size  = len(index_dict) + 1
    word_dict  = defaultdict(str)
    for x, y in index_dict.items():
        word_dict[y] = x

    config_rnn = get_rnn_config(dict_size, cuda)

    for item_set in item_sets:
        print(item_set)
        data = pd.DataFrame()

        for ep in epochs:
            model_path = model_loc.format(language, ep)
            print(model_path)
            
            ###############################################################################
            # load the pretrained model
            model     = torch.load(model_path, map_location = 'cpu')
            nwp_model = nwp_rnn(config_rnn)
            nwp_model.load_state_dict(model)

            # set to eval mode to disable dropout
            nwp_model.eval()

            # get all the surprisal or gradient values
            if measure == "surp":
                preds = calc_surprisal(items_loc, item_set, index_dict, nwp_model)
            elif measure == "grad":
                preds = calc_gradient(items_loc, item_set, index_dict, nwp_model)
            else:
                raise ValueError("Unknown measure type:" + measure)

            # add sentence and word position indices to the predicted values and 
            # convert to DataFrame object
            preds = [(sent_index + 1, word_index +1, pred) for sent_index, sent in enumerate(preds) for word_index, pred in enumerate(sent)]
            preds = pd.DataFrame(np.array(preds))

            # create a unique name for the current predicted values 
            if measure == "surp":
                pred_name = 'surp_{}'.format(ep)
            else:
                pred_name = 'grad_{}'.format(ep)
            preds.columns = ['sent_nr', 'word_pos', pred_name]

            item_nr = []
            for x, y in zip(preds.sent_nr, preds.word_pos):    
                x = x*100
                item_nr.append(int(x+y))
                
            preds['item'] = pd.Series(item_nr)
            if not data.empty:
                data[pred_name] = data.join(preds[[pred_name, 'item']].set_index('item'), on = 'item')[pred_name]
            else:
                data = preds

        ###############################################################################
        # now sort the column names 

        col_names = data.columns.tolist()
        models    = col_names[2:3] + col_names[4:]
        col_names = col_names[0:2] + col_names[3:4] + models
        data      = data[col_names]

        # round the surprisal to 6 decimals and convert the sent_nr and word_pos from 
        # float to int
        data[models]  = data[models].round(6)
        data.sent_nr  = data.sent_nr.astype(int)
        data.word_pos = data.word_pos.astype(int)

        data.to_csv(path_or_buf = pred_loc.format(item_set), sep='\t')
    