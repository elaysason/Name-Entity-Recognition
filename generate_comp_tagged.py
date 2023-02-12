import re
import torch
import numpy as np
from gensim import downloader
from torch import nn


class FC(nn.Module):
    """ neural network model for predicting """

    def __init__(self, vec_dim, num_classes, hidden_dim=70):
        super(FC, self).__init__()
        self.first_layer = nn.Linear(vec_dim, int(hidden_dim))
        self.second_layer = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.third_layer = nn.Linear(int(hidden_dim / 2), num_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.6)
        # self.loss = F1_Loss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, labels=None):
        x = self.first_layer(input_ids)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.second_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.third_layer(x)
        x = self.softmax(x)
        if labels is None:
            return x, None
        loss = self.loss(x, labels)
        return x, loss

# paths to files
comp_model_path = 'model.pkl'
WORD_2_VEC_PATH = 'word2vec-google-news-300'
GLOVE_PATH = 'glove-twitter-200'

# load the glove
pre_model = FC(num_classes=2, vec_dim=600)
pre_model.load_state_dict(torch.load('comp_model', map_location=torch.device('cpu')))

# prepre data for network - embedding
def data_create(file_path):
    blank_sapce = []
    with open(file_path, encoding="utf8") as f:
        contents = f.read()
    model = downloader.load(GLOVE_PATH)
    sentence_all_words = []
    sentence_words = []
    just_words = []

    for inx, row in enumerate(contents.split('\n')):
        if not row.strip():
            sentence_all_words.append(sentence_words)
            sentence_words = []
            blank_sapce.append(inx)

        elif row.split('\t') != ['\ufeff']:
            sentence_words.append(row.split('\t')[0])
            just_words.append(row.split('\t')[0])

    #################################
    representation = []
    for word_vec in sentence_all_words:
        for i, word in enumerate(word_vec):
            word = re.sub(r'\W+', '', word.lower())
            if word not in model.key_to_index:
                vec = model['unk']
            else:
                vec = model[word]
            # word i-1
            if i > 0:
                pre_word = word_vec[i - 1]
                pre_word = re.sub(r'\W+', '', pre_word.lower())
                if pre_word not in model.key_to_index:
                    pre_rep = model['unk']
                else:
                    pre_rep = model[pre_word]
            else:
                pre_rep = np.zeros(len(model['unk']))

            if i < len(word_vec) - 1:
                next_word = word_vec[i + 1]
                next_word = re.sub(r'\W+', '', next_word.lower())
                # next_word = re.sub('[!^&()%@#$]', '', next_word)
                if next_word not in model.key_to_index:
                    next_rep = model['unk']
                else:
                    next_rep = model[next_word]
            else:
                next_rep = np.zeros(len(model['unk']))
            context_rep = np.concatenate([pre_rep, vec, next_rep])
            representation.append(context_rep)
        if len(representation) == 0:
            print(f'Sentence {word_vec} cannot be represented!')
            continue
    return representation, blank_sapce

rep, sen_indexs = data_create('data/test.untagged')
predictions = pre_model(torch.tensor(rep).type(torch.float32))


# get original words
with open('data/test.untagged', encoding="utf8") as f:
    contents = f.read()

# write predictions in predictions file
i = 0
pred = predictions[0].argmax(dim=-1).clone().detach().cpu()
with open('test.tagged', 'w', encoding="utf8") as file:
    for line in contents.split('\n'):
        if line.strip():
            if pred[i].item() == 0:
                file.write(line.replace("\n", "") + "\t" + "O" + "\n")
            else:
                file.write(line.replace("\n", "") + "\t" + str(pred[i].item()) + "\n")
            i += 1
        else:
            file.write("\n")


