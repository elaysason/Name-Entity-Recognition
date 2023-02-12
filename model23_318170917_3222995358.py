import numpy as np
from gensim import downloader
import re
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from torch import nn
import torch
import torch.nn.functional as F

torch.manual_seed(25)

# path to pre-trained embedding
WORD_2_VEC_PATH = 'word2vec-google-news-300'
GLOVE_PATH = 'glove-twitter-200'


class F1_Loss(nn.Module):
    """ Calculate F1 score and used as a loss function for the model. """

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, ):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)

        y_pred = F.softmax(y_pred, dim=1)
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


class FC(nn.Module):
    """ neural network model class """

    def __init__(self, vec_dim, num_classes, hidden_dim=70):
        # define the network parts
        super(FC, self).__init__()
        self.first_layer = nn.Linear(vec_dim, int(hidden_dim))
        self.second_layer = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.third_layer = nn.Linear(int(hidden_dim / 2), num_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.6)
        self.loss = F1_Loss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, labels=None):
        # run the data in the network
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


class MyDataset(Dataset):
    """" Build the data with embedding of glove or V2W according to vector_type parameter
    (with content), preprocess the data for the model """

    def __init__(self, file_path, vector_type, tokenizer=None):
        self.file_path = file_path
        with open(self.file_path, encoding="utf8") as f:
            contents = f.read()
        self.vector_type = vector_type
        if vector_type == 'w2v':
            model = downloader.load(WORD_2_VEC_PATH)
        elif vector_type == 'glove':
            model = downloader.load(GLOVE_PATH)
        else:
            raise KeyError(f"{vector_type} is not a supported vector type")
        sentence_all_words = []
        sentence_all_tags = []
        sentence_words = []
        sentence_tags = []
        just_words = []
        just_tags = []

        for row in contents.split('\n'):
            if not row.strip():
                sentence_all_words.append(sentence_words)
                sentence_all_tags.append(sentence_tags)
                sentence_words = []
                sentence_tags = []

            elif row.split('\t') != ['\ufeff']:
                if row.split('\t')[1] == 'O':
                    cur_tag = 0
                else:
                    cur_tag = 1
                sentence_words.append(row.split('\t')[0])
                sentence_tags.append(cur_tag)
                just_tags.append(cur_tag)
                just_words.append(row.split('\t')[0])

        # representation via glove' with context
        representation = []
        for word_vec, tag_vec in zip(sentence_all_words, sentence_all_tags):
            for i, word in enumerate(word_vec):
                # current word
                word = re.sub(r'\W+', '', word.lower())
                if word not in model.key_to_index:
                    vec = model['unk']
                else:
                    vec = model[word]
                # previous word
                if i > 0:
                    pre_word = word_vec[i - 1]
                    pre_word = re.sub(r'\W+', '', pre_word.lower())
                    # pre_word = re.sub('[!^&()%@#$]', '', pre_word)
                    if pre_word not in model.key_to_index:
                        pre_rep = model['unk']
                    else:
                        pre_rep = model[pre_word]
                else:
                    pre_rep = np.zeros(len(model['unk']))
                # next word
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

                # concat to word and it's content
                context_rep = np.concatenate([pre_rep, vec, next_rep])
                representation.append(context_rep)
            if len(representation) == 0:
                print(f'Sentence {word_vec} cannot be represented!')
                continue
        self.vector_dim = len(representation[0])
        self.tokenized_sen = representation
        self.labels = just_tags

    def __getitem__(self, item):

        cur_sen = self.tokenized_sen[item]
        if self.vector_type == 'tf-idf':
            cur_sen = torch.FloatTensor(cur_sen.toarray()).squeeze()
        else:
            cur_sen = torch.FloatTensor(cur_sen).squeeze()
        label = self.labels[item]
        # label = self.tags_to_idx[label]
        data = {"input_ids": cur_sen, "labels": label}
        return data

    def __len__(self):
        return len(self.labels)


def train(model, data_sets, optimizer, num_epochs: int, batch_size=100):
    " Training the model, checking its performance and saving the best model"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
                    "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False)}
    model.to(device)

    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            labels, preds = [], []

            for batch in data_loaders[phase]:
                batch_size = 0
                for k, v in batch.items():
                    batch[k] = v.to(device)
                    batch_size = v.shape[0]

                optimizer.zero_grad()
                if phase == 'train':
                    outputs, loss = model(**batch)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs, loss = model(**batch)
                pred = outputs.argmax(dim=-1).clone().detach().cpu()

                labels += batch['labels'].cpu().view(-1).tolist()
                real_ratio = sum(labels) / len(labels)
                preds += pred.view(-1).tolist()
                # adaption to 'upsample' the minority class
                running_loss += loss.item() * batch_size * (1 + 100 * real_ratio)

            epoch_loss = running_loss / len(data_sets[phase])
            epoch_f1 = f1_score(labels, preds)

            if phase.title() == "test":
                print(f'{phase.title()} Loss: {epoch_loss:.4e} F1: {epoch_f1}')
            else:
                print(f'{phase.title()} Loss: {epoch_loss:.4e} F1: {epoch_f1}')
            # comparing current epoch f1 score against best one, and updating the value
            # and saving the model if its better
            if phase == 'test' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                torch.save(model.state_dict(), 'comp_model')

        print()

    print(f'Best Validation F1: {best_f1:4f}')

# get train set
train_ds = MyDataset('data/train.tagged', vector_type='glove')
print('created train')

#get validation set
test_ds = MyDataset('data/dev.tagged', vector_type='glove')

# define a run the model
datasets = {"train": train_ds, "test": test_ds}
model = FC(num_classes=2, vec_dim=train_ds.vector_dim)
optimizer = Adam(params=model.parameters())
train(model=model, data_sets=datasets, optimizer=optimizer, num_epochs=45)
