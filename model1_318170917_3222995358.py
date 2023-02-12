import numpy as np
from gensim import downloader
import re
from sklearn.svm import SVC
from sklearn.metrics import f1_score

#load the glove
GLOVE_PATH = 'glove-twitter-200'
model = downloader.load(GLOVE_PATH)

def create_data(path):
    """
    preprocess the data from file path
    :param path: path to data
    :return: representation of the words and their tags
    """
    sentence_all_words = []
    sentence_all_tags = []
    sentence_words = []
    sentence_tags = []
    just_words = []
    just_tags = []
    #load data from file
    with open(path, encoding="utf8") as f:
        contents = f.read()
    for row in contents.split('\n'):
        #split to sentences by empty the sentence list when seeing an empty line
        if not row.strip():
            sentence_all_words.append(sentence_words)
            sentence_all_tags.append(sentence_tags)
            sentence_words = []
            sentence_tags = []

        # if not the end of the file, change the tags to 0 and 1
        elif row.split('\t') != ['\ufeff']:
            if row.split('\t')[1] == 'O':
                cur_tag = 0
            else:
                cur_tag = 1
            sentence_words.append(row.split('\t')[0])
            sentence_tags.append(cur_tag)
            just_tags.append(cur_tag)
            just_words.append(row.split('\t')[0])

    # get representation
    representation = []
    for word_vec, tag_vec in zip(sentence_all_words, sentence_all_tags):
        for i, word in enumerate(word_vec):
            word = re.sub(r'\W+', '', word.lower())
            # current word
            if word not in model.key_to_index:
                vec = model['unk']
            else:
                vec = model[word]
            # previous word
            if i > 0:
                pre_word = word_vec[i - 1]
                pre_word = re.sub(r'\W+', '', pre_word.lower())
                if pre_word not in model.key_to_index:
                    pre_rep = model['unk']
                else:
                    pre_rep = model[pre_word]
            else:
                pre_rep = np.zeros(len(model['unk']))  #model['BOS']
            # next word
            if i<len(word_vec)-1:
                next_word = word_vec[i+1]
                next_word = re.sub(r'\W+', '', next_word.lower())
                if next_word not in model.key_to_index:
                    next_rep = model['unk']
                else:
                    next_rep = model[next_word]
            else:
                next_rep = np.zeros(len(model['unk']))
            # combine the representation to include the next word, previous word and current word
            context_rep = np.concatenate([pre_rep, vec, next_rep])
            representation.append(context_rep)
        if len(representation) == 0:
            continue
    return representation, just_tags

# get representation and tags for train and test
train_words, train_tags = create_data('data/train.tagged')
dev_words, dev_tags = create_data('data/dev.tagged')
svm1 = SVC(kernel='rbf', random_state=9)

# fit the model on train set
svm1.fit(train_words, train_tags)

# predict in dev
pred1 = svm1.predict(dev_words)

# culc f1 score
print("svm F1 score" + ":" + str(f1_score(dev_tags, pred1)))

