# %%
import re
import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

# %%
DATA_IN_PATH = '../data/'
train_data = pd.read_csv(DATA_IN_PATH + 'labeledTrainData.tsv',
                         header=0,
                         delimiter='\t',
                         quoting=3)
print(train_data['review'][0])

# %%
review = train_data['review'][0]
review_text = BeautifulSoup(review, 'html5lib').get_text()
review_text = re.sub("[^a-zA-Z]", " ", review_text)
print(review_text)

# %%
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
review_text = review_text.lower()
words = review_text.split()
words = [w for w in words if not w in stop_words]
print(words)

# %%
clean_review = ' '.join(words)
print(clean_review)

# %%
def preprocessing(review, remove_stopwords = False):
    review_text = BeautifulSoup(review, "html5lib").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        clean_review = ' '.join(words)
    else:
        clean_review = ' '.join(words)

    return clean_review

# %%
clean_train_reviews = []
for review in train_data['review']:
    clean_train_reviews.append(preprocessing(review, remove_stopwords = True))

clean_train_reviews[0]

# %%
clean_train_df = pd.DataFrame({'review': clean_train_reviews,
                               'sentiment': train_data['sentiment']})
tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_train_reviews)
text_sequences = tokenizer.texts_to_sequences(clean_train_reviews)

print(text_sequences[0])

# %%
word_vocab = tokenizer.word_index
word_vocab["<PAD>"] = 0
print(word_vocab)

# %%
print(f"Total words count : {len(word_vocab)}")

# %%
data_configs = {}
data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab)

MAX_SEQUENCE_LENGTH = 174
train_inputs = pad_sequences(text_sequences,
                             maxlen=MAX_SEQUENCE_LENGTH,
                             padding='post')
print(f'Shape of train data: {train_inputs.shape}')

# %%
train_labels = np.array(train_data['sentiment'])
print(f'Shape of label tensor: {train_labels.shape}')

# %%
TRAIN_INPUT_DATA = 'train_input.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
TRAIN_CLEAN_DATA = 'train_clean.csv'
DATA_CONFIGS = 'data_configs.json'

import os

if not os.path.exists(DATA_IN_PATH):
    os.makedirs(DATA_IN_PATH)

np.save(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'wb'), train_inputs)
np.save(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'wb'), train_labels)

clean_train_df.to_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA, index=False)

json.dump(data_configs, open(DATA_IN_PATH + DATA_CONFIGS, 'w'), ensure_ascii=False)

test_data = pd.read_csv(DATA_IN_PATH + "testData.tsv",
                        header=0,
                        delimiter="\t",
                        quoting=3)
clean_test_reviews = []

for review in test_data['review']:
    clean_test_reviews.append(preprocessing(review, remove_stopwords=True))
clean_test_df = pd.DataFrame({'review': clean_test_reviews, 'id': test_data['id']})
test_id = np.array(test_data['id'])

text_sequences = tokenizer.texts_to_sequences(clean_test_reviews)
test_inputs = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

TEST_INPUT_DATA = 'test_input.npy'
TEST_CLEAN_DATA = 'test_clean.csv'
TEST_ID_DATA = 'test_id.npy'

np.save(open(DATA_IN_PATH + TEST_INPUT_DATA, 'wb'), test_inputs)
np.save(open(DATA_IN_PATH + TEST_ID_DATA, 'wb'), test_id)
clean_test_df.to_csv(DATA_IN_PATH + TEST_CLEAN_DATA, index=False)
