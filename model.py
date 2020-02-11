''' Copyright (C) 2020 Fady Benattayallah '''
import string
import re
import numpy as np
from nltk.corpus import stopwords
from os import listdir
from pickle import dump
from pickle import load
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

DIRECTORY = "review_polarity/txt_sentoken"

# return a string containing the whole document
def load_movie_review(filename):
    file = open(filename, 'r')
    doc = file.read()
    file.close()
    return doc

# turn a doc (string sentence) into a list of cleaned tokens but as a single string
def clean_doc(doc):
    tokens = doc.split()
    
    # remove all punctuation from tokens
    re_punc = re.compile("[%s]" % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    
    # remove non-alphabetic tokens, stop words and words length then length 1
    tokens = [w for w in tokens if w.isalpha() and w not in set(stopwords.words('english')) and len(w) > 1]
    tokens = ' '.join(tokens)
    return tokens


# loads all cleaned movie reviews in a directory into an array
# ["cleaned review 1", "cleaned review 2", ...]
def process_docs(directory, is_train, whole_dataset=False):
    documents = list()
    for filename in listdir(directory):
        if not whole_dataset and is_train and filename.startswith('cv9'):
            continue
        if not whole_dataset and not is_train and not filename.startswith('cv9'):
            continue
        doc = load_movie_review(directory + "/" + filename)
        tokens = clean_doc(doc)
        documents.append(tokens)
    return documents

# returns [[tokens]], [label]]
def load_clean_dataset(is_train, whole_dataset=False):
    neg = process_docs(DIRECTORY + "/neg", is_train, whole_dataset)
    pos = process_docs(DIRECTORY + "/pos", is_train, whole_dataset)
    docs = neg + pos
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels


# save object using pickle
def save_object(dataset, filename):
    dump(dataset, open(filename, 'wb'))
    print('Saved: %s' % filename)
    
def load_object(filename):
    return load(open(filename, 'rb'))

def create_tokenizer(lines, count_thres=0):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    
    if count_thres > 0:
        low_count_words = [w for w,c in tokenizer.word_counts.items() if c < count_thres]
        for w in low_count_words:
            del tokenizer.word_index[w]
            del tokenizer.word_docs[w]
            del tokenizer.word_counts[w]
            
    return tokenizer

# calculates document with most tokens in ["sentence 1", "sentence 2", ...]
def max_length(lines):
    return max([len(s.split()) for s in lines])

# calculates mean length of tokens in documents in ["sentence 1", ...]
def mean_length(lines):
    lengths = [len(s.split()) for s in lines]
    return int(sum(lengths) / len(lengths))

def third_percentile_length(lines):
    lengths = [len(s.split()) for s in lines]
    lengths.sort()
    return lengths[3 * (len(lengths) // 4)]
    

def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad them so that all integer numpy arrays encoded are the same length
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

def load_glove_embedding(glove_filename, tokenizer):
    # load glove as dictionary {word : embedding, ...} for only the words in
    # the tokenizer vocabulary
    glove_file = open(glove_filename, mode="rt", encoding="utf-8")
    word_dict = dict()
    for line in glove_file:
        values = line.split()
        word = values[0]
        word_dict[word] = np.asarray(values[1:], dtype="float32")
    glove_file.close()
    
    # create an embedding matrix which is indexed by words in training docs
    vocab_size= len(tokenizer.word_index) + 1
    dimensions = 50
    if "100" in glove_filename:
        dimensions = 100
    elif "200" in glove_filename:
        dimensions = 200
    else:
        dimensions = 300
        
    embedding_matrix = np.zeros((vocab_size, dimensions))
    for word, unique_index in tokenizer.word_index.items():
        # get the embedding vector from the dictionary created above
        vec = word_dict.get(word)
        if vec is not None:
            embedding_matrix[unique_index] = vec
    
    return embedding_matrix

def define_model(length, vocab_size, embedding_matrix):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=length, trainable=True)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    
    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=length, trainable=True)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    
    # channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=length, trainable=True)(inputs3)
    conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    
    merged = concatenate([flat1, flat2, flat3])
    
    # decision net
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.summary()
    plot_model(model, show_shapes=True, to_file="model.png")
    return model

def TRAIN(save_name):
    # load the training set
    trainLines, trainLabels = load_object("train.pkl")
    
    tokenizer = create_tokenizer(trainLines)
    save_object(tokenizer, "tokenizer.pkl")
    
    vocab_size = len(tokenizer.word_index) + 1
    length = max_length(trainLines)
    save_object(length, "length.pkl")
    
    trainX = encode_text(tokenizer, trainLines, length)
    
    embedding_matrix = load_glove_embedding("glove.6B.300d.txt", tokenizer)
    
    model = define_model(length, vocab_size, embedding_matrix)
    model.fit([trainX, trainX, trainX], trainLabels, epochs=7, batch_size=16, shuffle=True)
    model.save(save_name)

def EVAL(load_name):
    # load the saved data sets
    trainLines, trainLabels = load_object("train.pkl")
    testLines, testLabels = load_object("test.pkl")
    
    # re-create the same tokenizer used for training
    tokenizer = load_object("tokenizer.pkl")
    
    # calculate the max document length for padding
    length = load_object("length.pkl")
    
    # encode the data to get [[unique indexes]] numpy array
    trainX = encode_text(tokenizer, trainLines, length)
    testX  = encode_text(tokenizer, testLines, length)
    model = load_model(load_name)
    _, acc = model.evaluate([trainX, trainX, trainX], trainLabels, verbose=0)
    print("Training accuracy: %.2f" % (acc * 100))
    _, acc = model.evaluate([testX], testLabels, verbose=0)
    print("Test accuracy: %.2f" % (acc * 100))
    
def make_prediction(model, tokenizer, movie_review, length):
    # clean the movie review
    cleaned = clean_doc(movie_review)
    encoded = encode_text(tokenizer, cleaned, length)
    pred = model.predict(encoded, verbose=0)
    
    # get the probability from sigmoid activation
    probability = pred[0, 0]
    if round(probability) == 0:
        return (1 - probability), 'NEGATIVE'
    return probability, 'POSITIVE'


model = load_model("model.kernels=4_6_8_filters=32.h5")
tokenizer = load_object("tokenizer.pkl")
length = load_object("length.pkl")
movie_review = load_movie_review(DIRECTORY + "/neg/cv000_29416.txt")
# clean the movie review
cleaned = clean_doc(movie_review)
encoded = encode_text(tokenizer, cleaned, length)
pred = model.predict([np.array(encoded), np.array(encoded), np.array(encoded)], verbose=0)

# get the probability from sigmoid activation
probability = pred[0, 0]
if round(probability) == 0:
    print( (1 - probability), 'NEGATIVE')
else:
    print(probability, 'POSITIVE')
#print(make_prediction(model, tokenizer, movie_review, length))
