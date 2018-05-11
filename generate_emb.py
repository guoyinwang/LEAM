import numpy as np
import gensim
import cPickle

loadpath = "./data/yahoo.p"
x = cPickle.load(open(loadpath, "rb"))
train, val, test = x[0], x[1], x[2]
train_lab, val_lab, test_lab = x[3], x[4], x[5]
wordtoix, ixtoword = x[6], x[7]                        

class_name = ['Society Culture',
            'Science Mathematics',
            'Health' ,
            'Education Reference' ,
            'Computers Internet' ,
            'Sports' ,
            'Business Finance' ,
            'Entertainment Music' ,
            'Family Relationships' , 
            'Politics Government']

filename = './data/glove.840B.300d.w2vformat.txt'
model = gensim.models.KeyedVectors.load_word2vec_format(filename)
vector_size = model.vector_size
embedding_vectors = np.random.uniform(-0.001, 0.001, (len(wordtoix), vector_size))
glove_vocab = list(model.vocab.keys())
count = 0
mis_count = 0
for word in wordtoix.keys():
    idx = wordtoix.get(word)
    if word in glove_vocab:
        embedding_vectors[idx] = model.wv[word]
        count += 1
    else:
        mis_count += 1
print("num of vocab in glove: {}".format(count))
print("num of vocab not in glove: {}".format(mis_count))

print("load class embedding")
name_list = [ k.lower().split(' ') for k in class_name]
id_list = [ [ wordtoidx[i] for i in l] for l in name_list]
value_list = [ [ opt.W_emb[i] for i in l]    for l in id_list]
value_mean = [ np.mean(l)  for l in id_list]

cPickle.save(open('./data/yahoo_emb.p', 'wb'), [embedding_vectors, value_mean])

