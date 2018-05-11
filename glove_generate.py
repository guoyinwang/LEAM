
import gensim
import cPickle
import numpy as np

def load_embedding_vectors_glove_gensim(vocabulary, filename):
    print("loading embedding")
    model = gensim.models.KeyedVectors.load_word2vec_format(filename)
    vector_size = model.vector_size
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    glove_vocab = list(model.vocab.keys())
    count = 0
    mis_count = 0
    for word in vocabulary.keys():
        idx = vocabulary.get(word)
        if word in glove_vocab:
            embedding_vectors[idx] = model.wv[word]

            count += 1
        else:
            mis_count += 1
    print("num of vocab in glove: {}".format(count))
    print("num of vocab not in glove: {}".format(mis_count))
    return embedding_vectors

if __name__ == '__main__':

    loadpath = "yelp_full.p"
    embpath = "yelp_full_glove.p"
    x = cPickle.load(open(loadpath, "rb"))
    train, val, test = x[0], x[1], x[2]
    train_lab, val_lab, test_lab = x[3], x[4], x[5]
    wordtoix, ixtoword = x[6], x[7]

    print("load data finished")

    y = load_embedding_vectors_glove_gensim(wordtoix,'glove.840B.300d.w2vformat.txt' )
    cPickle.dump([y.astype(np.float32)], open(embpath, 'wb'))

