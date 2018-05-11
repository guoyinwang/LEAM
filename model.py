import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import metrics
from tensorflow.contrib import slim
# from tensorflow.contrib.learn import monitors
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_decoder, sequence_loss, embedding_rnn_seq2seq, \
    embedding_tied_rnn_seq2seq
import pdb
import copy
from utils import normalizing
from tensorflow.python.ops import nn_ops, math_ops
import numpy as np


# def embedding(features, opt, prefix = '', is_reuse = None):
#     """Customized function to transform batched x into embeddings."""
#     # Convert indexes of words into embeddings.
#     with tf.variable_scope(prefix+'embed', reuse=is_reuse):
#         weightInit = tf.random_uniform_initializer(-0.001, 0.001)
#         W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer=weightInit)
#     #    b = tf.get_variable('b', [opt.embed_size], initializer = tf,random_uniform_initializer(-0.01, 0.01))
#     W_norm = normalizing(W, 1)
#     word_vectors = tf.nn.embedding_lookup(W_norm, features)
#
#     return word_vectors, W_norm

def embedding(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    #  b = tf.get_variable('b', [opt.embed_size], initializer = tf,random_uniform_initializer(-0.01, 0.01))
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_emb'))
            assert (np.shape(np.array(opt.W_emb)) == (opt.n_words, opt.embed_size))
            W = tf.get_variable('W', initializer=opt.W_emb, trainable=True)
            #pdb.set_trace()
            print("initialize word embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer=weightInit)
            # tf.stop_gradient(W)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)

    # W_norm = normalizing(W, 1)
    word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W

def embedding_normalization(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    #  b = tf.get_variable('b', [opt.embed_size], initializer = tf,random_uniform_initializer(-0.01, 0.01))
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_emb'))
            assert (np.shape(np.array(opt.W_emb)) == (opt.n_words, opt.embed_size))
            W = tf.get_variable('W', initializer=opt.W_emb, trainable=True)
            #pdb.set_trace()
            print("initialize word embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer=weightInit)
            # tf.stop_gradient(W)
        
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)

    # W_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1,))
    W_proj = tf.stack(tf.scan(lambda x: x if tf.sqrt(tf.reduce_sum(tf.square(x))) < 1 else x / tf.sqrt(tf.reduce_sum(tf.square(x))) - 1e-5, tf.unstack(W)))

        

    # W_norm = normalizing(W, 1)
    word_vectors = tf.nn.embedding_lookup(W_proj, features)

    return word_vectors, W_proj

def embedding_class(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched y into embeddings."""
    # Convert indexes of words into embeddings.
    #  b = tf.get_variable('b', [opt.embed_size], initializer = tf,random_uniform_initializer(-0.01, 0.01))
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_class_emb'))
            # assert (np.shape(np.array(opt.W_class_emb)) == (opt.num_class, opt.embed_size)) # c * e
            W = tf.get_variable('W_class', initializer=opt.W_class_emb, trainable=True)
            print("initialize class embedding finished")
            #pdb.set_trace()
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W_class', [opt.num_class, opt.embed_size], initializer=weightInit)
            # tf.stop_gradient(W)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)

    # W_norm = normalizing(W, 1)
    word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W

def embedding_class_attribute(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched y into embeddings."""
    # Convert indexes of words into embeddings.
    #  b = tf.get_variable('b', [opt.embed_size], initializer = tf,random_uniform_initializer(-0.01, 0.01))
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            # assert (hasattr(opt, 'W_class_emb'))
            # assert (np.shape(np.array(opt.W_class_emb)) == (opt.num_class, opt.embed_size)) # c * e
            W = tf.get_variable('W_class', initializer=opt.W_class_emb, trainable=True)
            print("initialize class embedding finished")
            #pdb.set_trace()
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W_class', [opt.num_class, opt.embed_size], initializer=weightInit)
            # tf.stop_gradient(W)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)

    # W_norm = normalizing(W, 1)
    word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W

def embedding_class_normalized(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched y into embeddings."""
    # Convert indexes of words into embeddings.
    #  b = tf.get_variable('b', [opt.embed_size], initializer = tf,random_uniform_initializer(-0.01, 0.01))
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_class_emb'))
            assert (np.shape(np.array(opt.W_class_emb)) == (opt.num_class, opt.embed_size))
            W = tf.get_variable('W_class', initializer=opt.W_class_emb, trainable=True)
            print("initialize class embedding finished")
            #pdb.set_trace()
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W_class', [opt.num_class, opt.embed_size], initializer=weightInit)
            # tf.stop_gradient(W)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)

    # W_norm = normalizing(W, 1)
    W_proj = tf.stack(tf.scan(lambda x: x if tf.sqrt(tf.reduce_sum(tf.square(x))) < 1 else x / tf.sqrt(tf.reduce_sum(tf.square(x))) - 1e-5, tf.unstack(W)))

    word_vectors = tf.nn.embedding_lookup(W_proj, features)

    return word_vectors, W_proj

def embedding_joint(features, features_label, opt, prefix='', is_reuse=None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer=weightInit)
    # b = tf.get_variable('b', [opt.embed_size], initializer = tf,random_uniform_initializer(-0.01, 0.01))
    W_norm = normalizing(W, 1)
    word_vectors = tf.nn.embedding_lookup(W_norm, features)
    word_vectors_label = tf.nn.embedding_lookup(W_norm, features_label)

    return word_vectors, word_vectors_label, W_norm


def embedding_only(opt, prefix='', is_reuse=None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer=weightInit)
    # b = tf.get_variable('b', [opt.embed_size], initializer = tf,random_uniform_initializer(-0.01, 0.01))
    W_norm = normalizing(W, 1)

    return W_norm


def aver_emb_encoder(x_emb, x_mask):
    """ compute the average over all word embeddings """
    x_mask = tf.expand_dims(x_mask, axis=-1)
    x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_sum = tf.multiply(x_emb, x_mask)  # batch L emb 1
    H_enc_0 = tf.reduce_sum(x_sum, axis=1, keep_dims=True)  # batch 1 emb 1
    H_enc = tf.squeeze(H_enc_0, [1, 3])  # batch emb
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2, 3])  # batch 1

    #pdb.set_trace()

    H_enc = H_enc / x_mask_sum  # batch emb

    return H_enc

def att_emb_encoder(x_emb, x_mask, W_class, W_class_tran, opt):
    """ compute the average over all word embeddings """

    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    # x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    

    x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2) # b * s * e
    W_class_norm = tf.nn.l2_normalize(W_class_tran, dim = 0) # e * c

    # x_emb_norm = x_emb_1 # b * s * e
    # W_class_norm = W_class_tran # e * c


    
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
    t = 10
    G = G * t
    # pad_zeros = 1 - x_mask # b * s * c

    # G = G + tf.square(G)

    Att_h = tf.nn.softmax(G , dim=-1,  ) # b * s * c
    # Att_h = tf.nn.softmax(G, dim=-1) # b * s * c

    # Att_h_gate = tf.reduce_max(G, axis=2, keep_dims=True) # b * s * c
    # Att_h = tf.multiply(Att_h, Att_h_gate) # b * s * c

    Att_emb = tf.contrib.keras.backend.dot(Att_h, W_class) # b * s * e

    x_full_emb = tf.concat([x_emb_0, Att_emb], -1) # b * s * 2e

    x_sum = tf.multiply(x_full_emb, x_mask)  # batch L emb 1 # b * s * 2e
    # x_sum_0 = tf.squeeze(x_sum)

    Att_v = tf.contrib.layers.conv2d(tf.expand_dims(G,-1),
                                    num_outputs=1,
                                    kernel_size=[1,opt.num_class],
                                    padding='VALID',
                                    activation_fn=tf.nn.tanh) #b * s * 1 * 1
    Att_v = tf.squeeze(Att_v, -1) # b * s * 1
    Att_v_exp  = tf.exp(Att_v * t ) # b * s * 1
    Att_v_exp_mask = tf.multiply(Att_v_exp, x_mask) # b * s * 1
    Att_v_max = Att_v_exp_mask / tf.reduce_sum(Att_v_exp_mask, axis=1, keep_dims=True) # b * s * 1
    x_att = tf.multiply(x_sum, Att_v_max) # b * s * 2e
    H_enc_0 = tf.reduce_sum(x_att, axis=1, keep_dims=True)  # batch 1 emb 1
    H_enc = tf.squeeze(H_enc_0, 1)  # batch emb # b * 2e
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1 # b * 1 * 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2])  # batch 1

    #pdb.set_trace()

    H_enc = H_enc / x_mask_sum  # batch emb

    return H_enc, G, Att_v_max, x_emb_norm, W_class_norm

def att_dev_emb_encoder(x_emb, x_mask, W_class, W_class_tran, opt):
    """ compute the average over all word embeddings """

    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    # x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    

    x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2) # b * s * e
    W_class_norm = tf.nn.l2_normalize(W_class_tran, dim = 0) # e * c

    # x_emb_norm = x_emb_1
    # W_class_norm = W_class_tran
    # W_inner = tf.get_variable('Winner', [opt.embed_size, opt.embed_size], initializer = tf.contrib.layers.xavier_initializer(), trainable=True)

    # G = tf.contrib.keras.backend.dot(tf.contrib.keras.backend.dot(x_emb_norm, W_inner), W_class_norm)  # b * s * c

    b_att = tf.get_variable('binner', [opt.num_class], initializer = tf.contrib.layers.xavier_initializer(), trainable=True)
    
    b_att = tf.expand_dims(b_att,0)
    b_att = tf.expand_dims(b_att,0)
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
    # t = 10
    # G = G * t
    # pad_zeros = 1 - x_mask # b * s * c

    G = G + b_att
    # G = G + tf.square(G)

    Att_h = tf.nn.softmax(G , dim=-1,  ) # b * s * c
    # Att_h = tf.nn.softmax(G, dim=-1) # b * s * c

    # Att_h_gate = tf.reduce_max(G, axis=2, keep_dims=True) # b * s * c
    # Att_h = tf.multiply(Att_h, Att_h_gate) # b * s * c

    Att_emb = tf.contrib.keras.backend.dot(Att_h, W_class) # b * s * e

    x_full_emb = tf.concat([x_emb_0, Att_emb], -1) # b * s * 2e

    x_sum = tf.multiply(x_full_emb, x_mask)  # batch L emb 1 # b * s * 2e
    # x_sum_0 = tf.squeeze(x_sum)
    H_enc = tf.reduce_sum(x_sum, axis=1) # b * 2e

    # Att_v = tf.contrib.layers.conv2d(tf.expand_dims(G,-1),
    #                                 num_outputs=1,
    #                                 kernel_size=[1,opt.num_class],
    #                                 padding='VALID',
    #                                 activation_fn=tf.nn.tanh) #b * s * 1 * 1
    # Att_v = tf.squeeze(Att_v, -1) # b * s * 1
    # Att_v_exp  = tf.exp(Att_v * t ) # b * s * 1
    # Att_v_exp_mask = tf.multiply(Att_v_exp, x_mask) # b * s * 1
    # Att_v_max = Att_v_exp_mask / tf.reduce_sum(Att_v_exp_mask, axis=1, keep_dims=True) # b * s * 1
    # x_att = tf.multiply(x_sum, Att_v_max) # b * s * 2e
    # H_enc_0 = tf.reduce_sum(x_att, axis=1, keep_dims=True)  # batch 1 emb 1
    # H_enc = tf.squeeze(H_enc_0, 1)  # batch emb # b * 2e
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1 # b * 1 * 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2])  # batch 1

    #pdb.set_trace()

    H_enc = H_enc / x_mask_sum  # batch emb

    return H_enc, G,  x_emb_norm, W_class_norm




def att_poincare_dev_emb_encoder(x_emb, x_mask, W_class, W_class_tran, opt):
    """ compute the average over all word embeddings """

    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    # x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    

    # x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2) # b * s * e
    # W_class_norm = tf.nn.l2_normalize(W_class_tran, dim = 0) # e * c

    x_emb_norm = x_emb_1
    W_class_norm = W_class_tran
    # W_inner = tf.get_variable('Winner', [opt.embed_size, opt.embed_size], initializer = tf.contrib.layers.xavier_initializer(), trainable=True)

    # G = tf.contrib.keras.backend.dot(tf.contrib.keras.backend.dot(x_emb_norm, W_inner), W_class_norm)  # b * s * c

    # b_att = tf.get_variable('binner', [opt.num_class], initializer = tf.contrib.layers.xavier_initializer(), trainable=True)
    
    # b_att = tf.expand_dims(b_att,0)
    # b_att = tf.expand_dims(b_att,0)
    G = tf.get_variable('G', [opt.batch_size, opt.sent_len, opt.num_class])
    for i in range(opt.batch_size):
        for j in range(opt.sent_len):
            for k in range(opt.num_class):
                G[i,j,k] = tf.acosh(1 + 2 * (tf.reduce_sum(tf.square(x_emb_norm[i,j,:] - W_class_norm[:,k]))) / (1 - tf.reduce_sum(tf.square(x_emb_norm[i,j,:]))) / (1 - tf.reduce_sum(tf.square(W_class_norm[:,k]))) )
    # G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
    # t = 10
    # G = G * t
    # pad_zeros = 1 - x_mask # b * s * c

    # G = G + b_att
    # G = G + tf.square(G)

    Att_h = tf.nn.softmax(G , dim=-1,  ) # b * s * c
    # Att_h = tf.nn.softmax(G, dim=-1) # b * s * c

    # Att_h_gate = tf.reduce_max(G, axis=2, keep_dims=True) # b * s * c
    # Att_h = tf.multiply(Att_h, Att_h_gate) # b * s * c

    Att_emb = tf.contrib.keras.backend.dot(Att_h, W_class) # b * s * e

    x_full_emb = tf.concat([x_emb_0, Att_emb], -1) # b * s * 2e

    x_sum = tf.multiply(x_full_emb, x_mask)  # batch L emb 1 # b * s * 2e
    # x_sum_0 = tf.squeeze(x_sum)
    H_enc = tf.reduce_sum(x_sum, axis=1) # b * 2e

    # Att_v = tf.contrib.layers.conv2d(tf.expand_dims(G,-1),
    #                                 num_outputs=1,
    #                                 kernel_size=[1,opt.num_class],
    #                                 padding='VALID',
    #                                 activation_fn=tf.nn.tanh) #b * s * 1 * 1
    # Att_v = tf.squeeze(Att_v, -1) # b * s * 1
    # Att_v_exp  = tf.exp(Att_v * t ) # b * s * 1
    # Att_v_exp_mask = tf.multiply(Att_v_exp, x_mask) # b * s * 1
    # Att_v_max = Att_v_exp_mask / tf.reduce_sum(Att_v_exp_mask, axis=1, keep_dims=True) # b * s * 1
    # x_att = tf.multiply(x_sum, Att_v_max) # b * s * 2e
    # H_enc_0 = tf.reduce_sum(x_att, axis=1, keep_dims=True)  # batch 1 emb 1
    # H_enc = tf.squeeze(H_enc_0, 1)  # batch emb # b * 2e
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1 # b * 1 * 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2])  # batch 1

    #pdb.set_trace()

    H_enc = H_enc / x_mask_sum  # batch emb

    return H_enc, G,  x_emb_norm, W_class_norm
def att_G_emb_encoder(x_emb, x_mask, W_class, W_class_tran, opt):
    """ compute the average over all word embeddings """

    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    # x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    # x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2) # b * s * e
    # W_class_norm = tf.nn.l2_normalize(W_class_tran, dim = 0) # e * c

    x_emb_norm = x_emb_1
    W_class_norm = W_class_tran

    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
    # t = 10
    # G = G * t
    # pad_zeros = 1 - x_mask # b * s * c

    Att_h = tf.nn.softmax(G , dim=-1,  ) # b * s * c
    
    G_sum = tf.multiply(G, x_mask)
    H_enc = tf.reduce_sum(G_sum, axis=1) # b * e
    # H_enc = tf.reduce_max(G_sum, axis=1) # b * e

    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1 # b * 1 * 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2])  # batch 1

    #pdb.set_trace()

    H_enc = H_enc / x_mask_sum  # batch emb

    return H_enc, G,  x_emb_norm, W_class_norm
def att_dim_emb_encoder(x_emb, x_mask, W_class, W_class_tran, opt):
    """ compute the average over all word embeddings """

    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    # x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    x_emb_norm = tf.multiply(tf.exp(x_emb_1), x_mask) / tf.reduce_sum(tf.multiply(tf.exp(x_emb_1), x_mask), axis=2, keep_dims=True)
    # x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2) # b * s * e
    W_class_norm = tf.nn.l2_normalize(W_class_tran, dim = 0) # e * c

    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
    t = 10
    G = G * t
    # pad_zeros = 1 - x_mask # b * s * c

    Att_h = tf.nn.softmax(G , dim=-1,  ) # b * s * c
    # Att_h = tf.nn.softmax(G, dim=-1) # b * s * c

    Att_h_gate = tf.reduce_max(G, axis=2, keep_dims=True) # b * s * c
    Att_h = tf.multiply(Att_h, Att_h_gate) # b * s * c

    Att_emb = tf.contrib.keras.backend.dot(Att_h, W_class) # b * s * e

    x_full_emb = tf.concat([x_emb_0, Att_emb], -1) # b * s * 2e

    x_sum = tf.multiply(x_full_emb, x_mask)  # batch L emb 1 # b * s * 2e
    # x_sum_0 = tf.squeeze(x_sum)

    Att_v = tf.contrib.layers.conv2d(tf.expand_dims(G,-1),
                                    num_outputs=1,
                                    kernel_size=[1,opt.num_class],
                                    padding='VALID',
                                    activation_fn=tf.nn.tanh) #b * s * 1 * 1
    Att_v = tf.squeeze(Att_v, -1) # b * s * 1
    Att_v_exp  = tf.exp(Att_v * t ) # b * s * 1
    Att_v_exp_mask = tf.multiply(Att_v_exp, x_mask) # b * s * 1
    Att_v_max = Att_v_exp_mask / tf.reduce_sum(Att_v_exp_mask, axis=1, keep_dims=True) # b * s * 1
    x_att = tf.multiply(x_sum, Att_v_max) # b * s * 2e
    H_enc_0 = tf.reduce_sum(x_att, axis=1, keep_dims=True)  # batch 1 emb 1
    H_enc = tf.squeeze(H_enc_0, 1)  # batch emb # b * 2e
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1 # b * 1 * 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2])  # batch 1

    #pdb.set_trace()

    H_enc = H_enc / x_mask_sum  # batch emb

    return H_enc, Att_h, Att_v_max, x_emb_norm, W_class_norm

def att_emb_l1_encoder(x_emb, x_mask, W_class, W_class_tran, opt):
    """ compute the average over all word embeddings """

    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    # x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    x_emb_norm = tf.nn.l2_normalize(x_emb_0, dim=-1) # b * s * e
    # W_class_norm = tf.nn.l2_normalize(W_class_tran, dim = -1) # e * c
    # x_emb_norm = x_emb_1 / tf.norm(x_emb_1, ord=1, axis=1, keep_dims=True) # b * s * e
    # W_class_norm = W_class_tran / tf.norm(W_class_tran, ord=1, axis=0, keep_dims=True) # e * c

    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
    # add extra for padding to be non-zeros
    pad_zeros = 1 - x_mask # b * s * c

    Att_h = tf.nn.softmax(G + pad_zeros, dim=-1) # b * s * c

    Att_emb = tf.contrib.keras.backend.dot(Att_h, W_class) # b * s * e

    x_full_emb = tf.concat([x_emb_1, Att_emb], -1) # b * s * 2e
    # x_sum = x_full_emb # b * s * 2e

    x_sum = tf.multiply(x_full_emb, x_mask)  # batch L emb 1 # b * s * 2e
    # x_sum_0 = tf.squeeze(x_sum)

    Att_v = tf.contrib.layers.conv2d(G,
                                    num_outputs=1,
                                    kernel_size=[opt.num_class],
                                    ) #b * s * 1
    Att_v_exp  = tf.exp(Att_v) # b * s * 1
    Att_v_exp_mask = tf.multiply(Att_v_exp, x_mask) # b * s * 1
    Att_v_max = Att_v_exp_mask / tf.reduce_sum(Att_v_exp_mask, axis=1, keep_dims=True) # b * s * 1
    x_att = tf.multiply(x_sum, Att_v_max) # b * s * 2e
    H_enc_0 = tf.reduce_sum(x_att, axis=1, keep_dims=True)  # batch 1 emb 1 # b * 1 * 2e
    H_enc = tf.squeeze(H_enc_0, 1)  # batch emb # b * 2e
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1 # b  * 2e
    x_mask_sum = tf.squeeze(x_mask_sum, [2])  # batch 1

    #pdb.set_trace()

    H_enc = H_enc / x_mask_sum  # batch emb

    return H_enc

def att_emb_dimsum_encoder(x_emb, x_mask, W_class, W_class_tran, opt):
    """ compute the average over all word embeddings """

    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    # x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    x_emb_norm = tf.nn.l2_normalize(x_emb_0, dim=-1) # b * s * e
    # W_class_norm = tf.nn.l2_normalize(W_class_tran, dim = -1) # e * c
    # x_emb_norm = x_emb_1 / tf.norm(x_emb_1, ord=1, axis=1, keep_dims=True) # b * s * e
    # W_class_norm = W_class_tran / tf.norm(W_class_tran, ord=1, axis=0, keep_dims=True) # e * c

    H_enc_0 = tf.multiply(x_emb_norm, x_emb_1) # b * s * e

    H_enc = tf.reduce_sum(H_enc_0, 1) # b * s * e


    # x_emb_att = tf.reduce_max(x_emb_norm, axis=-1, keep_dims=True) # b * s * 1

    # x_emb_att_exp = tf.exp(x_emb_att) # b * s * 1
    # x_emb_att_mask = tf.multiply(x_emb_att_exp, x_mask) # b * s * e
    # x_emb_att_max = x_emb_att_mask / tf.reduce_sum(x_emb_att_mask, axis=1, keep_dims=True) # b * s * e

    # x_emb_dim = tf.multiply(x_emb_att_max, x_emb_1) # b*s * e

    # H_enc_0 = tf.reduce_sum(x_emb_dim, axis=1, keep_dims=True)  # batch 1 emb 1 # b * 1 * 2e
    # H_enc = tf.squeeze(H_enc_0, 1)  # batch emb # b * 2e
    # x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1 # b  * 2e
    # x_mask_sum = tf.squeeze(x_mask_sum, [2])  # batch 1

    #pdb.set_trace()

    # H_enc = H_enc / x_mask_sum  # batch emb

    return H_enc

def att_dim_emb_encoder(x_emb, x_mask, W_class, W_class_tran, opt):
    """ compute the average over all word embeddings """

    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    # x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    W_class_tran_1= tf.expand_dims(W_class_tran, 0) # 1 * e * c
    W_class_tran_1= tf.expand_dims(W_class_tran, 0) # 1 * 1 * e * c

    G = tf.multiply(x_emb, W_class_tran_1) #b * s * e * c

    Att_h = tf.nn.softmax(G, dim=-1) # b * s * e * c

    Att_emb = tf.multiply(x_emb, W_class_tran_1) # b * s * e * c

    Att_emb = tf.reduce_sum(Att_emb, -1) # b * s * e

    x_full_emb = tf.concat([x_emb_0, Att_emb], -1) # b * s * 2e
    x_emb_mask = tf.multiply(x_emb_0, x_mask) # b *s * e
    Att_emb_mask = tf.multiply(Att_emb, x_mask) # b *s * e

    x_sum = tf.multiply(x_full_emb, x_mask)  # batch L emb 1 # b * s * 2e
    # x_sum_0 = tf.squeeze(x_sum)

    Att_v = layers.separable_conv2d(tf.transpose(G, [0,1,3,2]),
                                    num_outputs=None,
                                    kernel_size=[1,opt.num_class],
                                    padding='VALID',
                                    depth_multiplier=1) #b * s * e *1
    Att_v = tf.squeeze(Att_v) # b * s * e
    Att_v_exp  = tf.exp(Att_v) # b * s * e
    Att_v_exp_mask = tf.multiply(Att_v_exp, x_mask) # b * s * e
    Att_v_max = Att_v_exp_mask / tf.reduce_sum(Att_v_exp_mask, axis=1, keep_dims=True) # b * s * e
    x_att = tf.concat([tf.multiply(x_emb_mask, Att_v_max), tf.multiply(Att_emb_mask, Att_v_max)], -1) # b * s * 2e
    H_enc_0 = tf.reduce_sum(x_att, axis=1, keep_dims=True)  # batch 1 emb 1
    H_enc = tf.squeeze(H_enc_0, 1)  # batch emb # b * 2e
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1 # b * 1 * 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2])  # batch 1

    #pdb.set_trace()

    H_enc = H_enc / x_mask_sum  # batch emb

    return H_enc

def att_emb_ngram_encoder(x_emb, x_mask, W_class, W_class_tran, opt):
    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2) # b * s * e
    W_class_norm = tf.nn.l2_normalize(W_class_tran, dim = 0) # e * c
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
    Att_h = tf.nn.softmax(G , dim=-1,  ) # b * s * c
    Att_h = tf.multiply(x_mask, Att_h)

    Att_emb = tf.contrib.keras.backend.dot(Att_h, W_class) # b * s * e
    x_full_emb = tf.concat([x_emb_0, Att_emb], -1) # b * s * 2e

    Att_v = tf.contrib.layers.conv2d(G, num_outputs=1,kernel_size=[45], padding='SAME',activation_fn=tf.nn.relu) #b * s *  1

    Att_v_max = partial_softmax(Att_v, x_mask, 1, 'Att_v_max') # b * s * 1

    x_att = tf.multiply(x_full_emb, Att_v_max)
    H_enc = tf.reduce_sum(x_att, axis=1)

    return H_enc

def att_emb_ngram_encoder_maxout(x_emb, x_mask, W_class, W_class_tran, opt):
    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2) # b * s * e
    W_class_norm = tf.nn.l2_normalize(W_class_tran, dim = 0) # e * c
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
    # Att_h = G # b * s * c
    # # #TODO
    # G = 3 * G
    Att_h = tf.nn.softmax(G , dim=-1,  ) # b * s * c
    Att_h = tf.multiply(x_mask, Att_h)

    Att_emb = tf.contrib.keras.backend.dot(Att_h, W_class) # b * s * e
    # x_full_emb = tf.concat([x_emb_0, Att_emb], -1) # b * s * 2e
    x_full_emb = x_emb_0
    # x_full_emb = Att_emb


    Att_v = tf.contrib.layers.conv2d(G, num_outputs=opt.num_class,kernel_size=[15], padding='SAME',activation_fn=tf.nn.relu) #b * s *  c

    Att_v = tf.reduce_max(Att_v, axis=-1, keep_dims=True)


    Att_v_max = partial_softmax(Att_v, x_mask, 1, 'Att_v_max') # b * s * 1

    x_att = tf.multiply(x_full_emb, Att_v_max)
    H_enc = tf.reduce_sum(x_att, axis=1)

    # # TODO
    # x_att = x_full_emb
    # H_enc = tf.reduce_mean(x_att, axis=1)   

    return H_enc, Att_h, Att_v_max

def att_emb_ngram_encoder_maxout_concat(x_emb, x_mask, W_class, W_class_tran, opt):
    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2) # b * s * e
    W_class_norm = tf.nn.l2_normalize(W_class_tran, dim = 0) # e * c
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
    # Att_h = G # b * s * c
    # # #TODO
    # G = 3 * G
    Att_h = tf.nn.softmax(G , dim=-1,  ) # b * s * c
    Att_h = tf.multiply(x_mask, Att_h)

    Att_emb = tf.contrib.keras.backend.dot(Att_h, W_class) # b * s * e
    # x_full_emb = tf.concat([x_emb_0, Att_emb], -1) # b * s * 2e
    x_full_emb = x_emb_0
    # x_full_emb = Att_emb


    Att_v = tf.contrib.layers.conv2d(G, num_outputs=opt.num_class,kernel_size=[7], padding='SAME',activation_fn=tf.nn.relu) #b * s *  c

    Att_v = tf.reduce_max(Att_v, axis=-1, keep_dims=True)


    Att_v_max = partial_softmax(Att_v, x_mask, 1, 'Att_v_max') # b * s * 1

    x_att = tf.multiply(x_full_emb, Att_v_max)
    H_enc = tf.reduce_sum(x_att, axis=1)

    H_enc_cnn = conv_pooling(x_emb, opt, prefix='ngram-cnn', is_reuse=None, num_outputs=None)

    H_enc = tf.concat([H_enc, tf.squeeze(H_enc_cnn)], -1)

    # # TODO
    # x_att = x_full_emb
    # H_enc = tf.reduce_mean(x_att, axis=1)   

    return H_enc, Att_h, Att_v_max



def att_emb_ngram_encoder_maxout_cnn(x_emb, x_mask, W_class, W_class_tran, opt):
    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2) # b * s * e
    W_class_norm = tf.nn.l2_normalize(W_class_tran, dim = 0) # e * c
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
    # Att_h = G # b * s * c
    # # #TODO
    # G = 3 * G
    Att_h = tf.nn.softmax(G , dim=-1,  ) # b * s * c
    Att_h = tf.multiply(x_mask, Att_h)

    Att_emb = tf.contrib.keras.backend.dot(Att_h, W_class) # b * s * e
    # x_full_emb = tf.concat([x_emb_0, Att_emb], -1) # b * s * 2e
    x_full_emb = x_emb_0
    # x_full_emb = Att_emb


    Att_v = tf.contrib.layers.conv2d(G, num_outputs=opt.num_class,kernel_size=[1], padding='SAME',activation_fn=tf.nn.relu) #b * s *  c
    # Att_v = tf.nn.relu()
    Att_v = tf.reduce_max(Att_v, axis=-1, keep_dims=True)


    Att_v_max = partial_softmax(Att_v, x_mask, 1, 'Att_v_max') # b * s * 1

    x_att = tf.multiply(x_full_emb, Att_v_max)
    # H_enc = conv_pooling(tf.expand_dims(x_att, -1), opt, prefix='ngram-cnn', is_reuse=None, num_outputs=opt.embed_size)
    H_enc = conv_pooling(tf.expand_dims(x_att, -1), opt, prefix='ngram-cnn', is_reuse=None, num_outputs=None)
    # H_enc = tf.reduce_sum(x_att, axis=1)

    # # TODO
    # x_att = x_full_emb
    # H_enc = tf.reduce_mean(x_att, axis=1)   

    return H_enc, Att_h, Att_v_max

def att_emb_ngram_encoder_maxout_more(x_emb, x_mask, W_class, W_class_tran, opt):
    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2) # b * s * e
    W_class_norm = tf.nn.l2_normalize(W_class_tran, dim = 0) # e * c
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
    # Att_h = G # b * s * c
    Att_h = tf.nn.softmax(G , dim=-1,  ) # b * s * c
    Att_h = tf.multiply(x_mask, Att_h)

    # W = tf.get_variable('innerW', shape=[opt.emb_dim, opt.emb_dim], dtype=tf.float32, initializer=tf.nn.tr)

    Att_emb = tf.contrib.keras.backend.dot(Att_h, W_class) # b * s * e
    # x_full_emb = tf.concat([x_emb_0, Att_emb], -1) # b * s * 2e
    x_full_emb = x_emb_0
    # x_full_emb = Att_emb


    Att_v = tf.contrib.layers.conv2d(G, num_outputs=902,kernel_size=[55], padding='SAME',activation_fn=tf.nn.relu) #b * s *  c

    Att_v = tf.reduce_max(Att_v, axis=-1, keep_dims=True)


    Att_v_max = partial_softmax(Att_v, x_mask, 1, 'Att_v_max') # b * s * 1

    x_att = tf.multiply(x_full_emb, Att_v_max)
    H_enc = tf.reduce_sum(x_att, axis=1)

    return H_enc, Att_h, Att_v_max


def att_emb_ngram_encoder_cnn(x_emb, x_mask, W_class, W_class_tran, opt):
    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e


    H = tf.contrib.layers.conv2d(x_emb_0, num_outputs=opt.embed_size,kernel_size=[10], padding='SAME',activation_fn=tf.nn.relu) #b * s *  c


    G = tf.contrib.keras.backend.dot(H, W_class_tran) # b * s * c
    # x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2) # b * s * e
    # W_class_norm = tf.nn.l2_normalize(W_class_tran, dim = 0) # e * c
    # G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
    # Att_h = G # b * s * c
    # # #TODO
    # G = 3 * G
    # Att_h = tf.nn.softmax(G , dim=-1,  ) # b * s * c
    # Att_h = tf.multiply(x_mask, Att_h)

    # Att_emb = tf.contrib.keras.backend.dot(Att_h, W_class) # b * s * e
    # # x_full_emb = tf.concat([x_emb_0, Att_emb], -1) # b * s * 2e
    # x_full_emb = x_emb_0
    # x_full_emb = Att_emb


    # Att_v = tf.contrib.layers.conv2d(G, num_outputs=opt.num_class,kernel_size=[55], padding='SAME',activation_fn=tf.nn.relu) #b * s *  c


    # Att_v = tf.reduce_max(Att_v, axis=-1, keep_dims=True)


    Att_v_max = partial_softmax(G, x_mask, 1, 'Att_v_max') # b * s * c

    x_att = tf.contrib.keras.backend.batch_dot(tf.transpose(H,[0,2,1]), Att_v_max)
    # x_att = tf.multiply(tf.expand_dims(H,-1), tf.expand_dims(Att_v_max,2))
    # H_enc = tf.reduce_sum(x_att,1)
    H_enc = tf.squeeze(x_att)

    # H_enc = tf.reduce_sum(x_att, axis=1)

    # # TODO
    # x_att = x_full_emb
    # H_enc = tf.reduce_mean(x_att, axis=1)   

    return H_enc


def max_emb_encoder(x_emb, x_mask, opt):
    """ compute the max over every dimension of word embeddings """
    x_mask_1 = tf.expand_dims(x_mask, axis=-1)
    x_mask_1 = tf.expand_dims(x_mask_1, axis=-1)
    H_enc = tf.nn.max_pool(tf.multiply(x_emb, x_mask_1), [1, opt.maxlen, 1, 1], [1, 1, 1, 1], 'VALID')
    H_enc = tf.squeeze(H_enc)

    return H_enc


def concat_emb_encoder(x_emb, x_mask, opt):
    """ concat both the average and max over all word embeddings """
    x_mask = tf.expand_dims(x_mask, axis=-1)
    x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_sum = tf.multiply(x_emb, x_mask)  # batch L emb 1
    H_enc = tf.reduce_sum(x_sum, axis=1, keep_dims=True)  # batch 1 emb 1
    H_enc = tf.squeeze(H_enc, [1, 3])  # batch emb
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2, 3])  # batch 1

    H_enc_1 = H_enc / x_mask_sum  # batch emb

    H_enc_2 = tf.nn.max_pool(x_emb, [1, opt.maxlen, 1, 1], [1, 1, 1, 1], 'VALID')
    #pdb.set_trace()
    H_enc_2 = tf.squeeze(H_enc_2, [1, 3])

    H_enc = tf.concat([H_enc_1, H_enc_2], 1)

    return H_enc



def conv_encoder(x_emb, is_train, opt, res, is_reuse = None, prefix=''):
    if hasattr(opt, 'multiplier'):
        multiplier = opt.multiplier
    else:
        multiplier = 2
    if opt.layer == 4:
        H_enc = conv_model_4layer(x_emb, opt, is_train = is_train, is_reuse = is_reuse, prefix = prefix)
    elif opt.layer == 3:
        H_enc = conv_model_3layer(x_emb, opt, is_train = is_train, multiplier = multiplier, is_reuse = is_reuse, prefix = prefix)
    elif opt.layer == 0:
        H_enc = conv_model_3layer_old(x_emb, opt, is_reuse = is_reuse, prefix = prefix)
    else:
        H_enc = conv_model(x_emb, opt, is_train = is_train, is_reuse = is_reuse, prefix = prefix)
    return H_enc, res


# def dim_attention(x_emb. x_mask, W_class, W_class_trans, opt):
#     with tf.name_scope('class_att_para'):
#         att_class_weight = slim.variable(shape=[opt.num_class, opt.emb_size, opt.emb_size])



def conv_model(X, opt, prefix='', is_reuse=None):
    # XX = tf.reshape(X, [-1, , 28, 1])
    # X shape: batchsize L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        weightInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H1 = layers.conv2d(X, num_outputs=opt.filter_size, kernel_size=[opt.filter_shape, opt.embed_size],
                       stride=[opt.stride[0], 1],
                       weights_initializer=weightInit, biases_initializer=biasInit,
                       activation_fn=tf.nn.relu, padding='VALID', scope=prefix + 'H1',
                       reuse=is_reuse)  # batch L-3 1 Filtersize
    acf = tf.nn.tanh if opt.tanh else tf.nn.relu
    H2 = layers.conv2d(H1, num_outputs=opt.filter_size * 2, kernel_size=[opt.sent_len2, 1],
                       biases_initializer=biasInit, activation_fn=acf, padding='VALID', scope=prefix + 'H2',
                       reuse=is_reuse)  # batch 1 1 2*Filtersize
    return H2


def conv_model_3layer(X, opt, prefix='', is_reuse=None, num_outputs=None):
    # XX = tf.reshape(X, [-1, , 28, 1])
    # X shape: batchsize L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

    H1 = layers.conv2d(X, num_outputs=opt.filter_size, kernel_size=[opt.filter_shape, opt.embed_size],
                       stride=[opt.stride[0], 1], weights_initializer=weightInit,
                       biases_initializer=biasInit, activation_fn=tf.nn.relu, padding='VALID', scope=prefix + 'H1_3',
                       reuse=is_reuse)  # batch L-3 1 Filtersize
    H2 = layers.conv2d(H1, num_outputs=opt.filter_size * 2, kernel_size=[opt.filter_shape, 1],
                       stride=[opt.stride[1], 1], biases_initializer=biasInit, activation_fn=tf.nn.relu,
                       padding='VALID', scope=prefix + 'H2_3', reuse=is_reuse)
    # print H2.get_shape()
    acf = tf.nn.tanh if opt.tanh else None
    H3 = layers.conv2d(H2, num_outputs=(num_outputs if num_outputs else opt.n_gan), kernel_size=[opt.sent_len3, 1],
                       biases_initializer=biasInit, activation_fn=acf, padding='VALID',
                       scope=prefix + 'H3_3', reuse=is_reuse)  # batch 1 1 2*Filtersize
    return H3


def conv_pooling(X, opt, prefix='', is_reuse=None, num_outputs=None):
    # XX = tf.reshape(X, [-1, , 28, 1])
    # X shape: batchsize L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

    H1 = layers.conv2d(X, num_outputs=(num_outputs if num_outputs else opt.n_gan),
                       kernel_size=[opt.filter_shape, opt.embed_size],
                       stride=[opt.stride[0], 1], weights_initializer=weightInit,
                       biases_initializer=biasInit, activation_fn=tf.nn.relu, padding='VALID', scope=prefix + 'H1_3',
                       reuse=is_reuse)  # batch L-3 1 Filtersize

    H = tf.nn.max_pool(H1, [1, opt.sent_len2, 1, 1], [1, 1, 1, 1], 'VALID')

    return H

def lstm(X, opt, prefix='', is_reuse=None, num_outputs=None):
    # XX = tf.reshape(X, [-1, , 28, 1])
    # X shape: batchsize L emb 1

    with tf.name_scope(prefix):
        # cell = tf.contrib.rnn.BasicLSTMCell(opt.n_gan, state_is_tuple=True)
        cell = tf.contrib.rnn.BasicLSTMCell(opt.n_gan, )
        cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=opt.dropout)

        # initial_state = cell.zero_state(opt.batch_size, tf.float32)
        # rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, time_major=True)
        X = tf.squeeze(X)

        # rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs=X, initial_state=initial_state)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs=X, dtype=tf.float32, scope=prefix)
        # rnn_outputs, rnn_states = tf.nn.bidrectional_dynamic_rnn(cell_fw=cell,cell_bw=cell, inputs=X, initial_state=initial_state, sequence_length=[opt.maxlen] * opt.batch_size)
        H = tf.reduce_mean(rnn_outputs,axis=1)

    return H


def conv_model_4layer(X, opt, prefix='', is_reuse=None, num_outputs=None):
    # XX = tf.reshape(X, [-1, , 28, 1])
    # X shape: batchsize L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

    H1 = layers.conv2d(X, num_outputs=opt.filter_size, kernel_size=[opt.filter_shape, opt.embed_size],
                       stride=[opt.stride[0], 1], weights_initializer=weightInit, biases_initializer=biasInit,
                       activation_fn=tf.nn.relu, padding='VALID', scope=prefix + 'H1',
                       reuse=is_reuse)  # batch L-3 1 Filtersize
    H2 = layers.conv2d(H1, num_outputs=opt.filter_size * 2, kernel_size=[opt.filter_shape, 1],
                       stride=[opt.stride[1], 1], biases_initializer=biasInit, activation_fn=tf.nn.relu,
                       padding='VALID', scope=prefix + 'H2', reuse=is_reuse)
    H3 = layers.conv2d(H2, num_outputs=opt.filter_size * 4, kernel_size=[opt.filter_shape, 1],
                       stride=[opt.stride[2], 1], biases_initializer=biasInit, activation_fn=tf.nn.relu,
                       padding='VALID', scope=prefix + 'H3', reuse=is_reuse)
    # print H2.get_shape()
    acf = tf.nn.tanh if opt.tanh else None
    H4 = layers.conv2d(H3, num_outputs=(num_outputs if num_outputs else opt.n_gan), kernel_size=[opt.sent_len4, 1],
                       biases_initializer=biasInit, activation_fn=acf, padding='VALID', scope=prefix + 'H4',
                       reuse=is_reuse)  # batch 1 1 2*Filtersize
    return H4


def deconv_model(H, opt, prefix='', is_reuse=None):
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    # H2t = tf.reshape(H, [H.shape[0],1,1,H.shape[1]])
    #    print tf.shape(H)
    #    H2t = tf.expand_dims(H,1)
    #    H2t = tf.expand_dims(H,1)
    H2t = H
    H1t = layers.conv2d_transpose(H2t, num_outputs=opt.filter_size, kernel_size=[opt.sent_len2, 1],
                                  stride=[opt.stride[0], 1], biases_initializer=biasInit, activation_fn=tf.nn.relu,
                                  padding='VALID', scope=prefix + 'H1_t', reuse=is_reuse)
    Xhat = layers.conv2d_transpose(H1t, num_outputs=1, kernel_size=[opt.filter_shape, opt.embed_size],
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, padding='VALID',
                                   scope=prefix + 'Xhat_t', reuse=is_reuse)
    # print H2t.get_shape(), H1t.get_shape(), Xhat.get_shape()
    return Xhat


def deconv_model_3layer(H, opt, prefix='', is_reuse=None):
    # XX = tf.reshape(X, [-1, , 28, 1])
    # X shape: batchsize L emb 1
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)

    H3t = H
    H2t = layers.conv2d_transpose(H3t, num_outputs=opt.filter_size * 2, kernel_size=[opt.sent_len3, 1],
                                  biases_initializer=biasInit, activation_fn=tf.nn.relu, padding='VALID',
                                  scope=prefix + 'H2_t_3', reuse=is_reuse)
    H1t = layers.conv2d_transpose(H2t, num_outputs=opt.filter_size, kernel_size=[opt.filter_shape, 1],
                                  stride=[opt.stride[1], 1], biases_initializer=biasInit, activation_fn=tf.nn.relu,
                                  padding='VALID', scope=prefix + 'H1_t_3', reuse=is_reuse)
    Xhat = layers.conv2d_transpose(H1t, num_outputs=1, kernel_size=[opt.filter_shape, opt.embed_size],
                                   stride=[opt.stride[0], 1], biases_initializer=biasInit, activation_fn=tf.nn.relu,
                                   padding='VALID', scope=prefix + 'Xhat_t_3', reuse=is_reuse)
    # print H2t.get_shape(),H1t.get_shape(),Xhat.get_shape()
    return Xhat


def deconv_model_4layer(H, opt, prefix='', is_reuse=None):
    # XX = tf.reshape(X, [-1, , 28, 1])
    # X shape: batchsize L emb 1
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)

    H4t = H
    H3t = layers.conv2d_transpose(H4t, num_outputs=opt.filter_size * 4, kernel_size=[opt.sent_len4, 1],
                                  biases_initializer=biasInit, activation_fn=tf.nn.relu, padding='VALID',
                                  scope=prefix + 'H3_t', reuse=is_reuse)
    H2t = layers.conv2d_transpose(H3t, num_outputs=opt.filter_size * 2, kernel_size=[opt.filter_shape, 1],
                                  stride=[opt.stride[2], 1], biases_initializer=biasInit, activation_fn=tf.nn.relu,
                                  padding='VALID', scope=prefix + 'H2_t', reuse=is_reuse)
    H1t = layers.conv2d_transpose(H2t, num_outputs=opt.filter_size, kernel_size=[opt.filter_shape, 1],
                                  stride=[opt.stride[1], 1], biases_initializer=biasInit, activation_fn=tf.nn.relu,
                                  padding='VALID', scope=prefix + 'H1_t', reuse=is_reuse)
    Xhat = layers.conv2d_transpose(H1t, num_outputs=1, kernel_size=[opt.filter_shape, opt.embed_size],
                                   stride=[opt.stride[0], 1], biases_initializer=biasInit, activation_fn=tf.nn.relu,
                                   padding='VALID', scope=prefix + 'Xhat_t', reuse=is_reuse)
    # print H2t.get_shape(),H1t.get_shape(),Xhat.get_shape()
    return Xhat


def seq2seq(x, y, opt, prefix='', feed_previous=False, is_reuse=None, is_tied=True):
    # y batch * len   H batch * h
    x = tf.unstack(x, axis=1)  # X Y Z
    y = tf.unstack(y, axis=1)  # GO_ A B C

    with tf.variable_scope(prefix + 'lstm_seq2seq', reuse=is_reuse):
        cell = tf.contrib.rnn.LSTMCell(opt.n_gan)

        # with tf.variable_scope(prefix+'lstm_seq2seq', reuse=is_reuse):
        #        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        #        W = tf.get_variable('W', [opt.n_gan, opt.n_words], initializer = weightInit)
        #        b = tf.get_variable('b', [opt.n_words], initializer = tf.random_uniform_initializer(-0.001, 0.001))
        if is_tied:
            outputs, _ = embedding_tied_rnn_seq2seq(encoder_inputs=x, decoder_inputs=y, cell=cell,
                                                    feed_previous=feed_previous, num_symbols=opt.n_words,
                                                    embedding_size=opt.embed_size)
        else:
            outputs, _ = embedding_rnn_seq2seq(encoder_inputs=x, decoder_inputs=y, cell=cell,
                                               feed_previous=feed_previous, num_encoder_symbols=opt.n_words,
                                               num_decoder_symbols=opt.n_words, embedding_size=opt.embed_size)

    # logits = [nn_ops.xw_plus_b(out, W, b) for out in outputs]
    logits = outputs

    syn_sents = [math_ops.argmax(l, 1) for l in logits]
    syn_sents = tf.stack(syn_sents, 1)

    loss = sequence_loss(outputs[:-1], y[1:], [tf.cast(tf.ones_like(yy), tf.float32) for yy in y[1:]])

    return loss, syn_sents, logits


def lstm_decoder(H, y, opt, prefix='', feed_previous=False, is_reuse=None):
    # y  len* batch * [0,V]   H batch * h

    # y = [tf.squeeze(y[:,i]) for i in xrange(y.get_shape()[1])]
    y = tf.unstack(y, axis=1)
    H0 = tf.squeeze(H)
    H1 = (H0, tf.zeros_like(H0))  # initialize H and C

    with tf.variable_scope(prefix + 'lstm_decoder', reuse=True):
        cell = tf.contrib.rnn.LSTMCell(opt.n_gan)
    with tf.variable_scope(prefix + 'lstm_decoder', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        W = tf.get_variable('W', [opt.n_gan, opt.n_words], initializer=weightInit)
        b = tf.get_variable('b', [opt.n_words], initializer=tf.random_uniform_initializer(-0.001, 0.001))
        out_proj = (W, b) if feed_previous else None
        outputs, _ = embedding_rnn_decoder(decoder_inputs=y, initial_state=H1, cell=cell, feed_previous=feed_previous,
                                           output_projection=out_proj, num_symbols=opt.n_words,
                                           embedding_size=opt.embed_size)

    logits = [nn_ops.xw_plus_b(out, W, b) for out in outputs]
    syn_sents = [math_ops.argmax(l, 1) for l in logits]
    syn_sents = tf.stack(syn_sents, 1)

    # outputs, _ = embedding_rnn_decoder(decoder_inputs = y, initial_state = H, cell = tf.contrib.rnn.BasicLSTMCell, num_symbols = opt.n_words, embedding_size = opt.embed_size, scope = prefix + 'lstm_decoder')

    # outputs : batch * len

    loss = sequence_loss(logits[:-1], y[1:], [tf.cast(tf.ones_like(yy), tf.float32) for yy in y[1:]])

    return loss, syn_sents, logits


# def discriminator_2layer(H, opt, dropout, prefix = '', num_outputs=1, is_reuse= None):
#     # last layer must be linear
#     H = tf.squeeze(H)
#     biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
#     H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob = dropout), num_outputs = opt.H_dis, biases_initializer=biasInit, activation_fn = tf.nn.relu, scope = prefix + 'dis_1', reuse = is_reuse)
#     logits = layers.linear(tf.nn.dropout(H_dis, keep_prob = dropout), num_outputs = num_outputs, biases_initializer=biasInit, scope = prefix + 'dis_2', reuse = is_reuse)
#     return logits

def discriminator_1layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    H = tf.squeeze(H)
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    return H_dis


def discriminator_0layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    H = tf.squeeze(H)
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    logits = layers.linear(tf.nn.dropout(H, keep_prob=dropout), num_outputs=num_outputs, biases_initializer=biasInit,
                           scope=prefix + 'dis', reuse=is_reuse)
    return logits


def softmax_prediction(X, opt, is_reuse=None):
    # X shape: batchsize L emb 1
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    pred_H = layers.conv2d(X, num_outputs=opt.n_words, kernel_size=[1, opt.embed_size], biases_initializer=biasInit,
                           activation_fn=tf.nn.relu, padding='VALID', scope='pred', reuse=is_reuse)  # batch L 1 V

    pred_prob = layers.softmax(pred_H, scope='pred')  # batch L 1 V
    pred_prob = tf.squeeze(pred_prob)  # batch L V
    return pred_prob


def soft_att_layer_multi(X, opt, K=5, prefix='', is_reuse=None):
    # X: B L E 1
    sent_length = X.get_shape().as_list()[1]
    emb_size = X.get_shape().as_list()[2]
    with tf.variable_scope(prefix + 'soft_att', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        V0 = tf.get_variable('V0', [1, emb_size, K], initializer=weightInit)  # 1 E K

    # pdb.set_trace()
    V_norm = tf.tile(normalizing(V0, 1), [opt.batch_size, 1, 1])  # B E K
    # V_norm = tf.expand_dims(V_norm,2)

    prob_logits = tf.squeeze(tf.matmul(tf.squeeze(X), V_norm))  # B L K

    prob = tf.nn.softmax(prob_logits * opt.att_L, dim=1, name=None)  # B L K
    prob = tf.reduce_mean(prob, 2, keep_dims=True)
    prob = tf.expand_dims(prob, 2)  # B L 1
    return X * prob * sent_length, tf.squeeze(prob)


def soft_att_layer(X, opt, prefix='', is_reuse=None):
    # X: B L E 1
    sent_length = X.get_shape().as_list()[1]
    emb_size = X.get_shape().as_list()[2]
    with tf.variable_scope(prefix + 'soft_att', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        V0 = tf.get_variable('V0', [1, emb_size], initializer=weightInit)

    # pdb.set_trace()
    V_norm = tf.tile(normalizing(V0, 1), [opt.batch_size, 1])  # B E
    V_norm = tf.expand_dims(V_norm, 2)

    prob_logits = tf.squeeze(tf.matmul(tf.squeeze(X), V_norm))  # B L

    prob = tf.nn.softmax(prob_logits * opt.att_L, dim=-1, name=None)

    prob = tf.expand_dims(prob, 2)
    prob = tf.expand_dims(prob, 2)
    return X * prob * sent_length, tf.squeeze(prob)


def soft_att_layer_smooth(X, opt, prefix='', is_reuse=None):
    # X: B L E 1
    sent_length = X.get_shape().as_list()[1]
    emb_size = X.get_shape().as_list()[2]
    with tf.variable_scope(prefix + 'soft_att', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        V0 = tf.get_variable('V0', [1, emb_size], initializer=weightInit)

    # pdb.set_trace()
    V_norm = tf.tile(normalizing(V0, 1), [opt.batch_size, 1])  # B E
    V_norm = tf.expand_dims(V_norm, 2)

    prob_logits = tf.squeeze(tf.matmul(tf.squeeze(X), V_norm))  # B L
    prob = tf.nn.softmax(prob_logits * opt.att_L, dim=-1, name=None)
    prob = tf.expand_dims(prob, 2)
    prob = tf.expand_dims(prob, 2)

    weights = np.array([[0.1], [0.2], [0.5], [0.2], [0.1]]).astype('float32')
    prob_new = layers.conv2d(prob, num_outputs=1, weights_initializer=tf.constant_initializer(weights),
                             kernel_size=[5, 1], activation_fn=None, padding='SAME', trainable=False)

    return X * prob_new * sent_length, tf.squeeze(prob_new), tf.squeeze(prob)


def soft_att_layer_2way(X, y, opt, prefix='', is_reuse=None):
    # X: B L E 1
    with tf.variable_scope(prefix + 'soft_att', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        V0 = tf.get_variable('V0', [1, opt.embed_size], initializer=weightInit)
        V1 = tf.get_variable('V1', [1, opt.embed_size], initializer=weightInit)

    # pdb.set_trace()
    V_norm = tf.where(tf.tile(tf.equal(y, tf.zeros_like(y)), [1, opt.embed_size]),
                      tf.tile(normalizing(V0, 0), [opt.batch_size, 1]),
                      tf.tile(normalizing(V1, 0), [opt.batch_size, 1]))  # B E
    V_norm = tf.expand_dims(V_norm, 2)

    prob_logits = tf.squeeze(tf.matmul(tf.squeeze(X), V_norm))  # B L

    prob = tf.nn.softmax(prob_logits * opt.att_L, dim=-1, name=None)
    prob = tf.expand_dims(prob, 2)
    prob = tf.expand_dims(prob, 2)
    return X * prob


def gaussian_att_layer(X, y, opt, prefix=''):
    # X: batchsize * L * E * 1
    # h: batchsize * feature
    L = X.get_shape().as_list()[1]
    F, gamma = attn_window(prefix + 'G_', x, y, opt.att_N, L)
    gamma = tf.reshape(gamma, [-1, 1, 1, 1])
    x_attn = filter_x(X, F) * gamma  # batch_size x read_n x read_n x C
    # x_hat = filter_x(X_hat, F) * gamma  # batch_size x read_n x read_n x C
    # x_input = tf.concat(0, [x_attn, x_hat])  # -1 x read_n x read_n x C
    return x_attn


def filterbank(gx, sigma2, delta, N, L):
    """
    (gx, gy): grid centre position
    sigma2: variance for gaussian filter
    delta: stride (distance between two grids)
    N: grid size is N * N
    """
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])  # 1 * N, [0,1,...,N-1]
    mu = gx + (grid_i - N / 2 - 0.5) * delta  # eq 19 tf can broadcast with (1, N) * (BS, 1) = (BS, N)
    mu = tf.reshape(mu, [-1, N, 1])
    # mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
    # mu_y = tf.reshape(mu_y, [-1, N, 1])
    a = tf.reshape(tf.cast(tf.range(L), tf.float32), [1, 1, -1])  # 1 * 1 * L, [0,1,...,W-1]

    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    F = tf.exp(-tf.square((a - mu_x) / (2 * sigma2)))  # batch_size x N x L
    # Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch_size x N x H
    # normalize, sum over A and B dims
    F_all = tf.reduce_sum(F, 1)  # batch_size x 1 x L

    F_all = F_all / tf.maximum(tf.reduce_sum(F_all, 2), self.eps)
    F_all = tf.squeeze(F_all)
    # Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), self.eps)
    return F_all


def attn_window(prefix, x, y, N, L, is_reuse=None):
    """
    return F, gamma
    F : B L
    """
    with tf.variable_scope(prefix, reuse=is_reuse):
        params = att_layer(x, y, 4, prefix=prefix + '_att')  # batch_size * 5
    gx_, log_sigma2, log_delta, log_gamma = tf.split(1, 4, params)  # tf.unstack(params, axis=1) #
    gx_ = tf.nn.tanh(gx_)
    gx = tf.floor((L + 1) / 2.0 * (gx_ + 1))

    sigma2 = tf.exp(log_sigma2)
    delta = (L - 1) / (N - 1) * tf.exp(log_delta)
    return filterbank(gx, sigma2, delta, N, L) + (tf.exp(log_gamma),)


def att_layer(x, y, opt, output_dim, prefix=''):
    """
    x : batch_size x L x E x 1
    y : batch_size x 1

    """
    H_enc = tf.cond(tf.equal(y, tf.zeros_like(y)), conv_model_3layer(x_emb, opt, prefix=prefix + 'conv_0'),
                    conv_model_3layer(x_emb, opt, prefix=prefix + 'conv_1'))

    return discriminator_2layer(H_enc, opt, dropout, prefix='', num_outputs=output_dim, is_reuse=None)


def filter_x(x, F):
    """
    only for encoder:
        x : batch_size x L x E x 1
        F : batch_size x L
        output :  batch_size x L x E x 1
    """
    # glimpse = tf.batch_matmul(F, x)
    glimpse = x * F

    return glimpse


def linear_layer(x, output_dim):
    input_dim = x.get_shape().as_list()[1]
    thres = np.sqrt(6.0 / (input_dim + output_dim))
    W = tf.get_variable("W", [input_dim, output_dim], scope=prefix + '_W',
                        initializer=tf.random_uniform_initializer(minval=-thres, maxval=thres))
    b = tf.get_variable("b", [output_dim], scope=prefix + '_b', initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, W) + b


def discriminator_2layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    # H = tf.squeeze(H, [1,2])
    # pdb.set_trace()
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    logits = layers.linear(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=num_outputs,
                           biases_initializer=biasInit, scope=prefix + 'dis_2', reuse=is_reuse)
    return logits

def discriminator_3layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    # H = tf.squeeze(H, [1,2])
    # pdb.set_trace()
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    H_dis = layers.fully_connected(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_2',
                                   reuse=is_reuse)
    logits = layers.linear(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=num_outputs,
                           biases_initializer=biasInit, scope=prefix + 'dis_3', reuse=is_reuse)
    return logits

def discriminator_res(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    # H = tf.squeeze(H, [1,2])
    # pdb.set_trace()
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis_0 = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.embed_size,
                                   biases_initializer=biasInit, activation_fn=None, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    H_dis_0n = tf.nn.relu(H_dis_0)                               
    H_dis_1 = layers.fully_connected(tf.nn.dropout(H_dis_0n, keep_prob=dropout), num_outputs=opt.embed_size,
                                   biases_initializer=biasInit, activation_fn=None, scope=prefix + 'dis_2',
                                   reuse=is_reuse)
    H_dis_1n = tf.nn.relu(H_dis_1) + H_dis_0
    H_dis_2 = layers.fully_connected(tf.nn.dropout(H_dis_1n, keep_prob=dropout), num_outputs=opt.embed_size,
                                   biases_initializer=biasInit, activation_fn=None, scope=prefix + 'dis_3',
                                   reuse=is_reuse)
    H_dis_2n = tf.nn.relu(H_dis_2) + H_dis_1
    H_dis_3 = layers.fully_connected(tf.nn.dropout(H_dis_2n, keep_prob=dropout), num_outputs=opt.embed_size,
                                   biases_initializer=biasInit, activation_fn=None, scope=prefix + 'dis_4',
                                   reuse=is_reuse)
    # H_dis_3n = tf.nn.relu(H_dis_3) + H_dis_2
    # H_dis_4 = layers.fully_connected(tf.nn.dropout(H_dis_3n, keep_prob=dropout), num_outputs=opt.embed_size,
    #                                biases_initializer=biasInit, activation_fn=None, scope=prefix + 'dis_5',
    #                                reuse=is_reuse)
    # H_dis_4n = tf.nn.relu(H_dis_4) + H_dis_3
    # H_dis_5 = layers.fully_connected(tf.nn.dropout(H_dis_4n, keep_prob=dropout), num_outputs=opt.embed_size,
    #                                biases_initializer=biasInit, activation_fn=None, scope=prefix + 'dis_6',
    #                                reuse=is_reuse)
    # H_dis_5n = tf.nn.relu(H_dis_5) + H_dis_4
    # H_dis_6 = layers.fully_connected(tf.nn.dropout(H_dis_5n, keep_prob=dropout), num_outputs=opt.embed_size,
    #                                biases_initializer=biasInit, activation_fn=None, scope=prefix + 'dis_7',
    #                                reuse=is_reuse)
    # H_dis_6n = tf.nn.relu(H_dis_6) + H_dis_5
    # H_dis_7 = layers.fully_connected(tf.nn.dropout(H_dis_6n, keep_prob=dropout), num_outputs=opt.embed_size,
    #                                biases_initializer=biasInit, activation_fn=None, scope=prefix + 'dis_8',
    #                                reuse=is_reuse)
    # H_dis_7n = tf.nn.relu(H_dis_7) + H_dis_6
    # H_dis_8 = layers.fully_connected(tf.nn.dropout(H_dis_7n, keep_prob=dropout), num_outputs=opt.embed_size,
    #                                biases_initializer=biasInit, activation_fn=None, scope=prefix + 'dis_9',
    #                                reuse=is_reuse)
    # H_dis_8n = tf.nn.relu(H_dis_8)
    

    logits = layers.linear(tf.nn.dropout(H_dis_3, keep_prob=dropout), num_outputs=num_outputs,
                           biases_initializer=biasInit, scope=prefix + 'dis_10', reuse=is_reuse)
    return logits


def conv_conv_model_3layer(X, opt, prefix='', is_reuse=None, num_outputs=None):
    # XX = tf.reshape(X, [-1, , 28, 1])
    # X shape: batchsize L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

    # generate filter for the first layer
    # W = tf.get_variable(prefix+'filter_w', [opt.sent_len, opt.filter_shape, opt.filter_size], initializer=weightInit)  # L 5 300
    # filters = tf.nn.relu(tf.squeeze(tf.tensordot(X, W, [[1], [0]])))  # batchsize emb 5 300
    # filters = tf.reshape(filters, [opt.batch_size, 1, 1, opt.embed_size*opt.filter_shape, opt.filter_size])

    # batchsize, opt.sent_len2, 1, opt.filter_shape*opt.embed_size
    embedding_patches = tf.extract_image_patches(X, [1, opt.filter_shape, opt.embed_size, 1], [1, opt.stride[0], 1, 1],
                                                 [1, 1, 1, 1], padding='VALID')
    H1 = tf.reduce_sum(tf.multiply(tf.expand_dims(embedding_patches, -1), filters), 3,
                       keep_dims=False)  # batch opt.sent_len2 1 Filtersize

    # pdb.set_trace()

    # H1 = layers.conv2d(X, num_outputs=opt.filter_size, kernel_size=[opt.filter_shape, opt.embed_size],
    #                    stride=[opt.stride[0], 1], weights_initializer=weightInit,
    #                    biases_initializer=biasInit, activation_fn=tf.nn.relu, padding='VALID', scope=prefix + 'H1_3',
    #                    reuse=is_reuse)  # batch L-3 1 Filtersize
    H2 = layers.conv2d(H1, num_outputs=opt.filter_size * 2, kernel_size=[opt.filter_shape, 1],
                       stride=[opt.stride[1], 1], biases_initializer=biasInit, activation_fn=tf.nn.relu,
                       padding='VALID', scope=prefix + 'H2_3', reuse=is_reuse)
    # print H2.get_shape()
    acf = tf.nn.tanh if opt.tanh else None
    H3 = layers.conv2d(H2, num_outputs=(num_outputs if num_outputs else opt.n_gan), kernel_size=[opt.sent_len3, 1],
                       biases_initializer=biasInit, activation_fn=acf, padding='VALID',
                       scope=prefix + 'H3_3', reuse=is_reuse)  # batch 1 1 2*Filtersize
    return H3


def mlp_conv_model_3layer(X, opt, prefix='', is_reuse=None, num_outputs=None):
    # XX = tf.reshape(X, [-1, , 28, 1])
    # X shape: batchsize L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

    # generate filter for the first layer
    W = tf.get_variable(prefix + 'filter_w', [opt.sent_len, opt.filter_shape, opt.filter_size],
                        initializer=weightInit)  # L 5 300
    b = tf.get_variable(prefix + 'filter_b', [1, 1, opt.filter_shape, opt.filter_size],
                        initializer=weightInit)  # L 5 300
    filters = tf.squeeze(tf.tensordot(X, W, [[1], [0]])) + b  # batchsize emb 5 300
    filters = tf.nn.relu(filters)
    filters = tf.reshape(filters, [opt.batch_size, 1, 1, opt.embed_size * opt.filter_shape, opt.filter_size])

    # batchsize, opt.sent_len2, 1, opt.filter_shape*opt.embed_size
    embedding_patches = tf.extract_image_patches(X, [1, opt.filter_shape, opt.embed_size, 1], [1, opt.stride[0], 1, 1],
                                                 [1, 1, 1, 1], padding='VALID')
    H1 = tf.reduce_sum(tf.multiply(tf.expand_dims(embedding_patches, -1), filters), 3,
                       keep_dims=False)  # batch opt.sent_len2 1 Filtersize

    # pdb.set_trace()

    # H1 = layers.conv2d(X, num_outputs=opt.filter_size, kernel_size=[opt.filter_shape, opt.embed_size],
    #                    stride=[opt.stride[0], 1], weights_initializer=weightInit,
    #                    biases_initializer=biasInit, activation_fn=tf.nn.relu, padding='VALID', scope=prefix + 'H1_3',
    #                    reuse=is_reuse)  # batch L-3 1 Filtersize
    H2 = layers.conv2d(H1, num_outputs=opt.filter_size * 2, kernel_size=[opt.filter_shape, 1],
                       stride=[opt.stride[1], 1], biases_initializer=biasInit, activation_fn=tf.nn.relu,
                       padding='VALID', scope=prefix + 'H2_3', reuse=is_reuse)
    # print H2.get_shape()
    acf = tf.nn.tanh if opt.tanh else None
    H3 = layers.conv2d(H2, num_outputs=(num_outputs if num_outputs else opt.n_gan), kernel_size=[opt.sent_len3, 1],
                       biases_initializer=biasInit, activation_fn=acf, padding='VALID',
                       scope=prefix + 'H3_3', reuse=is_reuse)  # batch 1 1 2*Filtersize
    return H3


def linear_conv_model_3layer(X, opt, prefix='', is_reuse=None, num_outputs=None):
    # XX = tf.reshape(X, [-1, , 28, 1])
    # X shape: batchsize L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

    # generate filter for the first layer
    W = tf.get_variable(prefix + 'filter_w', [opt.sent_len, opt.filter_shape, opt.filter_size],
                        initializer=weightInit)  # L 5 300
    filters = tf.squeeze(tf.tensordot(X, W, [[1], [0]]))  # batchsize emb 5 300
    filters = tf.reshape(filters, [opt.batch_size, 1, 1, opt.embed_size * opt.filter_shape, opt.filter_size])

    # batchsize, opt.sent_len2, 1, opt.filter_shape*opt.embed_size
    embedding_patches = tf.extract_image_patches(X, [1, opt.filter_shape, opt.embed_size, 1], [1, opt.stride[0], 1, 1],
                                                 [1, 1, 1, 1], padding='VALID')
    H1 = tf.reduce_sum(tf.multiply(tf.expand_dims(embedding_patches, -1), filters), 3,
                       keep_dims=False)  # batch opt.sent_len2 1 Filtersize

    # pdb.set_trace()

    # H1 = layers.conv2d(X, num_outputs=opt.filter_size, kernel_size=[opt.filter_shape, opt.embed_size],
    #                    stride=[opt.stride[0], 1], weights_initializer=weightInit,
    #                    biases_initializer=biasInit, activation_fn=tf.nn.relu, padding='VALID', scope=prefix + 'H1_3',
    #                    reuse=is_reuse)  # batch L-3 1 Filtersize
    H2 = layers.conv2d(H1, num_outputs=opt.filter_size * 2, kernel_size=[opt.filter_shape, 1],
                       stride=[opt.stride[1], 1], biases_initializer=biasInit, activation_fn=tf.nn.relu,
                       padding='VALID', scope=prefix + 'H2_3', reuse=is_reuse)
    # print H2.get_shape()
    acf = tf.nn.tanh if opt.tanh else None
    H3 = layers.conv2d(H2, num_outputs=(num_outputs if num_outputs else opt.n_gan), kernel_size=[opt.sent_len3, 1],
                       biases_initializer=biasInit, activation_fn=acf, padding='VALID',
                       scope=prefix + 'H3_3', reuse=is_reuse)  # batch 1 1 2*Filtersize
    return H3


def conv_filter_pooling(X, opt, prefix='', is_reuse=None, num_outputs=None):
    # XX = tf.reshape(X, [-1, , 28, 1])
    # X shape: batchsize L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

    # generate filters for the conv layer
    H_enc = conv_pooling(X, opt, prefix='gen_filter', is_reuse=None, num_outputs=None)

    # TensorShape([Dimension(50), Dimension(5), Dimension(100), Dimension(100)])
    filters = layers.conv2d_transpose(H_enc, num_outputs=opt.filter_size,
                                      kernel_size=[opt.filter_shape, opt.embed_size],
                                      biases_initializer=biasInit, activation_fn=tf.nn.relu,
                                      padding='VALID', scope=prefix + 'filter_dcnn', reuse=is_reuse)

    # input: [batch, in_depth, in_height, in_width, in_channels]
    # filter: [filter_depth, filter_height, filter_width, in_channels, out_channels]
    # H1 = tf.nn.conv3d(input=tf.expand_dims(X, 0), filter=tf.expand_dims(filters, 3), strides=[1, 1, opt.stride[0], 1, 1], padding='VALID')

    # batchsize, opt.sent_len2, 1, opt.filter_shape*opt.embed_size (* 1)
    embedding_patches = tf.extract_image_patches(X, [1, opt.filter_shape, opt.embed_size, 1], [1, opt.stride[0], 1, 1],
                                                 [1, 1, 1, 1], padding='VALID')

    # filters = tf.reshape(filters, [tf.shape(filters)[0], 1, 1, opt.filter_shape*opt.embed_size, opt.filter_size])
    filters = tf.reshape(filters, [opt.batch_size, 1, 1, opt.filter_shape * opt.embed_size, opt.filter_size])
    # filters = tf.nn.softmax(filters**opt.L, dim=3)
    H1 = tf.reduce_sum(tf.multiply(tf.expand_dims(embedding_patches, -1), filters), 3, keep_dims=False)  # batch opt.sent_len2 1 Filtersize

    H = tf.nn.max_pool(H1, [1, opt.sent_len2, 1, 1], [1, 1, 1, 1], 'VALID')

    #pdb.set_trace()

    return H


def conv_filter_attention(X, X_2, opt, prefix='', is_reuse=None, num_outputs=None):
    # XX = tf.reshape(X, [-1, , 28, 1])
    # X shape: batchsize L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

    # generate filters for the conv layer
    H_enc = conv_pooling(X, opt, prefix='gen_filter', is_reuse=None, num_outputs=None)

    # TensorShape([Dimension(50), Dimension(5), Dimension(100), Dimension(100)])
    filters = layers.conv2d_transpose(H_enc, num_outputs=opt.filter_size,
                                      kernel_size=[opt.filter_shape, opt.embed_size],
                                      biases_initializer=biasInit, activation_fn=tf.nn.relu,
                                      padding='VALID', scope=prefix + 'filter_dcnn', reuse=is_reuse)

    # input: [batch, in_depth, in_height, in_width, in_channels]
    # filter: [filter_depth, filter_height, filter_width, in_channels, out_channels]
    # H1 = tf.nn.conv3d(input=tf.expand_dims(X, 0), filter=tf.expand_dims(filters, 3), strides=[1, 1, opt.stride[0], 1, 1], padding='VALID')

    # batchsize, opt.sent_len2, 1, opt.filter_shape*opt.embed_size (* 1)
    embedding_patches = tf.extract_image_patches(X_2, [1, opt.filter_shape, opt.embed_size, 1],
                                                 [1, opt.stride[0], 1, 1],
                                                 [1, 1, 1, 1], padding='VALID')

    filters = tf.reshape(filters, [tf.shape(filters)[0], 1, 1, opt.filter_shape * opt.embed_size, opt.filter_size])
    # filters = tf.reshape(filters, [opt.batch_size, 1, 1, opt.filter_shape * opt.embed_size, opt.filter_size])
    # filters = tf.nn.softmax(filters**opt.L, dim=3)
    H1 = tf.reduce_sum(tf.multiply(tf.expand_dims(embedding_patches, -1), filters), 3,
                       keep_dims=False)  # batch opt.sent_len2 1 Filtersize
    ## need to add nonlinearity? or stress this advantage

    H = tf.nn.max_pool(H1, [1, opt.sent_len2, 1, 1], [1, 1, 1, 1], 'VALID')

    return H_enc, H


def gated_cnn(X, opt, prefix='', is_reuse=None, num_outputs=None):
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

    linear = layers.conv2d(X, num_outputs=(num_outputs if num_outputs else opt.n_gan),
                           kernel_size=[opt.filter_shape, opt.embed_size],
                           stride=[opt.stride[0], 1], weights_initializer=weightInit,
                           biases_initializer=biasInit, activation_fn=tf.nn.relu, padding='VALID',
                           scope=prefix + 'conv',
                           reuse=is_reuse)  # batch L-3 1 Filtersize

    gate = layers.conv2d(X, num_outputs=(num_outputs if num_outputs else opt.n_gan),
                         kernel_size=[opt.filter_shape, opt.embed_size],
                         stride=[opt.stride[0], 1], weights_initializer=weightInit,
                         biases_initializer=biasInit, activation_fn=None, padding='VALID', scope=prefix + 'gate',
                         reuse=is_reuse)  # batch L-3 1 Filtersize
    gate = tf.nn.sigmoid(gate)
    H_enc = tf.multiply(linear, gate, name='gated')

    H = tf.reduce_sum(H_enc, axis=1, keep_dims=True)

    return H


def context_gate(X, opt, prefix='', is_reuse=None, num_outputs=None):
    # X: # batch L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

    # pdb.set_trace()

    X = tf.squeeze(X)  # batch L emb
    gate = layers.linear(X, num_outputs=opt.embed_size, biases_initializer=biasInit, scope=prefix + 'gate',
                         reuse=is_reuse)  # batch L emb
    gate = tf.nn.sigmoid(gate)
    H = tf.multiply(X, gate, name='gated')  # batch L emb
    H = tf.expand_dims(H, axis=-1)

    H_enc = conv_pooling(H, opt, prefix='conv', is_reuse=None, num_outputs=None)

    return H_enc


def soft_att_layer(X, opt, prefix='', is_reuse=None):
    # X: B L E 1
    sent_length = X.get_shape().as_list()[1]
    emb_size = X.get_shape().as_list()[2]
    with tf.variable_scope(prefix + 'soft_att', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        V0 = tf.get_variable('V0', [1, emb_size], initializer=weightInit)

    # pdb.set_trace()
    V_norm = tf.tile(normalizing(V0, 1), [opt.batch_size, 1])  # B E
    V_norm = tf.expand_dims(V_norm, 2)  # B E 1

    prob_logits = tf.squeeze(tf.matmul(tf.squeeze(X), V_norm))  # B L

    prob = tf.nn.softmax(prob_logits * opt.att_L, dim=-1, name=None)

    prob = tf.expand_dims(prob, 2)
    prob = tf.expand_dims(prob, 2)
    return X * prob * sent_length, tf.squeeze(prob)


def attention_1(X, opt, prefix='', is_reuse=None):
    # X: B L E 1
    sent_length = X.get_shape().as_list()[1]
    emb_size = X.get_shape().as_list()[2]
    with tf.variable_scope(prefix + 'soft_att', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        V0 = tf.get_variable('V0', [1, emb_size], initializer=weightInit)

    # pdb.set_trace()
    V_norm = tf.tile(normalizing(V0, 1), [opt.batch_size, 1])  # B E
    V_norm = tf.expand_dims(V_norm, 2)  # B E 1

    prob_logits = tf.squeeze(tf.matmul(tf.squeeze(X), V_norm))  # B L
    prob = tf.nn.softmax(prob_logits, dim=-1, name=None)  # B L

    prob = tf.expand_dims(prob, 2)
    prob = tf.expand_dims(prob, 2)  # B L 1 1
    return X * prob * sent_length, tf.squeeze(prob)

    # # X: batch L emb 1
    # weightInit = tf.random_uniform_initializer(-0.001, 0.001)
    # biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    #
    # X = tf.squeeze(X)  # batch L emb
    # mu = layers.fully_connected(X, num_outputs=opt.embed_size, biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'trans', euse=is_reuse)  # batch L emb
    # U = tf.get_variable('U', [opt.embed_size], initializer=weightInit)  # emb

def weighted_average(X, opt, prefix='', is_reuse=None, num_outputs=None):
    # X: # batch L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

    # pdb.set_trace()

    X = tf.squeeze(X)  # batch L emb
    X_trans = layers.fully_connected(X, num_outputs=opt.embed_size, biases_initializer=biasInit,
                                     activation_fn=tf.nn.relu,
                                     scope=prefix + 'trans', reuse=is_reuse)  # batch L emb
    gate = layers.linear(X_trans, num_outputs=1, biases_initializer=None, scope=prefix + 'gate',
                         reuse=is_reuse)  # batch L 1
    # gate = tf.nn.sigmoid(gate)
    gate = tf.nn.softmax(gate, dim=1)  # batch L 1
    H = tf.multiply(X, gate, name='gated')  # batch L emb
    # H = tf.expand_dims(H, axis=-1)
    H_enc = tf.reduce_mean(H, axis=1, keep_dims=False)  # batch emb

    return H_enc

def gated_average(X, opt, prefix='', is_reuse=None, num_outputs=None):
    # X: # batch L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        weightInit = tf.constant_initializer(0.001, dtype=tf.float32)

    # pdb.set_trace()

    X = tf.squeeze(X)  # batch L emb
    X_trans = layers.fully_connected(X, num_outputs=opt.embed_size, biases_initializer=biasInit,
                                     activation_fn=tf.nn.relu,
                                     scope=prefix + 'trans', reuse=is_reuse)  # batch L emb
    gate = layers.linear(X_trans, num_outputs=1, biases_initializer=None, scope=prefix + 'gate', reuse=is_reuse)  # batch L 1
    gate = tf.nn.sigmoid(gate) # batch L 1
    #gate = tf.nn.softmax(gate, dim=1)  # batch L 1
    H = tf.multiply(X, gate, name='gated')  # batch L emb
    # H = tf.expand_dims(H, axis=-1)
    H_enc = tf.reduce_mean(H, axis=1, keep_dims=False)  # batch emb

    return H_enc, gate


def partial_softmax(logits, weights, dim, name,):
    with tf.name_scope('partial_softmax'):
        exp_logits = tf.exp(logits)
        if len(exp_logits.get_shape()) == len(weights.get_shape()):
            exp_logits_weighted = tf.multiply(exp_logits, weights)
        else:
            exp_logits_weighted = tf.multiply(exp_logits, tf.expand_dims(weights, -1))
        exp_logits_sum = tf.reduce_sum(exp_logits_weighted, axis=dim, keep_dims=True)
        partial_softmax_score = tf.div(exp_logits_weighted, exp_logits_sum, name=name)
        return partial_softmax_score

