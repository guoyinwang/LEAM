import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

def embedding(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_emb'))
            assert (np.shape(np.array(opt.W_emb)) == (opt.n_words, opt.embed_size))
            W = tf.get_variable('W', initializer=opt.W_emb, trainable=True)
            print("initialize word embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer=weightInit)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)
    
    word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W

def embedding_class(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched y into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_class_emb'))
            W = tf.get_variable('W_class', initializer=opt.W_class_emb, trainable=True)
            print("initialize class embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W_class', [opt.num_class, opt.embed_size], initializer=weightInit)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)
    word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W

def att_emb_ngram_encoder_maxout(x_emb, x_mask, W_class, W_class_tran, opt):
    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2) # b * s * e
    W_class_norm = tf.nn.l2_normalize(W_class_tran, dim = 0) # e * c
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
    x_full_emb = x_emb_0
    Att_v = tf.contrib.layers.conv2d(G, num_outputs=opt.num_class,kernel_size=[opt.ngram], padding='SAME',activation_fn=tf.nn.relu) #b * s *  c

    Att_v = tf.reduce_max(Att_v, axis=-1, keep_dims=True)
    Att_v_max = partial_softmax(Att_v, x_mask, 1, 'Att_v_max') # b * s * 1

    x_att = tf.multiply(x_full_emb, Att_v_max)
    H_enc = tf.reduce_sum(x_att, axis=1)  
    return H_enc

def att_emb_ngram_encoder_cnn(x_emb, x_mask, W_class, W_class_tran, opt):
    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    H = tf.contrib.layers.conv2d(x_emb_0, num_outputs=opt.embed_size,kernel_size=[10], padding='SAME',activation_fn=tf.nn.relu) #b * s *  c


    G = tf.contrib.keras.backend.dot(H, W_class_tran) # b * s * c
    Att_v_max = partial_softmax(G, x_mask, 1, 'Att_v_max') # b * s * c

    x_att = tf.contrib.keras.backend.batch_dot(tf.transpose(H,[0,2,1]), Att_v_max)
    H_enc = tf.squeeze(x_att)
    return H_enc


def aver_emb_encoder(x_emb, x_mask):
    """ compute the average over all word embeddings """
    x_mask = tf.expand_dims(x_mask, axis=-1)
    # x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_sum = tf.multiply(x_emb, x_mask)  # batch L emb 
    H_enc_0 = tf.reduce_sum(x_sum, axis=1, keep_dims=True)  # batch 1 emb 
    H_enc = tf.squeeze(H_enc_0, [1,])  # batch emb
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 
    x_mask_sum = tf.squeeze(x_mask_sum, [2,])  # batch 1

    #pdb.set_trace()

    H_enc = H_enc / x_mask_sum  # batch emb

    return H_enc

def gru_encoder(X_emb, opt, prefix = '', is_reuse=None, res=None):
    with tf.variable_scope(prefix + 'gru_encoder', reuse=True):
        cell_fw = tf.contrib.rnn.GRUCell(opt.n_hid)
        cell_bw = tf.contrib.rnn.GRUCell(opt.n_hid)
    with tf.variable_scope(prefix + 'gru_encoder', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        
        packed_output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, X_emb , dtype = tf.float32)
        h_fw = state[0]
        h_bw = state[1]
  
        hidden = tf.concat((h_fw, h_bw), 1)
        
        hidden = tf.nn.l2_normalize(hidden, 1)
    return hidden, res

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

def linear_layer(x, output_dim):
    input_dim = x.get_shape().as_list()[1]
    thres = np.sqrt(6.0 / (input_dim + output_dim))
    W = tf.get_variable("W", [input_dim, output_dim], scope=prefix + '_W',
                        initializer=tf.random_uniform_initializer(minval=-thres, maxval=thres))
    b = tf.get_variable("b", [output_dim], scope=prefix + '_b', initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, W) + b


def discriminator_2layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    logits = layers.linear(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=num_outputs,
                           biases_initializer=biasInit, scope=prefix + 'dis_2', reuse=is_reuse)
    return logits

def discriminator_3layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
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

