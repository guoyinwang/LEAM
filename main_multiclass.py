# -*- coding: utf-8 -*-
"""
Guoyin Wang

LEAM
"""

import os, sys, cPickle
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import scipy.io as sio
from math import floor

from model import *
from utils import get_minibatches_idx, restore_from_save, tensors_key_in_file, prepare_data_for_emb, load_class_embedding

from evaluation import macro_f1, micro_f1, precision_at_k
from sklearn.metrics import roc_auc_score, f1_score

class Options(object):
    def __init__(self):
        self.GPUID = 0
        self.dataset = 'mimic'
        self.fix_emb = True
        self.restore = False
        self.W_emb = None
        self.W_class_emb = None
        self.maxlen = 305
        self.n_words = None
        self.embed_size = 300
        self.lr = 1e-3
        self.batch_size = 10
        self.max_epochs = 100
        self.dropout = 0.5
        self.part_data = False
        self.portion = 1.0 
        self.save_path = "./save/"
        self.log_path = "./log/"
        self.print_freq = 100
        self.valid_freq = 100

        self.optimizer = 'Adam'
        self.clip_grad = None
        self.class_penalty = 1.0
        self.ngram = 55
        self.H_dis = 300


    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

def emb_classifier(x, x_mask, y, dropout, opt, class_penalty):
    # comment notation
    #  b: batch size, s: sequence length, e: embedding dim, c : num of class
    x_emb, W_norm = embedding(x, opt)  #  b * s * e
    x_emb=tf.cast(x_emb,tf.float32)
    W_norm=tf.cast(W_norm,tf.float32)
    y_pos = tf.argmax(y, -1)
    y_emb, W_class = embedding_class(y_pos, opt, 'class_emb') # b * e, c * e
    y_emb=tf.cast(y_emb,tf.float32)
    W_class=tf.cast(W_class,tf.float32)
    W_class_tran = tf.transpose(W_class, [1,0]) # e * c
    x_emb = tf.expand_dims(x_emb, 3)  # b * s * e * 1
    H_enc = att_emb_ngram_encoder_cnn(x_emb, x_mask, W_class, W_class_tran, opt)
    H_enc_list= tf.unstack(H_enc, axis=-1)
    logits_list = []
    for i, ih in enumerate(H_enc_list):
        logits_list.append(discriminator_0layer(ih, opt, dropout, prefix='classify_{}'.format(i), num_outputs=1, is_reuse=False) )
    
    logits = tf.concat(logits_list,-1)
    prob = tf.nn.softmax(logits)
    # class_y = tf.constant(name='class_y', shape=[opt.num_class, opt.num_class], dtype=tf.float32, value=np.identity(opt.num_class),)
    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

    global_step = tf.Variable(0, trainable=False)
    train_op = layers.optimize_loss(
        loss,
        global_step=global_step,
        optimizer=opt.optimizer,
        learning_rate=opt.lr)

    return accuracy, loss, train_op, W_norm, global_step, logits, prob


def main():
    # Prepare training and testing data
    opt = Options()
    # load data
    loadpath = "./data/mimic3.p"
    embpath = "mimic3_emb.p"
    opt.num_class = 50
    x = cPickle.load(open(loadpath, "rb"))
    train, train_text, train_lab = x[0],x[1], x[2]
    val, val_text, val_lab = x[3],x[4], x[5]
    test, test_text, test_lab = x[6],x[7], x[8]
    wordtoix, ixtoword = x[10], x[9]
    del x
    print("load data finished")

    train_lab = np.array(train_lab, dtype='float32')
    val_lab = np.array(val_lab, dtype='float32')
    test_lab = np.array(test_lab, dtype='float32')    
    opt.n_words = len(ixtoword)
    if opt.part_data:
        #np.random.seed(123)
        train_ind = np.random.choice(len(train), int(len(train)*opt.portion), replace=False)
        train = [train[t] for t in train_ind]
        train_lab = [train_lab[t] for t in train_ind]
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.GPUID)

    print(dict(opt))
    print('Total words: %d' % opt.n_words)

    try:
        opt.W_emb = np.array(cPickle.load(open(embpath, 'rb')),dtype='float32')
        opt.W_class_emb =  load_class_embedding( wordtoix, opt)
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/gpu:1'):
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.maxlen],name='x_')
        x_mask_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.maxlen],name='x_mask_')
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        y_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.num_class],name='y_')
        class_penalty_ = tf.placeholder(tf.float32, shape=())
        accuracy_, loss_, train_op, W_norm_, global_step, logits_, prob_ = emb_classifier(x_, x_mask_, y_, keep_prob, opt, class_penalty_)
    uidx = 0
    max_val_accuracy = 0.
    max_test_accuracy = 0.
    max_val_auc_mean = 0.
    max_test_auc_mean = 0.

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, )
    config.gpu_options.allow_growth = True
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        if opt.restore:
            try:
                t_vars = tf.trainable_variables()
                save_keys = tensors_key_in_file(opt.save_path)
                ss = set([var.name for var in t_vars]) & set([s + ":0" for s in save_keys.keys()])
                cc = {var.name: var for var in t_vars}
                # only restore variables with correct shape
                ss_right_shape = set([s for s in ss if cc[s].get_shape() == save_keys[s[:-2]]])

                loader = tf.train.Saver(var_list=[var for var in t_vars if var.name in ss_right_shape])
                loader.restore(sess, opt.save_path)

                print("Loading variables from '%s'." % opt.save_path)
                print("Loaded variables:" + str(ss))

            except:
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())

        try:
            for epoch in range(opt.max_epochs):
                print("Starting epoch %d" % epoch)
                kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
                for _, train_index in kf:
                    uidx += 1
                    sents = [train[t] for t in train_index]
                    x_labels = [train_lab[t] for t in train_index]
                    x_labels = np.array(x_labels)
                    x_labels = x_labels.reshape((len(x_labels), opt.num_class))

                    x_batch, x_batch_mask = prepare_data_for_emb(sents, opt)
                    _, loss, step,  = sess.run([train_op, loss_, global_step], feed_dict={x_: x_batch, x_mask_: x_batch_mask, y_: x_labels, keep_prob: opt.dropout, class_penalty_:opt.class_penalty})

                    if uidx % opt.valid_freq == 0:
                        train_correct = 0.0
                        # sample evaluate accuaccy on 500 sample data
                        kf_train = get_minibatches_idx(500, opt.batch_size, shuffle=True)
                        for _, train_index in kf_train:
                            train_sents = [train[t] for t in train_index]
                            train_labels = [train_lab[t] for t in train_index]
                            train_labels = np.array(train_labels)
                            train_labels = train_labels.reshape((len(train_labels), opt.num_class))
                            x_train_batch, x_train_batch_mask = prepare_data_for_emb(train_sents, opt)  
                            train_accuracy = sess.run(accuracy_, feed_dict={x_: x_train_batch, x_mask_: x_train_batch_mask, y_: train_labels, keep_prob: 1.0, class_penalty_:0.0})

                            train_correct += train_accuracy * len(train_index)

                        train_accuracy = train_correct / 500

                        print("Iteration %d: Training loss %f " % (uidx, loss))
                        print("Train accuracy %f " % train_accuracy)

                        val_correct = 0.0
                        val_y = []
                        val_logits_list = []
                        val_prob_list = []
                        val_true_list = []

                        kf_val = get_minibatches_idx(len(val), opt.batch_size, shuffle=True)
                        for _, val_index in kf_val:
                            val_sents = [val[t] for t in val_index]
                            val_labels = [val_lab[t] for t in val_index]
                            val_labels = np.array(val_labels)
                            val_labels = val_labels.reshape((len(val_labels), opt.num_class))
                            x_val_batch, x_val_batch_mask = prepare_data_for_emb(val_sents, opt)
                            val_accuracy, val_logits, val_probs = sess.run([accuracy_, logits_, prob_], feed_dict={x_: x_val_batch, x_mask_: x_val_batch_mask,
                                y_: val_labels, keep_prob: 1.0,
                                class_penalty_:0.0                                         })

                            val_correct += val_accuracy * len(val_index)
                            val_y += np.argmax(val_labels, axis=1).tolist()
                            val_logits_list += val_logits.tolist()
                            val_prob_list += val_probs.tolist()
                            val_true_list += val_labels.tolist()

                        val_accuracy = val_correct / len(val)
                        val_logits_array = np.asarray(val_logits_list)
                        val_prob_array = np.asarray(val_prob_list)
                        val_true_array = np.asarray(val_true_list)
                        val_auc_list = []
                        val_auc_micro = roc_auc_score(y_true=val_true_array, y_score =val_logits_array,average='micro')
                        val_auc_macro = roc_auc_score(y_true=val_true_array, y_score =val_logits_array,average='macro')
                        for i in range(opt.num_class):
                            if np.max(val_true_array[:,i] > 0):
                                val_auc = roc_auc_score(y_true = val_true_array[:,i], y_score= val_logits_array[:,i],)
                                val_auc_list.append(val_auc)
                        val_auc_mean = np.mean(val_auc)

                        # print("Validation accuracy %f " % val_accuracy)
                        print("val auc macro %f micro %f " % (val_auc_macro, val_auc_micro))

                        if True:
                            test_correct = 0.0
                            test_y = []
                            test_logits_list = []
                            test_prob_list = []
                            test_true_list = []
                            
                            kf_test = get_minibatches_idx(len(test), opt.batch_size, shuffle=True)
                            for _, test_index in kf_test:
                                test_sents = [test[t] for t in test_index]
                                test_labels = [test_lab[t] for t in test_index]
                                test_labels = np.array(test_labels)
                                test_labels = test_labels.reshape((len(test_labels), opt.num_class))
                                x_test_batch, x_test_batch_mask = prepare_data_for_emb(test_sents, opt)

                                test_accuracy, test_logits, test_probs= sess.run([accuracy_, logits_, prob_],feed_dict={x_: x_test_batch, x_mask_: x_test_batch_mask,y_: test_labels, keep_prob: 1.0, class_penalty_: 0.0})

                                test_correct += test_accuracy * len(test_index)   

                                test_correct += test_accuracy * len(test_index)
                                test_y += np.argmax(test_labels, axis=1).tolist()
                                test_logits_list += test_logits.tolist()
                                test_prob_list += test_probs.tolist()
                                test_true_list += test_labels.tolist()                             
                            test_accuracy = test_correct / len(test)
                            test_logits_array = np.asarray(test_logits_list)
                            test_prob_array = np.asarray(test_prob_list)
                            test_true_array = np.asarray(test_true_list)
                            test_auc_list = []
                            test_auc_micro = roc_auc_score(y_true=test_true_array, y_score =test_logits_array,average='micro')
                            test_auc_macro = roc_auc_score(y_true=test_true_array, y_score =test_logits_array,average='macro')

                            test_f1_micro = micro_f1(test_prob_array.ravel() >0.5, test_true_array.ravel(), )
                            test_f1_macro = macro_f1(test_prob_array > 0.5, test_true_array, )
                            test_p5 = precision_at_k(test_logits_array, test_true_array , 5)
                            
                            for i in range(opt.num_class):
                                if np.max(test_true_array[:,i] > 0):
                                    test_auc = roc_auc_score(y_true = test_true_array[:,i], y_score= test_logits_array[:,i],)
                                    test_auc_list.append(test_auc)

                            test_auc_mean = np.mean(test_auc)
                            print("Test auc macro %f micro %f " % (test_auc_macro, test_auc_micro))
                            print("Test f1 macro %f micro %f " % (test_f1_macro, test_f1_micro))
                            print("P5 %f" % test_p5)
                            # max_test_accuracy = test_accuracy
                            max_test_auc_mean = test_auc_mean
                            # print("Test accuracy %f " % test_accuracy)
                            # max_test_accuracy = test_accuracy

                # print("Epoch %d: Max Test accuracy %f" % (epoch, max_test_accuracy))
                print("Epoch %d: Max Test auc %f" % (epoch, max_test_auc_mean))
                saver.save(sess, opt.save_path, global_step=epoch)
            print("Max Test accuracy %f " % max_test_accuracy)

        except KeyboardInterrupt:
            print('Training interupted')
            print("Max Test accuracy %f " % max_test_accuracy)

if __name__ == '__main__':
    main()
