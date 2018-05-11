# -*- coding: utf-8 -*-
"""
Yizhe Zhang

TextCNN
"""
## 152.3.214.203/6006

import os
GPUID = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
# from tensorflow.contrib import metrics
# from tensorflow.contrib.learn import monitors
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
# from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
import cPickle
import numpy as np
import os
import sys
import scipy.io as sio
from math import floor
import pdb

from model import *
from utils import prepare_data_for_cnn, prepare_data_for_rnn, get_minibatches_idx, normalizing, restore_from_save, \
    prepare_for_bleu, cal_BLEU, sent2idx, tensors_key_in_file, prepare_data_for_emb, load_embedding_vectors_glove_gensim, load_class_embedding, _clip_gradients_seperate_norm, embedding_view, embedding_center

# import tempfile
# from tensorflow.examples.tutorials.mnist import input_data

logging.set_verbosity(logging.INFO)
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS


# flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

class Options(object):
    def __init__(self):
        self.fix_emb = True
        self.reuse_w = False
        self.reuse_cnn = False
        self.reuse_discrimination = False  # reuse cnn for discrimination
        self.restore = True
        self.tanh = False  # activation fun for the top layer of cnn, otherwise relu
        self.model = 'cnn_deconv'  # 'cnn_rnn', 'rnn_rnn' , default: cnn_deconv

        self.permutation = 0
        self.substitution = 's'  # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)

        self.W_emb = None
        self.W_class_emb = None
        self.cnn_W = None
        self.cnn_b = None
        self.maxlen = 153
        self.n_words = None
        self.filter_shape = 5
        #self.filter_size = 300
        self.embed_size = 300
        self.lr = 1e-3#2e-4
        self.layer = 3
        self.stride = [2, 2]  # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = 100
        self.max_epochs = 1000
        self.n_gan = 500  # self.filter_size * 3
        self.L = 100
        self.drop_rate = 1.0

        self.part_data = False
        self.portion = 1.0   # 10%  1%  float(sys.argv[1])

        self.save_path = "./save/dbpedia_emb_plot"
        self.log_path = "./log/dbpedia_emb_plot"
        self.print_freq = 100
        self.valid_freq = 100

        self.discrimination = False
        self.dropout = 0.5
        self.H_dis = 300
        self.num_class = 14
        self.class_name = ['Company',
            'Educational Institution',
            'Artist',
            'Athlete',
            'Office Holder',
            'Mean Of Transportation',
            'Building',
            'Natural Place',
            'Village',
            'Animal',
            'Plant',
            'Album',
            'Film',
            'Written Work',
            ]

        self.optimizer = 'Adam'
        self.clip_grad = None

        self.sent_len = self.maxlen + 2 * (self.filter_shape - 1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape) / self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape) / self.stride[1]) + 1)
        # self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape)/self.stride[2]) + 1)
        print ('Use model %s' % self.model)
        print ('Use %d conv/deconv layers' % self.layer)

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value


def emb_classifier(x, x_mask, y, dropout, opt, class_penalty):
    # print x.get_shape()  # batch L
    x_emb, W_norm = embedding(x, opt)  # batch L emb b * s * e
    y_pos = tf.argmax(y, -1)
    y_emb, W_class = embedding_class(y_pos, opt, 'class_emb') # b * e, c * e
    W_class_tran = tf.transpose(W_class, [1,0]) # e * c
    x_emb = tf.expand_dims(x_emb, 3)  # batch L emb 1  b * s * e * 1
    # x_emb = tf.nn.dropout(x_emb, dropout)   # batch L emb 1 b * s * e * 1

    # x_mask = tf.expand_dims(x_mask, axis=-1)
    # x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    # x_sum = tf.multiply(x_emb, x_mask)  # batch L emb 1
    # H_enc = tf.reduce_sum(x_sum, axis=1, keep_dims=True)  # batch 1 emb 1
    # H_enc = tf.squeeze(H_enc)  # batch emb
    # x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1
    # x_mask_sum = tf.squeeze(x_mask_sum, [2, 3])  # batch 1

    # H_enc_1 = H_enc / x_mask_sum  # batch emb
    # H_enc_1 = aver_emb_encoder(x_emb, x_mask)
    # H_enc_1, Att_h, x_emb_norm, W_class_norm,  = att_dev_emb_encoder(x_emb, x_mask, W_class, W_class_tran, opt)
    # H_enc_1, Att_h, x_emb_norm, W_class_norm,  = att_G_emb_encoder(x_emb, x_mask, W_class, W_class_tran, opt)

    # H_enc_2 = tf.nn.max_pool(x_emb, [1, opt.maxlen, 1, 1], [1, 1, 1, 1], 'VALID')
    # H_enc_2 = tf.squeeze(H_enc_2)

    # H_enc_2 = max_emb_encoder(x_emb, x_mask, opt)

    # H_enc = tf.concat([H_enc_1, H_enc_2], 1)
    H_enc, Att_h, Att_v = att_emb_ngram_encoder_maxout(x_emb, x_mask, W_class, W_class_tran, opt)
    # H_enc = H_enc_1

    # if opt.layer == 3:
    #     H_enc = conv_model_3layer(x_emb, opt)
    # else:
    #     H_enc = conv_model(x_emb, opt)

    # biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    # x_emb = layers.fully_connected(tf.squeeze(x_emb), num_outputs=opt.embed_size, biases_initializer=biasInit,
    #                                  activation_fn=tf.nn.relu, scope='trans', reuse=True)  # batch L emb

    #x_emb = tf.expand_dims(x_emb, 3)  # batch L emb 1

    # MAX POOLING
    #H_enc = tf.nn.max_pool(x_emb, [1, opt.maxlen, 1, 1], [1, 1, 1, 1], 'VALID')
    #H_enc = tf.reduce_mean(x_emb, axis=1, keep_dims=False)  # batch L emb 1

    H_enc = tf.squeeze(H_enc)
    logits = discriminator_2layer(H_enc, opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=False)  # batch * 10
    # logits_class = discriminator_2layer(tf.expand_dims(tf.concat([W_class, W_class], -1), 0), opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=True)
    # logits_class = discriminator_2layer(tf.expand_dims(10 * tf.matmul(tf.nn.l2_normalize(W_class, dim = 1), tf.nn.l2_normalize(W_class_tran, dim = 0)), 0), opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=True)
    # logits_class = discriminator_2layer(tf.expand_dims(tf.matmul(W_class, W_class_tran), 0), opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=True)
    prob = tf.nn.softmax(logits)
    # prob_class = tf.nn.softmax(logits_class)
    # class_y = tf.constant(name='class_y', shape=[opt.num_class, opt.num_class], dtype=tf.float32, value=np.identity(opt.num_class),)
    # class_y = tf.expand_dims(class_y,0)

    #pdb.set_trace()

    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)) + class_penalty * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=class_y, logits=logits_class))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

    #pdb.set_trace()

    # *tf.cast(tf.not_equal(x_temp,0), tf.float32)
    # tf.summary.scalar('loss', loss)

    #t_vars = tf.trainable_variables()
    #d_vars = [var for var in t_vars if 'dis_' in var.name]

    #pdb.set_trace()

    global_step = tf.Variable(0, trainable=False)
    train_op = layers.optimize_loss(
        loss,
        # global_step=framework.get_global_step(),
        global_step=global_step,
        optimizer=opt.optimizer,
        # clip_gradients=(lambda grad: _clip_gradients_seperate_norm(grad, opt.clip_grad)) if opt.clip_grad else None,
        # learning_rate_decay_fn=lambda lr,g: tf.train.exponential_decay(learning_rate=lr, global_step=g, decay_rate=opt.decay_rate, decay_steps=3000),
        #variables=d_vars,
        learning_rate=opt.lr)

    # etra results
    Att_emb = tf.contrib.keras.backend.dot(W_norm, W_class_tran)

    return accuracy, loss, train_op, W_norm, global_step, Att_h, Att_v, H_enc, W_class


def main():
    # global n_words
    # Prepare training and testing data
    loadpath = "./data/dbpedia.p"
    embpath = "./data/dbpedia_glove.p"
    x = cPickle.load(open(loadpath, "rb"))
    train, val, test = x[0], x[1], x[2]
    train_lab, val_lab, test_lab = x[3], x[4], x[5]
    wordtoix, ixtoword = x[6], x[7]

    print("load data finished")

    #pdb.set_trace()

    #train = np.concatenate([train, val])
    #train_lab = np.concatenate([train_lab, val_lab])

    train_lab = np.array(train_lab, dtype='float32')
    val_lab = np.array(val_lab, dtype='float32')
    test_lab = np.array(test_lab, dtype='float32')

    #pdb.set_trace()

    opt = Options()
    opt.n_words = len(ixtoword)

    del x

    print(dict(opt))
    print('Total words: %d' % opt.n_words)

    # self.part_data = True
    # self.num_data = 14000

    if opt.part_data:
        #np.random.seed(123)
        train_ind = np.random.choice(len(train), int(len(train)*opt.portion), replace=False)
        train = [train[t] for t in train_ind]
        train_lab = [train_lab[t] for t in train_ind]

    # try:
    #     params = np.load('./param_g.npz')
    #     if params['Wemb'].shape == (opt.n_words, opt.embed_size):
    #         print('Use saved embedding.')
    #         opt.W_emb = params['Wemb']
    #     else:
    #         print('Emb Dimension mismatch: param_g.npz:' + str(params['Wemb'].shape) + ' opt: ' + str(
    #             (opt.n_words, opt.embed_size)))
    #         opt.fix_emb = False
    try:
        # opt.W_emb = cPickle.load(open(embpath, 'rb'))[0]
        opt.W_emb = cPickle.load(open(embpath, 'rb')).astype(np.float32)
        opt.W_class_emb =  load_class_embedding( wordtoix, opt)


    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/gpu:1'):
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.maxlen])
        x_mask_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.maxlen])
        keep_prob = tf.placeholder(tf.float32)
        y_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.num_class])
        class_penalty_ = tf.placeholder(tf.float32, shape=())
        # accuracy_, loss_, train_op, W_norm, global_step , Att_h_, H_enc_, W_class_, x_emb_norm_, W_class_norm_, Att_emb_= emb_classifier(x_, x_mask_, y_, keep_prob, opt, class_penalty_)
        accuracy_, loss_, train_op, W_norm, global_step, Att_h_, Att_v_ ,H_enc_, W_class_= emb_classifier(x_, x_mask_, y_, keep_prob, opt, class_penalty_)
        # merged = tf.summary.merge_all()

    uidx = 0
    max_val_accuracy = 0.
    max_test_accuracy = 0.
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
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
                # print([var.name[:-2] for var in t_vars])
                save_keys = tensors_key_in_file(opt.save_path)
                # print(save_keys.keys())
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
                # if epoch >= 10:
                #     print("Relax embedding ")
                #     opt.fix_emb = False
                #     opt.batch_size = 2
                kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
                for _, train_index in kf:
                    uidx += 1
                    sents = [train[t] for t in train_index]
                    x_labels = [train_lab[t] for t in train_index]
                    x_labels = np.array(x_labels)
                    x_labels = x_labels.reshape((len(x_labels), opt.num_class))

                    x_batch, x_batch_mask = prepare_data_for_emb(sents, opt)
                    # x_print = sess.run([x_emb],feed_dict={x_: x_train} )
                    # print x_print
                    # if 'step' in locals():
                    #     class_penalty = np.max([0.0, 1.0 - 1.0 * step / 14000])
                    # else:
                    #     class_penalty = 1.0

                    # _, loss, step,  = sess.run([train_op, loss_, global_step], feed_dict={x_: x_batch, x_mask_: x_batch_mask, y_: x_labels, keep_prob: opt.dropout, class_penalty_:class_penalty})
                    _, loss, step,  = sess.run([train_op, loss_, global_step], feed_dict={x_: x_batch, x_mask_: x_batch_mask, y_: x_labels, keep_prob: opt.dropout})

                    # pdb.set_trace()

                    if uidx % opt.valid_freq == 0:
                        train_correct = 0.0
                        kf_train = get_minibatches_idx(500, opt.batch_size, shuffle=True)
                        for _, train_index in kf_train:
                            train_sents = [train[t] for t in train_index]
                            train_labels = [train_lab[t] for t in train_index]
                            train_labels = np.array(train_labels)
                            train_labels = train_labels.reshape((len(train_labels), opt.num_class))
                            x_train_batch, x_train_batch_mask = prepare_data_for_emb(train_sents, opt)  # Batch L

                            train_accuracy = sess.run(accuracy_, feed_dict={x_: x_train_batch, x_mask_: x_train_batch_mask, y_: train_labels, keep_prob: 1.0, class_penalty_:0.0})

                            train_correct += train_accuracy * len(train_index)

                        train_accuracy = train_correct / 500

                        print("Iteration %d: Training loss %f " % (uidx, loss))
                        print("Train accuracy %f " % train_accuracy)

                        val_correct = 0.0
                        kf_val = get_minibatches_idx(len(val), opt.batch_size, shuffle=True)
                        for _, val_index in kf_val:
                            val_sents = [val[t] for t in val_index]
                            val_labels = [val_lab[t] for t in val_index]
                            val_labels = np.array(val_labels)
                            val_labels = val_labels.reshape((len(val_labels), opt.num_class))
                            x_val_batch, x_val_batch_mask = prepare_data_for_emb(val_sents, opt)

                            val_accuracy = sess.run(accuracy_, feed_dict={x_: x_val_batch, x_mask_: x_val_batch_mask,
                                y_: val_labels, keep_prob: 1.0,
                                class_penalty_:0.0                                         })

                            val_correct += val_accuracy * len(val_index)

                        val_accuracy = val_correct / len(val)

                        #loss_val = sess.run(loss_, feed_dict={x_: x_val_batch, y_: val_labels, keep_prob: 1.0})
                        #print("Validation loss %f " % loss_val)
                        print("Validation accuracy %f " % val_accuracy)

                        if val_accuracy > max_val_accuracy:
                            max_val_accuracy = val_accuracy

                            test_correct = 0.0
                            test_emb = []
                            test_y = []
                            kf_test = get_minibatches_idx(len(test), opt.batch_size, shuffle=True)
                            for _, test_index in kf_test:
                                test_sents = [test[t] for t in test_index]
                                test_labels = [test_lab[t] for t in test_index]
                                test_labels = np.array(test_labels)
                                test_labels = test_labels.reshape((len(test_labels), opt.num_class))
                                x_test_batch, x_test_batch_mask = prepare_data_for_emb(test_sents, opt)

                                # test_accuracy, Att_h, H_enc, x_emb_norm, W_class_norm = sess.run([accuracy_, Att_h_, H_enc_, x_emb_norm_, W_class_norm_,],feed_dict={x_: x_test_batch, x_mask_: x_test_batch_mask,y_: test_labels, keep_prob: 1.0,class_penalty_:0.0})
                                test_accuracy, Att_h, Att_v, H_enc= sess.run([accuracy_, Att_h_, Att_v_, H_enc_],feed_dict={x_: x_test_batch, x_mask_: x_test_batch_mask,y_: test_labels, keep_prob: 1.0})



                                test_correct += test_accuracy * len(test_index)
                                # create embedding
                                test_emb += H_enc.tolist()
                                test_y += np.argmax(test_labels, axis=1).tolist()


                            test_accuracy = test_correct / len(test)

                            print("Test accuracy %f " % test_accuracy)

                            max_test_accuracy = test_accuracy

                            # print test example
                            # if test_accuracy > 0.74:
                            #     for its, itl, iah, iav in zip(test_sents, test_labels,Att_h, Att_v):
                            #         print("Label is {} {}".format(opt.class_name[np.argmax(itl)], np.argmax(itl)))
                            #         print( " ".join([ixtoword[itw] + "(" + "{0:.2f}".format(iavw) + ")" for itw, iavw in zip(its, iav[:,0])]))
                            #         print(opt.class_name)
                            #         for itw, iaw in zip(its, iah):
                            #             print(ixtoword[itw] + ":" + np.array2string(iaw, precision=2, separator=',', suppress_small=True))
                                    
                                


                            if test_accuracy > 0.98:
                                W_class = sess.run(W_class_, feed_dict={x_: x_test_batch, x_mask_: x_test_batch_mask,
                                                                    y_: test_labels, keep_prob: 1.0,
                                                                    class_penalty_:0.0})
                                # W_class = np.hstack([W_class, W_class])
                                # for i in range(10):
                                #     print("Label is {} ".format(np.argmax(test_labels[i])) + opt.class_name[np.argmax(test_labels[i])])
                                #     print("Horizontal Att:" + "&".join(opt.class_name) + '\n' + " ".join([ixtoword[x] + "(" + np.array_str(a, ) + ") \n" for x,a in zip(x_test_batch[i][:100], Att_h[i][:100] )]))
                                    # print("Vertical Att: \n" + " ".join([ixtoword[x] + "(" + str(round(a[0], 4)) + ")" for x,a in zip( x_test_batch[i][:], Att_v_max[i][:])]))

                                # save embedding for view
                                embedding_center(test_emb, test_y, W_class.tolist(), opt)
                                embedding_view(test_emb, test_y, W_class.tolist(), sess, opt)
                                # embedding_view(test_emb, test_y, np.matmul(W_class, W_class.T).tolist(), sess, opt)
                                
                                # cPickle.dump([test_emb, test_y, np.matmul(W_class, W_class.T).tolist()], open(opt.save_path+'/Att_emb.p', 'wr'))
                                # Att_emb = sess.run(Att_emb_)
                                # for i in range(len(ixtoword)):
                                #     print(ixtoword[i] + ": " + "(" + np.array_str(Att_emb[i,:], precision=2) + ") \n" )
                                # Att_dict = { ixtoword[i] : Att_emb[i,:] for i in range(len(ixtoword))}
                                # np.save(opt.save_path+'/Att_dict.npy', Att_dict)
                                
                            # get W for initialization
                            # if test_accuracy > 0.73:
                            #     W_save = sess.run(W_norm)
                            #     cPickle.dump(W_save, open('./data/yahoo_W_train.p', 'wb') )


                                # summary = sess.run(merged, feed_dict={x_: x_val_batch, _: x_val_batch_org, keep_prob: 1.0})
                        # test_writer.add_summary(summary, uidx)

                print("Epoch %d: Max Test accuracy %f" % (epoch, max_test_accuracy))

                saver.save(sess, opt.save_path, global_step=epoch)

            print("Max Test accuracy %f " % max_test_accuracy)

        except KeyboardInterrupt:
            # print 'Training interupted'
            print('Training interupted')
            print("Max Test accuracy %f " % max_test_accuracy)

# def main(argv=None):
#     learn_runner.run(experiment_fn, FLAGS.train_dir)

if __name__ == '__main__':
    main()
