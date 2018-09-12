#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import cPickle
import data_helpers
from text_cnn import TextCNN
#from text_rnn import TextRNN
#from text_cnn_rnn import TextCNNRNN
#from text_rnn_cnn import TextRNNCNN
#from text_rnncnn import TextRNNandCNN
from tensorflow.contrib import learn
import sys
import json
import pdb

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("model_type", "cnn", "model type cnn or cnnrnn , rnncnn, rnnandcnn")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_file", "./data/cnews.train.seg", "train data for Chinese.")
tf.flags.DEFINE_string("test_data_file", "./data/cnew.test.seg", "test data for Chinese.")
tf.flags.DEFINE_integer("max_document_length", 600, "Max document length(default: 600)")
tf.flags.DEFINE_string("model_version", "", "model version")
tf.flags.DEFINE_integer("task_num", 3, "Task number for multi_task mission(default:2)")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 300, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("hidden_unit", 200, "Rnn hidden size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 25, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def get_real_len(x_text, max_len):
    real_len = []
    for item in x_text:
        tmp_list = item.split(" ")
        seq_len = len(tmp_list)
        if seq_len > max_len:
            seq_len = max_len
        real_len.append(seq_len)
    return real_len

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y, sorted_label = data_helpers.load_multi_task_data(FLAGS.train_data_file, FLAGS.task_num)

    # Build vocabulary
    #max_document_length = max([len(x.split(" ")) for x in x_text])
    max_document_length = FLAGS.max_document_length
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x_real_len = np.array(get_real_len(x_text, max_document_length))
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    assert len(x_real_len) == len(x)
    print("max document len:",max_document_length)
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(x)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = [tmp_y[shuffle_indices] for tmp_y in y]
    x_real_len_shuffled = x_real_len[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(x)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = [tmp_y[:dev_sample_index] for tmp_y in y_shuffled], [tmp_y[dev_sample_index:] for tmp_y in y_shuffled]
    x_real_len_train, x_real_len_dev = x_real_len_shuffled[:dev_sample_index], x_real_len_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled, x_real_len, x_real_len_shuffled
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(x_train), len(x_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev, x_real_len_train, x_real_len_dev, sorted_label

def train(x_train, y_train, vocab_processor, x_dev, y_dev, x_real_len_train, x_real_len_dev, sorted_label):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            if FLAGS.model_type == "cnnrnn":
                obj = TextCNNRNN(
                    sequence_length=FLAGS.max_document_length,
                    num_classes=[tmp_y.shape[1] for tmp_y in y_train],
                    vocab_size=len(vocab_processor.vocabulary_),
                    hidden_unit=FLAGS.hidden_unit, 
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
            elif FLAGS.model_type == "rnncnn":
                obj = TextRNNCNN(
                    sequence_length=FLAGS.max_document_length,
                    num_classes=[tmp_y.shape[1] for tmp_y in y_train],
                    vocab_size=len(vocab_processor.vocabulary_),
                    hidden_unit=FLAGS.hidden_unit, 
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
            elif FLAGS.model_type == "rnnandcnn":
                obj = TextRNNandCNN(
                    sequence_length=FLAGS.max_document_length,
                    num_classes=[tmp_y.shape[1] for tmp_y in y_train],
                    vocab_size=len(vocab_processor.vocabulary_),
                    hidden_unit=FLAGS.hidden_unit, 
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
            elif FLAGS.model_type == "rnn":
                obj = TextRNN(
                    sequence_length=FLAGS.max_document_length,
                    num_classes=[tmp_y.shape[1] for tmp_y in y_train],
                    vocab_size=len(vocab_processor.vocabulary_),
                    hidden_unit=FLAGS.hidden_unit, 
                    embedding_size=FLAGS.embedding_dim,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
            else:
                obj = TextCNN(
                    sequence_length=FLAGS.max_document_length,
                    num_classes=[tmp_y.shape[1] for tmp_y in y_train],
                    vocab_size=len(vocab_processor.vocabulary_),
                    task_num=FLAGS.task_num,
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads_and_vars = optimizer.compute_gradients(obj.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.model_version))
            print("Writing to {}\n".format(out_dir))
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            
            # Save train params since eval.py needs them
            trained_dir = os.path.abspath(os.path.join(out_dir, "trained_results"))
            if not os.path.exists(trained_dir):
                os.makedirs(trained_dir)
            
            cPickle.dump(sorted_label, open(trained_dir+"/sorted_label", "w"))

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch, x_real_len_batch):
                """
                A single training step
                """
                if FLAGS.model_type == "cnn":
                    feed_dict = {
                        obj.input_x: x_batch,
                        obj.dropout_keep_prob: FLAGS.dropout_keep_prob,
                        obj.is_training: True
                    }
                else:
                    feed_dict = {
                        obj.input_x: x_batch,
                        obj.dropout_keep_prob: FLAGS.dropout_keep_prob,
                        obj.real_len: x_real_len_batch
                    }
                for i in range(FLAGS.task_num):
                    feed_dict[obj.input_y[i]] = y_batch[i]
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, obj.loss, obj.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                for i in range(FLAGS.task_num):
                    print("{}: step {}, loss {:g}, acc_{:d} {:g}".format(time_str, step, loss, i, accuracy[i]))
            
            def overfit(dev_loss,eva_num=3):
                n = len(dev_loss)
                if n < eva_num:
                    return False
                for i in xrange(n-eva_num+1, n):
                    if dev_loss[i] > dev_loss[i-1]:
                        return False
                return True

            def dev_step(x_test, y_test, x_real_len_test):
                """
                Evaluates model on a dev set
                """
                all_pred = [[] for i in range(FLAGS.task_num)]
                all_act = [[] for i in range(FLAGS.task_num)]
                zip_list = []
                for i in range(FLAGS.task_num):
                    zip_list.append(list(zip(x_test, y_test[i], x_real_len_test)))
                batches, total_batch_num = data_helpers.multi_task_batch_iter(zip_list, FLAGS.batch_size, 1, shuffle=False)
                for i in xrange(total_batch_num):
                    y_batch = []
                    for j in range(FLAGS.task_num):
                        tmp_batch = batches[j].next()
                        x_batch, y_tmp_batch, x_real_len_batch = zip(*tmp_batch)
                        y_batch.append(y_tmp_batch)
                    if FLAGS.model_type == "cnn":
                        feed_dict = {
                            obj.input_x: x_batch,
                            obj.dropout_keep_prob: 1.0,
                            obj.is_training: False
                        }
                    else:
                        feed_dict = {
                            obj.input_x: x_batch,
                            obj.dropout_keep_prob: 1.0,
                            obj.real_len: x_real_len_batch
                        }
                    for j in range(FLAGS.task_num):
                        feed_dict[obj.input_y[j]] = y_batch[j]
                    step, pred = sess.run(
                        [global_step, obj.predictions], feed_dict)
                    for j in range(FLAGS.task_num):
                        all_pred[j] = np.concatenate([all_pred[j], pred[j]])
                        all_act[j] = np.concatenate([all_act[j], np.argmax(y_batch[j], axis=1)])
                err_cnt = 0
                for i in range(len(x_test)):
                    for j in range(FLAGS.task_num):
                        if all_pred[j][i] != all_act[j][i]:
                            err_cnt += 1
                            break

                dev_acc = 1.0 * (len(x_test) - err_cnt) / len(x_test)
                print("dev_sample {}, dev_acc {:g}".format(len(x_test), dev_acc))
                return dev_acc

            # Generate batches
            zip_list = []
            for i in range(FLAGS.task_num):
                zip_list.append(list(zip(x_train, y_train[i], x_real_len_train)))
            batches, total_batch_num = data_helpers.multi_task_batch_iter(zip_list, FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            dev_acc = []
            for i in xrange(total_batch_num):
                y_batch = []
                for j in range(FLAGS.task_num):
                    tmp_batch = batches[j].next()
                    x_batch, tmp_y_batch, x_real_len_batch = zip(*tmp_batch)
                    y_batch.append(tmp_y_batch)
                train_step(x_batch, y_batch, x_real_len_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:", current_step)
                    cur_acc = dev_step(x_dev, y_dev, x_real_len_dev)
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    dev_acc.append(cur_acc)
                    if overfit(dev_acc):
                        print("current accuracy drop and stop train..\n")
                        sys.exit(0)
                    print("")

def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev, x_real_len_train, x_real_len_dev, sorted_label = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev, x_real_len_train, x_real_len_dev, sorted_label)

if __name__ == '__main__':
    tf.app.run()
