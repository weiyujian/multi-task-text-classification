#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import cPickle
import data_helpers
from text_cnn import TextCNN
#from text_cnn_rnn import TextCNNRNN
#from text_rnn_cnn import TextRNNCNN
#from text_rnncnn import TextRNNandCNN
#from text_rnn import TextRNN
from tensorflow.contrib import learn
import csv
import json
import pdb
# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("model_type", "cnn", "model type cnn or cnnrnn or rnn, rnncnn, rnnandcnn")
tf.flags.DEFINE_string("test_data_file", "./data/cnews.test.seg", "test data.")
tf.flags.DEFINE_integer("task_num", 3, "Task number for multi_task mission(default:2)")
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("max_document_length", 600, "Max document length(default: 600)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
tf.flags.DEFINE_boolean("topk_eval", True, "get topk result or result that higher than a score(0.5)")
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

def load_train_params(train_dir):
    sorted_label = cPickle.load(open(train_dir + '/sorted_label'))
    return sorted_label

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    train_dir = os.path.join(FLAGS.checkpoint_dir, "..", "trained_results")
    sorted_label = load_train_params(train_dir)
    x_raw, y_test = data_helpers.load_test_data(FLAGS.test_data_file, sorted_label, FLAGS.task_num)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_real_len_test = np.array(get_real_len(x_raw, FLAGS.max_document_length))
x_test = np.array(list(vocab_processor.transform(x_raw)))
print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        print("Reading model parameters from %s" % checkpoint_file)
        if FLAGS.model_type == "cnnrnn" or FLAGS.model_type == "rnncnn" or FLAGS.model_type == "rnn" or FLAGS.model_type == "rnnandcnn":
            real_len = graph.get_operation_by_name("real_len").outputs[0]
        else:
            is_training = graph.get_operation_by_name("is_training").outputs[0]
        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = []
        for i in range(FLAGS.task_num):
            input_y.append(graph.get_operation_by_name("input_y_"+str(i)).outputs[0])
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = []
        for i in range(FLAGS.task_num):
            predictions.append(graph.get_operation_by_name("output_"+str(i)+"/predictions_"+str(i)).outputs[0])
        # Generate batches for one epoch
        all_pred = [[] for i in range(FLAGS.task_num)]
        all_act = [[] for i in range(FLAGS.task_num)]
        zip_list = []
        for i in range(FLAGS.task_num):
            zip_list.append(list(zip(x_test, y_test[i], x_real_len_test)))
        batches, total_batch_num = data_helpers.multi_task_batch_iter(zip_list, FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        for i in xrange(total_batch_num):
            y_batch = []
            for j in range(FLAGS.task_num):
                tmp_batch = batches[j].next()
                x_batch, y_tmp_batch, x_real_len_batch = zip(*tmp_batch)
                y_batch.append(y_tmp_batch)
            if FLAGS.model_type == "cnn":
                feed_dict = {
                    input_x: x_batch,
                    dropout_keep_prob: 1.0,
                    is_training: False
                }
                for j in range(FLAGS.task_num):
                    feed_dict[input_y[j]] = y_batch[j]
                batch_predictions = sess.run([predictions], feed_dict)
            else:
                batch_predictions, batch_corrct = sess.run([predictions, correct_pred_num], {input_x: x_test_batch, input_y: y_test_batch, dropout_keep_prob: 1.0, real_len: x_real_len_test_batch})
            for j in range(FLAGS.task_num):
                all_pred[j] = np.concatenate([all_pred[j], batch_predictions[0][j]])
                all_act[j] = np.concatenate([all_act[j], np.argmax(y_batch[j], axis=1)])
        err_cnt = 0
        for i in range(len(x_test)):
            for j in range(FLAGS.task_num):
                if all_pred[j][i] != all_act[j][i]:
                    err_cnt += 1
                    break
        acc = 1.0 * (len(x_test) - err_cnt) / len(x_test)
# Print accuracy if y_test is defined
if y_test is not None:
    print("Total number of test examples: {}".format(len(x_test)))
    print("Accuracy: {:g}".format(acc))

