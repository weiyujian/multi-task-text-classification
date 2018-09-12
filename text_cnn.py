import tensorflow as tf
import numpy as np
from util import highway
import pdb
class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size, task_num,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = [tf.placeholder(tf.float32, [None, num_classes[i]], name="input_y_"+str(i)) for i in range(task_num)]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.use_region_emb = True
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        num_filters_total = num_filters * len(filter_sizes)
        fc_hidden_size = num_filters_total 
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            if self.use_region_emb:
                self.region_size = 5
                self.region_radius = self.region_size / 2
                self.k_matrix_embedding = tf.Variable(tf.random_uniform([vocab_size, self.region_size, embedding_size], -1.0, 1.0), name="k_matrix")
                self.embedded_chars = self.region_embedding(self.input_x)
                sequence_length = int(self.embedded_chars.shape[1])
            else:
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                conv_bn = tf.layers.batch_normalization(conv, training=self.is_training)
                # Apply nonlinearity
                h = tf.nn.relu(conv_bn, name="relu")
                # Maxpooling over the outputs
                pool_size = sequence_length - filter_size + 1
                pooled = self._max_pooling(h, pool_size)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        # Fully Connected Layer
        with tf.name_scope("fc"):
            W_fc = tf.Variable(tf.truncated_normal(shape=[num_filters_total, fc_hidden_size],\
                stddev=0.1, dtype=tf.float32), name="W_fc")
            self.fc = tf.matmul(self.h_pool_flat, W_fc)
            self.fc_bn = tf.layers.batch_normalization(self.fc, training=self.is_training)
            self.fc_out = tf.nn.relu(self.fc_bn, name="relu")
        # Highway Layer
        self.highway = highway(self.fc_out, self.fc_out.get_shape()[1], num_layers=1, bias=-0.5, scope="Highway")

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.highway, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        self.scores = []
        self.predictions = []
        for i in range(task_num):
            with tf.name_scope("output_"+str(i)):
                W_out = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, num_classes[i]],\
                    stddev=0.1, dtype=tf.float32), name="W_out_"+str(i))
                b_out = tf.Variable(tf.constant(0.1, shape=[num_classes[i]]), name="b_out_"+str(i))
                l2_loss += tf.nn.l2_loss(W_out)
                l2_loss += tf.nn.l2_loss(b_out)
                tmp_score = tf.nn.xw_plus_b(self.h_drop, W_out, b_out, name="scores_"+str(i))
                self.scores.append(tmp_score)
                self.predictions.append(tf.argmax(tmp_score, 1, name="predictions_"+str(i)))

        # Calculate mean cross-entropy loss
        self.loss = 0.0
        for i in range(task_num):
            with tf.name_scope("loss_"+str(i)):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores[i], labels=self.input_y[i])
                self.loss += tf.reduce_mean(losses)
        self.loss += l2_reg_lambda * l2_loss

        # Accuracy
        self.accuracy = []
        for i in range(task_num):
            with tf.name_scope("accuracy_"+str(i)):
                correct_predictions = tf.equal(self.predictions[i], tf.argmax(self.input_y[i], 1))
                self.accuracy.append(tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy_"+str(i)))


    def _max_pooling(self, inputs, filter_size):
        # max pooling
        pooled = tf.nn.max_pool(
            inputs,
            ksize=[1, filter_size, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        return pooled
    
    def _k_max_pooling(self, inputs, top_k):
        # k max pooling
        #inputs : batch_size, sequence_length, hidden_size, chanel_size]
        inputs = tf.transpose(inputs, [0,3,2,1]) # [batch_size, chanel_size, hidden_size, sequence_length]
        k_pooled = tf.nn.top_k(inputs, k=top_k, sorted=True, name='top_k')[0] # [batch_size, chanel_size, hidden_size, top_k]
        k_pooled = tf.transpose(k_pooled, [0,3,2,1]) #[batch_size, top_k, hidden_size, chanel_size]
        return k_pooled

    def _chunk_max_pooling(self, inputs, chunk_size):
        #chunk max pooling
        seq_len = inputs.get_shape()[1].values
        inputs_ = tf.split(inputs, chunk_size, axis=1) # seq_len/chunk_size list,element is  [batch_size, seq_len/chunk_size, hidden_size, chanel_size]
        chunk_pooled_list = []
        for i in range(len(inputs_)):
            chunk_ = inputs_[i]
            chunk_pool_ = self._max_pooling(chunk_, seq_len/chunk_size)
            chunk_pooled_list.append(chunk_pool_)
        chunk_pooled = tf.concat(chunk_pooled_list, axis=1)
        return chunk_pooled
    
    def get_seq(self, inputs):
        neighbor_seq = map(lambda i: tf.slice(inputs, [0, i-self.region_radius], [-1, self.region_size]), xrange(self.region_radius, inputs.shape[1] - self.region_radius))
        neighbor_seq = tf.convert_to_tensor(neighbor_seq)
        neighbor_seq = tf.transpose(neighbor_seq, [1,0,2])
        return neighbor_seq
    
    def get_seq_without_loss(self, inputs):
        neighbor_seq = map(lambda i: tf.slice(inputs, [0, i-self.region_radius], [-1, self.region_size]), xrange(self.region_radius, inputs.shape[1] - self.region_radius))
        neighbor_begin = map(lambda i: tf.slice(inputs, [0, 0], [-1, self.region_size]), xrange(0, self.region_radius))
        neighbor_end = map(lambda i: tf.slice(inputs, [0, inputs.shape[1] - self.region_size], [-1, self.region_size]), xrange(0, self.region_radius))
        neighbor_seq = tf.concat([neighbor_begin, neighbor_seq, neighbor_end], 0)
        neighbor_seq = tf.convert_to_tensor(neighbor_seq)
        neighbor_seq = tf.transpose(neighbor_seq, [1,0,2])
        return neighbor_seq

    def region_embedding(self, inputs):
        region_k_seq = self.get_seq(inputs)
        region_k_emb = tf.nn.embedding_lookup(self.W, region_k_seq)
        trimed_inputs = inputs[:, self.region_radius: inputs.get_shape()[1] - self.region_radius]
        context_unit = tf.nn.embedding_lookup(self.k_matrix_embedding, trimed_inputs)
        projected_emb = region_k_emb * context_unit
        embedded_chars = tf.reduce_max(projected_emb, axis=2)
        return embedded_chars
