import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

from tqdm import trange
from flearn.utils.model_utils import batch_data
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad


class Model(object):
    def __init__(self, num_classes, q, optimizer, seed=1, input_shape=(64, 64, 3)):
        # Model parameters
        self.num_classes = num_classes
        self.input_shape = input_shape

        # Create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.output2, self.train_op, self.grads, self.kl_grads, self.eval_metric_ops, self.loss, self.kl_loss, self.soft_max, self.predictions = self.create_model(q, optimizer)
            self.saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        # Memory footprint and computation cost
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, q, optimizer):
        """CNN model definition for MRI data"""
        features = tf.placeholder(tf.float32, shape=[None] + list(self.input_shape), name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        output2 = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='output2')

        # First convolutional layer
        conv1 = tf.layers.conv2d(inputs=features, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Second convolutional layer
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Third convolutional layer (added for complexity)
        conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

        # Flattening the output of the third pooling layer
        pool3_flat = tf.reshape(pool3, [-1, (self.input_shape[0] // 8) * (self.input_shape[1] // 8) * 128])

        # Fully connected layer
        dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

        # Dropout layer for regularization
        dropout = tf.layers.dropout(inputs=dense, rate=0.5)

        # Output layer with logits
        logits = tf.layers.dense(inputs=dropout, units=self.num_classes)

        # Predictions and probabilities
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        # Loss function
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        # Evaluation metrics
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))

        # KL divergence loss (optional)
        kl_loss = tf.keras.losses.KLD(predictions['probabilities'], output2) + tf.keras.losses.KLD(output2, predictions['probabilities'])
        kl_grads_and_vars = optimizer.compute_gradients(kl_loss)
        kl_grads, _ = zip(*kl_grads_and_vars)

        return features, labels, output2, train_op, grads, kl_grads, eval_metric_ops, loss, kl_loss, predictions['probabilities'], predictions['classes']

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, mini_batch_data):
        with self.graph.as_default():
            grads = self.sess.run(self.grads, feed_dict={self.features: mini_batch_data[0], self.labels: mini_batch_data[1]})
        return grads

    def get_kl_gradients(self, data, output2):
        with self.graph.as_default():
            kl_grads = self.sess.run(self.kl_grads, feed_dict={self.features: data['x'], self.labels: data['y'], self.output2: output2})
        return kl_grads

    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''Solves the local optimization problem'''
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return soln, comp

    def get_loss(self, data):
        with self.graph.as_default():
            loss = self.sess.run(self.loss, feed_dict={self.features: data['x'], self.labels: data['y']})
        return loss
    
    def solve_sgd(self, mini_batch_data):
        with self.graph.as_default():
            grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op],
                                           feed_dict={self.features: mini_batch_data[0],
                                                      self.labels: mini_batch_data[1]})

        weights = self.get_params()
        return grads, loss, weights

    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss

    def close(self):
        self.sess.close()
