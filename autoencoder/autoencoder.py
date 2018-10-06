import numpy as np
import tensorflow as tf


def retrieve_data(letter):
	'''
	Return all sensor-values of alphabet <letter>.
	The sensor values should be normalized to be a float32 normalized to [-1, 1]
	'''
	pass

class autoencoder():

	def __init__(self)
		self.learning_rate = 0.001
		# Input and target placeholders
		self.inputs_ = tf.placeholder(tf.float32, (None, None, 6, 1), name="input")
		self.targets_ = tf.placeholder(tf.float32, (None, None, 6,1), name="target")

		### Encoder
		self.conv1 = tf.layers.conv2d(inputs = self.inputs_, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
		# Now 28x28x16
		self.maxpool1 = tf.layers.max_pooling2d(self.conv1, pool_size=(2,2), strides=(2,2), padding='same')
		# Now 14x14x16
		self.conv2 = tf.layers.conv2d(inputs = self.maxpool1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
		# Now 14x14x8
		self.maxpool2 = tf.layers.max_pooling2d(self.conv2, pool_size=(2,2), strides=(2,2), padding='same')
		# Now 7x7x8
		self.conv3 = tf.layers.conv2d(inputs = self.maxpool2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
		# Now 7x7x8
		self.encoded = tf.layers.max_pooling2d(self.conv3, pool_size=(2,2), strides=(2,2), padding='same')
		# Now 4x4x8

		### Decoder
		self.upsample1 = tf.image.resize_images(self.encoded, size=(7,7), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		# Now 7x7x8
		self.conv4 = tf.layers.conv2d(inputs = self.upsample1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
		# Now 7x7x8
		self.upsample2 = tf.image.resize_images(self.conv4, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		# Now 14x14x8
		self.conv5 = tf.layers.conv2d(inputs = self.upsample2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
		# Now 14x14x8
		self.upsample3 = tf.image.resize_images(self.conv5, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		# Now 28x28x8
		self.conv6 = tf.layers.conv2d(inputs = self.upsample3, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
		# Now 28x28x16

		self.logits = tf.layers.conv2d(inputs = self.conv6, filters = 1, kernel_size=(3,3), padding='same', activation = None)
		#Now 28x28x1

		# Pass logits through sigmoid to get reconstructed image
		self.decoded = tf.nn.sigmoid(self.logits)

		# Pass logits through sigmoid and calculate the cross-entropy loss
		self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.targets_, logits = self.logits)

		# Get cost and define the optimizer
		self.cost = tf.reduce_mean(self.loss)
		self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

		self.sess = None

	def assign_sess(self, sess):
		self.sess = sess

	def train_epoch(self, batch):
		cost, opt = self.sess.run([self.cost, self.opt], feed_dict = {self.inputs_: batch, self.targets_: batch})
		return cost, opt





