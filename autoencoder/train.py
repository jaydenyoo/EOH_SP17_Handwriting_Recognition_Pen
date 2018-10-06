import numpy as np
import tensorflow as tf
from autoencoder import autoencoder, retrieve_data

save_direct = ./mnt/c/Users/tjxor/Desktop/projects/smart_pen/model
model_name = input("Enter Model Name to Save:")

class training_data():
	def __init__(self):
		self.data = np.array([retrieve_data(letter) for letter in alphabets])
		self.training_size = self.data.shape[0]
		self.samples_count = 0

	def next_batch(self, batch_size):
		ub = samples_count + batch_size
		if ub > self.training_size:
			ub = self.training_size
		ret = self.data[self.samples_count: ub]
		self.samples_count = ub % self.training_size # Resets to 0 if epoch is done.


ae = autoencoder()
alphabets = ['A', 'B', 'C', 'D']

epochs = 20 # 1 epoch is one full 
batch_size = 100

train_data = training_data()

with tf.Session() as sess:
	ae.assign_sess(sess)
	sess.run(tf.global_variables_initializer())

	for epoch in range(epochs):
		for _ in range(training_size // batch_size):
			batch = train_data.next_batch(batch_size)
			batch_cost, o = ae.train_epoch(batch)
			print("Epoch: "+str(epoch)+" Training Loss: "+str(batch_cost))

		# Save Model after each epoch!
		tf.saved_model.simple_save(sess, save_dir, inputs = {"x": ae.inputs_, "y": ae.targets_})
