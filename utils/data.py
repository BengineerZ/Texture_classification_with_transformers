import tensorflow as tf
import numpy as np

def load_data(path, batch_size=50, k=0):
	X_train_icub = tf.convert_to_tensor(np.load(path + 'icub_train_' + str(k) + '.npy'), tf.float32)
	X_val_icub = tf.convert_to_tensor(np.load(path + 'icub_val_' + str(k) + '.npy'), tf.float32)
	X_train_bio = tf.convert_to_tensor(np.load(path + 'bio_train_' + str(k) + '.npy'), tf.float32)
	X_val_bio = tf.convert_to_tensor(np.load(path + 'bio_val_' + str(k) + '.npy'), tf.float32)
	y_train = tf.convert_to_tensor(np.load(path + 'labels_train_' + str(k) + '.npy'), tf.float32)
	y_val = tf.convert_to_tensor(np.load(path + 'labels_val_' + str(k) + '.npy'), tf.float32)

	X_test_icub = tf.convert_to_tensor(np.load(path + 'icub_test.npy'), tf.float32)
	X_test_bio = tf.convert_to_tensor(np.load(path + 'bio_test.npy'), tf.float32)
	y_test = tf.convert_to_tensor(np.load(path + 'labels_test.npy'), tf.float32)

	X_train_bio = tf.transpose(X_train_bio, [0,2,1])
	X_val_bio = tf.transpose(X_val_bio, [0,2,1])
	X_test_bio = tf.transpose(X_test_bio, [0,2,1])

	print(X_train_bio.shape)
	print(X_val_bio.shape)
	print(y_train.shape)
	print(y_val.shape)
	print(X_test_bio.shape)
	print(y_test.shape)



	### Only generate the biotac dataset: TODO - implement ICub dataset

	train_ds = tf.data.Dataset.from_tensor_slices((X_train_bio, y_train))
	val_ds = tf.data.Dataset.from_tensor_slices((X_val_bio, y_val))
	test_ds = tf.data.Dataset.from_tensor_slices((X_test_bio, y_test))

	def make_batches(ds, shuffle_len=600):
	  return (
	      ds
	      .cache()
	      .shuffle(shuffle_len)
	      .batch(batch_size)
	      .prefetch(tf.data.AUTOTUNE))
	train_batches = make_batches(train_ds)
	val_batches = make_batches(val_ds)
	test_batches = make_batches(test_ds)

	return train_batches, val_batches, test_batches