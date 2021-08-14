import tensorflow as tf
import numpy as np

def load_data(path, batch_size=50, k=0, dataset = "bio"):

	X_train_icub = tf.convert_to_tensor(np.load(path + 'icub_train_' + str(k) + '.npy'), tf.float32)
	X_train_bio = tf.convert_to_tensor(np.load(path + 'bio_train_' + str(k) + '.npy'), tf.float32)

	X_val_icub = tf.convert_to_tensor(np.load(path + 'icub_val_' + str(k) + '.npy'), tf.float32)
	X_val_bio = tf.convert_to_tensor(np.load(path + 'bio_val_' + str(k) + '.npy'), tf.float32)
	
	X_test_icub = tf.convert_to_tensor(np.load(path + 'icub_test.npy'), tf.float32)
	X_test_bio = tf.convert_to_tensor(np.load(path + 'bio_test.npy'), tf.float32)

	y_train = tf.convert_to_tensor(np.load(path + 'labels_train_' + str(k) + '.npy'), tf.float32)
	y_val = tf.convert_to_tensor(np.load(path + 'labels_val_' + str(k) + '.npy'), tf.float32)
	y_test = tf.convert_to_tensor(np.load(path + 'labels_test.npy'), tf.float32)

	X_train_bio = tf.transpose(X_train_bio, [0,2,1])
	X_val_bio = tf.transpose(X_val_bio, [0,2,1])
	X_test_bio = tf.transpose(X_test_bio, [0,2,1])

	X_train_icub = tf.transpose(X_train_icub, [0,3,1,2])[..., tf.newaxis]
	X_val_icub = tf.transpose(X_val_icub, [0,3,1,2])[..., tf.newaxis]
	X_test_icub = tf.transpose(X_test_icub, [0,3,1,2])[..., tf.newaxis]

	print("Loading Datasets - k-fold = ", k)
	print("."*80)

	print("BioTac training dataset shape: ", X_train_bio.shape)
	print("Icub training dataset shape: ", X_train_icub.shape)

	print("BioTac validation dataset shape: ", X_val_bio.shape)
	print("Icub validation dataset shape: ", X_val_icub.shape)

	print("."*80)

	bio_train_ds = tf.data.Dataset.from_tensor_slices((X_train_bio, y_train))
	bio_val_ds = tf.data.Dataset.from_tensor_slices((X_val_bio, y_val))
	bio_test_ds = tf.data.Dataset.from_tensor_slices((X_test_bio, y_test))

	icub_train_ds = tf.data.Dataset.from_tensor_slices((X_train_icub, y_train))
	icub_val_ds = tf.data.Dataset.from_tensor_slices((X_val_icub, y_val))
	icub_test_ds = tf.data.Dataset.from_tensor_slices((X_test_icub, y_test))

	def make_batches(ds, shuffle_len=600):
	  return (
	      ds
	      .cache()
	      .shuffle(shuffle_len)
	      .batch(batch_size)
	      .prefetch(tf.data.AUTOTUNE))

	tf_datasets = {

		"bio":{
			"train": make_batches(bio_train_ds),
			"val": make_batches(bio_val_ds),
			"test": make_batches(bio_test_ds),
		},

		"icub":{
			"train": make_batches(icub_train_ds),
			"val": make_batches(icub_val_ds),
			"test": make_batches(icub_test_ds),
		},

	}

	return tf_datasets