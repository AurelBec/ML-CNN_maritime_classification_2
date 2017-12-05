import os
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


"""global definitions"""
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.logging.set_verbosity(tf.logging.INFO)
ROOT_PATH = "/home/aurelien/workspace/python/cnn/"
train_data_directory = os.path.join(ROOT_PATH, "datasets/training_set")
test_data_directory = os.path.join(ROOT_PATH, "datasets/test_set")
__id2class__ = {0:'Alilaguna', 1:'Ambulanza', 2:'Barchino', 3:'Cacciapesca', 4:'Caorlina', 5:'Gondola', 6:'Lanciafino10m', 7:'Lanciafino10mBianca', 8:'Lanciafino10mMarrone', 9:'Lanciamaggioredi10mBianca', 10:'Lanciamaggioredi10mMarrone', 11:'Motobarca', 12:'Motopontonerettangolare', 13:'MotoscafoACTV', 14:'Mototopo', 15:'Patanella', 16:'Polizia', 17:'Raccoltarifiuti', 18:'Sandoloaremi', 19:'Sanpierota', 20:'Topa', 21:'VaporettoACTV', 22:'VigilidelFuoco', 23:'Water'}
__class2id__ = {'Alilaguna':0, 'Ambulanza':1, 'Barchino':2, 'Cacciapesca':3, 'Caorlina':4, 'Gondola':5, 'Lanciafino10m':6, 'Lanciafino10mBianca':7, 'Lanciafino10mMarrone':8, 'Lanciamaggioredi10mBianca':9, 'Lanciamaggioredi10mMarrone':10, 'Motobarca':11, 'Motopontonerettangolare':12, 'MotoscafoACTV':13, 'Mototopo':14, 'Patanella':15, 'Polizia':16, 'Raccoltarifiuti':17, 'Sandoloaremi':18, 'Sanpierota':19, 'Topa':20, 'VaporettoACTV':21, 'VigilidelFuoco':22, 'Water':23}


"""hyperparameters"""
resizeX = 60
resizeY = 200
colorchannel = 1
nbSteps = 500
nbFilter1 = 32
sizeFilter1 = 3
nbFilter2 = 64
sizeFilter2 = 3
nbClasses = 24


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".jpg")]

        for f in file_names:
            images.append(data.load(f))
            labels.append(__class2id__[d])

    print("{0} images loaded in {1} classes".format(len(images), len(set(labels))))
    return images, labels



def reshape_data(data, _resizeX = resizeX, _resizeY = resizeY, _colorchannel = colorchannel):
    data = [transform.resize(d, (_resizeX, _resizeY)) for d in data]
    data = np.array(data)
    if _colorchannel == 1:
        data = rgb2gray(data)

    print("{0} images resized to {1}x{2}".format(len(data), _resizeX, _resizeY))
    return np.array([d.ravel() for d in data])


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Images are XxY pixels, and have CC color channel
    input_layer = tf.reshape(features["x"], [-1, resizeX, resizeY, colorchannel])

    # Convolutional Layer #1
    # Computes NF1 features using a SF1xSF1 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, X, Y, CC]
    # Output Tensor Shape: [batch_size, X, Y, CC*NF1]
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=nbFilter1,
            kernel_size=[sizeFilter1, sizeFilter1],
            padding="same",
            activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, X, Y, CC*NF1]
    # Output Tensor Shape: [batch_size, X/2, Y/2, CC*NF1]
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2)

    # Convolutional Layer #2
    # Computes NF2 features using a SF2xSF2 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, X/2, Y/2, CC*NF1]
    # Output Tensor Shape: [batch_size, X/2, Y/2, CC*NF2]
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=nbFilter2,
            kernel_size=[sizeFilter2, sizeFilter2],
            padding="same",
            activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, X/2, Y/2, CC*NF2]
    # Output Tensor Shape: [batch_size, X/4, Y/4, CC*NF2]
    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, X/4, Y/4, CC*NF2]
    # Output Tensor Shape: [batch_size, X/4 * Y/4 * CC*NF2]
    pool2_flat = tf.reshape(pool2, [-1, int(resizeX/4 * resizeY/4 * colorchannel * nbFilter2)])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 15 * 50 * 192]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(
            inputs=pool2_flat,
            units=1024,
            activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
              inputs=dense,
              rate=0.4,
              training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, NC]
    logits = tf.layers.dense(
             inputs=dropout,
             units=nbClasses)

    predictions = {
                  # Generate predictions (for PREDICT and EVAL mode)
                  "classes": tf.argmax(input=logits, axis=1),
                  # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                  # `logging_hook`.
                  "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
               mode=mode,
               predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(
                    indices=tf.cast(labels,tf.int32),
                    depth=nbClasses)
    loss = tf.losses.softmax_cross_entropy(
           onehot_labels=onehot_labels,
           logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                   loss=loss,
                   global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
               mode=mode,
               loss=loss,
               train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,
                                        predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    """image loading"""
    images, labels = load_data(train_data_directory)
    images = reshape_data(images)
    images = np.array(images, dtype = 'f')
    labels = np.array(labels)
    assert images.shape[0] == labels.shape[0]


    """creating dataset"""
    # dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    """nice info display"""
    # Make a histogram with 24 bins of the `labels` data
    # plt.hist(labels, len(set(labels)))
    # plt.show()

    """create the Estimator"""
    maritime_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="test")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    """train the model"""
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                     x={"x": images},
                     y=labels,
                     batch_size=100,
                     num_epochs=None,
                     shuffle=True)

    maritime_classifier.train(
                        input_fn=train_input_fn,
                        hooks=[logging_hook],
                        steps= nbSteps)

    """evaluate the model and print results"""
    test_images, test_labels = load_data(test_data_directory)
    test_images = np.array(reshape_data(test_images), dtype = 'f')
    test_labels = np.array(test_labels)
    assert test_images.shape[0] == test_labels.shape[0]
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": test_images},
                    y=test_labels,
                    num_epochs=1,
                    shuffle=False)
    eval_results = maritime_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)



if __name__ == "__main__":
    tf.app.run()

#
#with tf.Session():
#    print('Confusion Matrix: \n\n', tf.Tensor.eval(tf.contrib.metrics.confusion_matrix(np.array(range(1, nbClasses)), np.array(range(1,24))),feed_dict=None, session=None))
