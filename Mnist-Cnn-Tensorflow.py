import datetime
import tensorflow as tf
from keras.datasets import mnist
from keras import backend as K

# Hyper parameters
learning_rate = 0.001
batch_size = 100
epochs = 1
dropout = 0.25
units = 128
num_steps = 60000 / 100
img_rows, img_cols = 28, 28
num_classes = 10

# --------------------------------------------------
# 学習用に60,000個、検証用に10,000個
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 60000x28x28
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# --------------------------------------------------

# 学習用に60,000個、検証用に10,000個
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)


# モデル構築
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('Test', reuse=reuse):
        x = x_dict['images']
        x = tf.reshape(x, shape=[-1, img_rows, img_cols, 1])

        # 畳み込み１
        conv1 = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        # 畳み込み２
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        # 全層結合
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, units)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        out = tf.layers.dense(fc1, n_classes)
    return out


def model_fn(features, labels, mode):
    logits_train = conv_net(features,
                            num_classes,
                            dropout,
                            reuse=False,
                            is_training=True)
    logits_test = conv_net(features,
                           num_classes,
                           dropout,
                           reuse=True,
                           is_training=False)

    pred_classes = tf.argmax(logits_test, axis=1)
    # pred_probas = tf.nn.softmax(logits_test)

    # 検証
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # 損失関数　ラベルone-hot 損失関数としてsoftmax_cross_entropy
    loss_op = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train,
                                                       labels=tf.cast(
                                                           labels,
                                                           dtype=tf.int32)))

    # 最適化　Adam
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # 評価
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


model = tf.estimator.Estimator(model_fn)

x_train = x_train.reshape([60000, 784])
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
# print('mnist.train.images shape:', mnist.train.images.shape)
# print('mnist.train.labels shape:', mnist.train.labels.shape)

input_fn_train = tf.estimator.inputs.numpy_input_fn(x={'images': x_train},
                                                    y=y_train,
                                                    batch_size=batch_size,
                                                    num_epochs=epochs,
                                                    shuffle=True)

input_fn_test = tf.estimator.inputs.numpy_input_fn(x={'images': x_test},
                                                   y=y_test,
                                                   batch_size=batch_size,
                                                   shuffle=False)

# input_fn_train = tf.estimator.inputs.numpy_input_fn(
#     x={'images': mnist.train.images}, y=mnist.train.labels,
#     batch_size=batch_size, num_epochs=epochs, shuffle=True)

# input_fn_test = tf.estimator.inputs.numpy_input_fn(
#     x={'images': mnist.test.images}, y=mnist.test.labels,
#     batch_size=batch_size, shuffle=False)

# 学習
print('------------------- train start:', datetime.datetime.today())
# model.train(input_fn_train, steps=num_steps)
model.train(input_fn_train)
print('------------------- train end:', datetime.datetime.today())

# テスト
print('------------------- test start:', datetime.datetime.today())
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={'images': mnist.test.images}, y=mnist.test.labels,
#     batch_size=batch_size, shuffle=False)

e = model.evaluate(input_fn_test)

print("Accuracy:", e['accuracy'])
print('------------------- end:', datetime.datetime.today())
