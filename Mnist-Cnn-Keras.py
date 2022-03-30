import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
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
# one hot
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# --------------------------------------------------

model = Sequential()
# 畳み込み１
model.add(
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 畳み込み２
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 全層結合
model.add(Flatten())
model.add(Dense(units, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(num_classes, activation='softmax'))

# モデル構築
# 損失関数　ラベルone-hot 損失関数としてcategorical_crossentropy
# 最適化　Adam
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])

# 学習
print('------------------- train start:', datetime.datetime.today())
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    steps_per_epoch=None,  # デフォルトのNoneはデータセットのサンプル数をバッチサイズで割ったもの
    verbose=0,
    shuffle=True,
    validation_data=(x_test, y_test))
print('------------------- train end:', datetime.datetime.today())

# テスト
print('------------------- test start:', datetime.datetime.today())
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print('Accuracy:', score[1])
print('------------------- end:', datetime.datetime.today())
