import scipy.misc
import numpy as np
from keras.datasets import mnist
from keras.layers import (Dense,
                          Flatten,
                          Input,
                          Conv2D,
                          UpSampling2D,
                          BatchNormalization,
                          Conv2DTranspose,
                          MaxPooling2D)
from keras.models import Model
from PIL import Image
from keras.utils import to_categorical


img_x, img_y = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

input_shape = (img_x, img_y, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


input_layer = Input(input_shape)

x = Conv2D(10, 5, activation='relu')(input_layer)
x = MaxPooling2D(2)(x)
x = Conv2D(20, 2, activation='relu')(x)
x = MaxPooling2D(2)(x)
encoded = x
x = UpSampling2D(2)(x)
x = Conv2DTranspose(20, 2, activation='relu')(x)
x = UpSampling2D(2)(x)
x = Conv2DTranspose(10, 5, activation='relu')(x)
x = Conv2DTranspose(1, 3, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=x)
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x_train_noisy, x_train, batch_size=32, epochs=1, validation_data=(x_test_noisy, x_test))

model.save('normal_autoencoder_noisy.h5')

test_sample = x_test_noisy[0]
test_prediction = model.predict(np.array([test_sample]))[0]

test_sample = np.array((test_sample * 255)[0], dtype=np.uint8)
img_input = Image.fromarray(test_sample)
img_input.save('input_normal_noisy.png')

test_prediction = np.array((test_prediction * 255)[0], dtype=np.uint8)
img_output = Image.fromarray(test_prediction)
img_output.save('output_normal_noisy.png')

