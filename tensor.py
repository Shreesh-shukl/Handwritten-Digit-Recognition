import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers
tf.random.set_seed(42)

(x_train,y_train),(x_test,y_test)= keras.datasets.mnist.load_data()
x_train.shape
x_test.shape
y_test.shape

y_train=keras.utils.to_categorical(y_train,num_classes=10)
y_test=keras.utils.to_categorical(y_test,num_classes=10)

y_test.shape

model=Sequential()
model.add(layers.Reshape((784,), input_shape=(28,28,)))
model.add(layers.BatchNormalization())

model.add(layers.Dense(200,activation="relu"))
model.add(layers.Dense(100,activation="relu"))
model.add(layers.Dense(60,activation="relu"))
model.add(layers.Dense(30,activation="relu"))
model.add(layers.Dense(20,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=24, validation_data=(x_test, y_test))

import numpy as np
np.expand_dims(x_test[0],axis=0).shape

prediction= model.predict(x_test[0:9])

predicted_num= np.argmax(prediction[0])

print(predicted_num)

import matplotlib.pyplot as plt
plt.imshow(x_test[0],cmap='gray')