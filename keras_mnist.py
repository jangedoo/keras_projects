import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense


# load the data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Shape of x_train  = ", x_train.shape)
print("Shape of y_train = ", y_train.shape)
print("Shape of x_test  = ", x_test.shape)
print("Shape of y_test = ", y_test.shape)

x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')

x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(784,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.3)

score = model.evaluate(x_test, y_test)
print("Test loss = ", score[0], " Test accuracy = ", score[1])