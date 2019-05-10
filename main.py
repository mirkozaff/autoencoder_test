import matplotlib
matplotlib.use('TKAgg')  # Or any other X11 back-end
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from model import autoencoder_model, cnn_model
from keras.layers import Input, Dense, Flatten
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard


def main():

    batch_size = 128
    num_classes = 10
    epochs = 50

    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    # callbacks
    tb = TensorBoard(log_dir='./graphs/encoded')

    # pretrain an autoencoder
    autoencoder = autoencoder_model()    
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(x_train_noisy, x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # first model
    x = autoencoder.get_layer('max_pooling2d_3').output 
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=autoencoder.input, outputs=output)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    # saving model
    model.save('my_model.h5')

    model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, y_test),
                    callbacks=[tb])
    
    # second model
    model_2 = load_model('my_model.h5')
    
    # Freeze the layers except the last 3 layers
    for layer in model_2.layers[:-3]:
        layer.trainable = False

    model_2.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # callbacks
    tb2 = TensorBoard(log_dir='./graphs/freezed')

    model_2.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, y_test),
                    callbacks=[tb2])

    # third model
    model_3 = cnn_model(num_classes)
    model_3.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    # callbacks
    tb3 = TensorBoard(log_dir='./graphs/conv')

    model_3.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, y_test),
                    callbacks=[tb3])

    # models evaluation
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\nModel 1')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    score = model_2.evaluate(x_test, y_test, verbose=0)
    print('\nModel 2')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    score = model_3.evaluate(x_test, y_test, verbose=0)
    print('\nModel 3')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # evaluate noisy data
    score = model.evaluate(x_test_noisy, y_test, verbose=0)
    print('\nModel 1')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    score = model_2.evaluate(x_test_noisy, y_test, verbose=0)
    print('\nModel 2')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    score = model_3.evaluate(x_test_noisy, y_test, verbose=0)
    print('\nModel 3')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()