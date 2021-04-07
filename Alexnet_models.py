# normalization before activation (Relu)

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential

def AlexNet(normalization = None):
    #Instantiation
    regularizer = tf.keras.regularizers.l2(5e-4)

    AlexNet = Sequential()

    #1st Convolutional Layer
    AlexNet.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11,11), strides=(4,4), padding='same', kernel_regularizer=regularizer))
    if normalization is not None:
        AlexNet.add(normalization())
    else:
        AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #2nd Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same', kernel_regularizer=regularizer))
    if normalization is not None:
        AlexNet.add(normalization())
    else:
        AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #3rd Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=regularizer))
    if normalization is not None:
        AlexNet.add(normalization())
    else:
        AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    #4th Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=regularizer))
    if normalization is not None:
        AlexNet.add(normalization())
    else:
        AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    

    #5th Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=regularizer))
    if normalization is not None:
        AlexNet.add(normalization())
    else:
        AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #Passing it to a Fully Connected layer
    AlexNet.add(Flatten())
    # 1st Fully Connected Layer
    AlexNet.add(Dense(4096, kernel_regularizer=regularizer))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    AlexNet.add(Dropout(0.4))

    #2nd Fully Connected Layer
    AlexNet.add(Dense(4096, kernel_regularizer=regularizer))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #3rd Fully Connected Layer
    AlexNet.add(Dense(1000, kernel_regularizer=regularizer))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #Output Layer
    AlexNet.add(Dense(1000, kernel_regularizer=regularizer))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('softmax'))

    return AlexNet


def small_AlexNet(normalization = None):
    #Instantiation
    regularizer = tf.keras.regularizers.l2(5e-4)

    AlexNet = Sequential()

    #1st Convolutional Layer
    AlexNet.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11,11), strides=(4,4), padding='same', kernel_regularizer=regularizer))
    if normalization is not None:
        AlexNet.add(normalization())
    else:
        AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #2nd Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same', kernel_regularizer=regularizer))
    if normalization is not None:
        AlexNet.add(normalization())
    else:
        AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    # #3rd Convolutional Layer
    # AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=regularizer))
    # if normalization is not None:
    #     AlexNet.add(normalization())
    # else:
    #     AlexNet.add(BatchNormalization())
    # AlexNet.add(Activation('relu'))

    # #4th Convolutional Layer
    # AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=regularizer))
    # if normalization is not None:
    #     AlexNet.add(normalization())
    # else:
    #     AlexNet.add(BatchNormalization())
    # AlexNet.add(Activation('relu'))
    

    #5th Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=regularizer))
    if normalization is not None:
        AlexNet.add(normalization())
    else:
        AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #Passing it to a Fully Connected layer
    AlexNet.add(Flatten())
    # 1st Fully Connected Layer
    AlexNet.add(Dense(4096, kernel_regularizer=regularizer))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    AlexNet.add(Dropout(0.4))

    #2nd Fully Connected Layer
    AlexNet.add(Dense(4096, kernel_regularizer=regularizer))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #3rd Fully Connected Layer
    AlexNet.add(Dense(1000, kernel_regularizer=regularizer))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #Output Layer
    AlexNet.add(Dense(1000, kernel_regularizer=regularizer))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('softmax'))

    return AlexNet