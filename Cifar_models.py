import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

def six_layer_normalization_models(normalization,batch,surround_dist=1,scale=1):
    layer_sizes = [int(x*scale) for x in [64, 128, 256]]
    model = Sequential()
    model.add(Conv2D(layer_sizes[0], 
                            (3, 3),
                            strides=(1, 1), 
                            padding='same', 
                            kernel_regularizer=keras.regularizers.l2(1e-4), 
                            input_shape=(32, 32, 3),
                            name = 'conv1'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))

    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())

    model.add(Activation('selu'))
    model.add(Dropout(0.1))

    model.add(Conv2D(layer_sizes[0], 
                            (3, 3),
                            strides=(1, 1), 
                            padding='same', 
                            kernel_regularizer=keras.regularizers.l2(1e-4), 
                            name = 'conv2'))
    
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(layer_sizes[1], 
                            (3, 3),
                            strides=(1, 1), 
                            padding='same', 
                            kernel_regularizer=keras.regularizers.l2(1e-4), 
                            name = 'conv3'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(layer_sizes[1], 
                            (3, 3),
                            strides=(1, 1), 
                            padding='same', 
                            kernel_regularizer=keras.regularizers.l2(1e-4), 
                            name = 'conv4'))
    
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(layer_sizes[2], 
                            (3, 3),
                            strides=(1, 1), 
                            padding='same', 
                            kernel_regularizer=keras.regularizers.l2(1e-4), 
                            name = 'conv5'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(layer_sizes[2], 
                            (3, 3),
                            strides=(1, 1), 
                            padding='same', 
                            kernel_regularizer=keras.regularizers.l2(1e-4), 
                            name = 'conv6'))
    
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',activity_regularizer=keras.regularizers.l1(0.015)))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    # opt = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)#SGD(learning_rate=0.01, momentum=0.9, decay=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

################################################################################################

def five_layer_normalization_models(normalization,batch, surround_dist=1,scale=1):
    layer_sizes = [int(x*scale) for x in [64, 128, 256]]
    model = Sequential()
    model.add(Conv2D(layer_sizes[0], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        input_shape=(32, 32, 3),
                        name = 'conv1'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(layer_sizes[0], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv2'))
  
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(layer_sizes[1], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv3'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(layer_sizes[1], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv4'))
  
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(layer_sizes[2], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv5'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    # model.add(Dropout(0.3))
    
    # model.add(Conv2D(layer_sizes[2], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv6'))
    
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',activity_regularizer=keras.regularizers.l1(0.015)))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    # opt = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)#SGD(learning_rate=0.01, momentum=0.9, decay=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
  
################################################################################################

def four_layer_normalization_models(normalization,batch, surround_dist=1,scale=1):
    layer_sizes = [int(x*scale) for x in [64, 128, 256]]
    model = Sequential()
    model.add(Conv2D(layer_sizes[0], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        input_shape=(32, 32, 3),
                        name = 'conv1'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(layer_sizes[0], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv2'))
  
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(layer_sizes[1], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv3'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    # model.add(Dropout(0.1))
    
    # model.add(Conv2D(layer_sizes[1], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv4'))
    
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(layer_sizes[2], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv5'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    # model.add(Dropout(0.3))
    
    # model.add(Conv2D(layer_sizes[2], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv6'))
    
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',activity_regularizer=keras.regularizers.l1(0.015)))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    # opt = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)#SGD(learning_rate=0.01, momentum=0.9, decay=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
    
################################################################################################


def three_layer_normalization_models(normalization,batch, surround_dist=1,scale=1):
    layer_sizes = [int(x*scale) for x in [64, 128, 256]]
    model = Sequential()
    model.add(Conv2D(layer_sizes[0], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        input_shape=(32, 32, 3),
                        name = 'conv1'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    #   model.add(Dropout(0.1))
    
    # model.add(Conv2D(layer_sizes[0], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv2'))
    
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
 
    model.add(Conv2D(layer_sizes[1], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv3'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    # model.add(Dropout(0.1))
    
    # model.add(Conv2D(layer_sizes[1], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv4'))
    
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(layer_sizes[2], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv5'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    # model.add(Dropout(0.3))
    
    # model.add(Conv2D(layer_sizes[2], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv6'))
    
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',activity_regularizer=keras.regularizers.l1(0.015)))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    # opt = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)#SGD(learning_rate=0.01, momentum=0.9, decay=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
  
################################################################################################


def two_layer_normalization_models(normalization,batch, surround_dist=1,scale=1):
    layer_sizes = [int(x*scale) for x in [64, 128, 256]]
    model = Sequential()
    model.add(Conv2D(layer_sizes[0], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        input_shape=(32, 32, 3),
                        name = 'conv1'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    #   model.add(Dropout(0.1))
    
    # model.add(Conv2D(layer_sizes[0], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv2'))
    
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(layer_sizes[1], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv3'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    # model.add(Dropout(0.1))
    
    # model.add(Conv2D(layer_sizes[1], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv4'))
    
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.3))
    
    # model.add(Conv2D(layer_sizes[2], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv5'))
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    # model.add(Dropout(0.3))
    
    # model.add(Conv2D(layer_sizes[2], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv6'))
    
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',activity_regularizer=keras.regularizers.l1(0.015)))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    # opt = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)#SGD(learning_rate=0.01, momentum=0.9, decay=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
    
################################################################################################

def one_layer_normalization_models(normalization,batch, surround_dist=1,scale=1):
    layer_sizes = [int(x*scale) for x in [64, 128, 256]]
    model = Sequential()
    model.add(Conv2D(layer_sizes[0], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        input_shape=(32, 32, 3),
                        name = 'conv1'))
    if batch:
        if normalization.__name__ is not 'no_norm':
            model.add(BatchNormalization(center=False, scale=False))
        else:
            model.add(BatchNormalization(center=True, scale=True))
    if normalization is not None:
        model.add(normalization())
        # model.add(normalization())
    model.add(Activation('selu'))
    #   model.add(Dropout(0.1))
    
    # model.add(Conv2D(layer_sizes[0], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv2'))
    
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.1))
    
    # model.add(Conv2D(layer_sizes[1], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv3'))
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    # model.add(Dropout(0.1))
    
    # model.add(Conv2D(layer_sizes[1], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv4'))
    
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.3))
    
    # model.add(Conv2D(layer_sizes[2], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv5'))
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    # model.add(Dropout(0.3))
    
    # model.add(Conv2D(layer_sizes[2], 
    #                       (3, 3),
    #                       strides=(1, 1), 
    #                       padding='same', 
    #                       kernel_regularizer=keras.regularizers.l2(1e-4), 
    #                       name = 'conv6'))
    
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(normalization(surround_dist=1,trainable=True))
    # model.add(Activation('selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',activity_regularizer=keras.regularizers.l1(0.015)))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    # opt = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)#SGD(learning_rate=0.01, momentum=0.9, decay=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model



def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 50:
        lrate = 0.0005
    if epoch > 75:
        lrate = 0.00025
    if epoch > 95:
        lrate = 0.00005
    if epoch > 105:
        lrate = 0.00001
    return lrate

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(x_train)




# Training code. Train for 130 epochs.
# batch = false
# surround_dist = 1
# model = six_layer_normalization_models(normalization,batch,surround_dist)
# history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
#                     steps_per_epoch=x_train.shape[0] // 32,
#                     epochs=130,
#                     verbose=1,
#                     validation_data=(x_test, y_test),
#                     callbacks=[keras.callbacks.LearningRateScheduler(lr_schedule)]
#                     )