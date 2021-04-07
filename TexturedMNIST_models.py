import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

def three_layer_normalization_models(normalization, batch=True, trainable=True):
  layer_sizes = [32, 64, 128]
  weight_decay = 1e-4
  model = Sequential()
  model.add(Conv2D(layer_sizes[0], 
                        (3, 3),
                        strides=(2, 2), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        input_shape=(226, 226, 1),
                        name = 'conv1'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 4
  model.add(Activation('selu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.3))
 
  model.add(Conv2D(layer_sizes[1], 
                        (3, 3),
                        strides=(2, 2), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv2'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 4
  model.add(Activation('selu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.3))
 
  model.add(Conv2D(layer_sizes[2], 
                        (3, 3),
                        strides=(2, 2), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv3'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 8
  model.add(Activation('selu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
 
  model.add(Flatten())
  model.add(Dense(num_classes, activation='softmax',activity_regularizer=keras.regularizers.l1(0.01)))
  opt = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)
  # opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.0001, nesterov=False)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model



def four_layer_normalization_models(normalization, batch=True, trainable=True):
  layer_sizes = [32, 64, 128]
  weight_decay = 1e-4
  model = Sequential()
  model.add(Conv2D(layer_sizes[0], 
                        (3, 3),
                        strides=(2, 2), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        input_shape=(226, 226, 1),
                        name = 'conv1'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 4
  model.add(Activation('selu'))
  model.add(Conv2D(layer_sizes[0], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        input_shape=(226, 226, 1),
                        name = 'conv2'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 4
  model.add(Activation('selu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.3))
 
  model.add(Conv2D(layer_sizes[1], 
                        (3, 3),
                        strides=(2, 2), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv3'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 4
  model.add(Activation('selu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.3))
 
  model.add(Conv2D(layer_sizes[2], 
                        (3, 3),
                        strides=(2, 2), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv4'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 8
  model.add(Activation('selu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
 
  model.add(Flatten())
  model.add(Dense(num_classes, activation='softmax',activity_regularizer=keras.regularizers.l1(0.01)))
  opt = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)
  # opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.0001, nesterov=False)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model



def five_layer_normalization_models(normalization, batch=True, trainable=True):
  layer_sizes = [32, 64, 128]
  weight_decay = 1e-4
  model = Sequential()
  model.add(Conv2D(layer_sizes[0], 
                        (3, 3),
                        strides=(2, 2), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        input_shape=(226, 226, 1),
                        name = 'conv1'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 4
  model.add(Activation('selu'))
  model.add(Conv2D(layer_sizes[0], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        input_shape=(226, 226, 1),
                        name = 'conv2'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 4
  model.add(Activation('selu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.3))
 
  model.add(Conv2D(layer_sizes[1], 
                        (3, 3),
                        strides=(2, 2), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv3'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 4
  model.add(Activation('selu'))
  model.add(Conv2D(layer_sizes[1], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv4'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 4
  model.add(Activation('selu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.3))
 
  model.add(Conv2D(layer_sizes[2], 
                        (3, 3),
                        strides=(2, 2), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv5'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 8
  model.add(Activation('selu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
 
  model.add(Flatten())
  model.add(Dense(num_classes, activation='softmax',activity_regularizer=keras.regularizers.l1(0.01)))
  opt = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)
  # opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.0001, nesterov=False)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model



def six_layer_normalization_models(normalization, batch=True, trainable=True):
  layer_sizes = [32, 64, 128]
  weight_decay = 1e-4
  model = Sequential()
  model.add(Conv2D(layer_sizes[0], 
                        (3, 3),
                        strides=(2, 2), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        input_shape=(226, 226, 1),
                        name = 'conv1'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 4
  model.add(Activation('selu'))
  model.add(Conv2D(layer_sizes[0], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        input_shape=(226, 226, 1),
                        name = 'conv2'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 4
  model.add(Activation('selu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.3))
 
  model.add(Conv2D(layer_sizes[1], 
                        (3, 3),
                        strides=(2, 2), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv3'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 4
  model.add(Activation('selu'))
  model.add(Conv2D(layer_sizes[1], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv4'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 4
  model.add(Activation('selu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.3))
 
  model.add(Conv2D(layer_sizes[2], 
                        (3, 3),
                        strides=(2, 2), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv5'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 8
  model.add(Activation('selu'))
  model.add(Conv2D(layer_sizes[2], 
                        (3, 3),
                        strides=(1, 1), 
                        padding='same', 
                        kernel_regularizer=keras.regularizers.l2(1e-4), 
                        name = 'conv6'))
  if batch:
    if normalization.__name__ is not 'no_norm':
        model.add(BatchNormalization(center=False,scale=False))
    else:
        model.add(BatchNormalization(center=True,scale=True))
  model.add(normalization()) # 8
  model.add(Activation('selu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
 
  model.add(Flatten())
  model.add(Dense(num_classes, activation='softmax',activity_regularizer=keras.regularizers.l1(0.01)))
  opt = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)
  # opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.0001, nesterov=False)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 20:
        lrate = 0.0003
    if epoch > 35:
        lrate = 0.0001
    return lrate