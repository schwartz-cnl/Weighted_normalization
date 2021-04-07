import numpy as np
import tensorflow as tf
from tensorflow import keras

# Weighted Normalization - center and surround normalization pool
class WN_s(tf.keras.layers.Layer):

    def get_config(self):
        config = {
            'surround_dist': self.surround_dist,
            'beta_min': self._beta_min,
            'beta_init': self._beta_init,
            'gamma_init': self._gamma_init,
            'data_format': self.data_format,
        }
        base_config = super(WN_s, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def __init__(self,
               surround_dist=1,
               beta_min=1e-6,
               beta_init = 1e-6,
               gamma_init=1/16,
               data_format='channels_last',
               name=None,
               trainable=True,
               **kwargs):
        super(WN_s, self).__init__(
            trainable=trainable,
            name=name)#,
            #**kwargs)
        self.surround_dist = surround_dist
        self._beta_min = beta_min
        self._beta_init = beta_init
        self._gamma_init = gamma_init
        self.data_format = data_format
        self._channel_axis()  # trigger ValueError early

    def _channel_axis(self):
        try:
            return {'channels_first': 1, 'channels_last': -1}[self.data_format]
        except KeyError:
            raise ValueError('Unsupported `data_format` for WN layer: {}.'.format(
            self.data_format))
    
    def build(self, input_shape):
        channel_axis = self._channel_axis()
        self.norm_groups = int(input_shape[channel_axis]/8)
        super(WN_s, self).build(input_shape)
        num_channels = input_shape[channel_axis]
        if num_channels is None:
            raise ValueError('The channel dimension of the inputs to `WN` '
                           'must be defined.')

        if input_shape[channel_axis] % self.norm_groups != 0:
            raise ValueError('The number of channels must be a multiple of '
                           'the normalization_groups.')
        
        self._input_rank = input_shape.ndims

        def beta_outside_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return tf.zeros(shape, dtype='float32')

        def gamma_outside_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return tf.ones(shape, dtype='float32')

        def beta_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return self._beta_init * tf.ones(shape, dtype='float32')

        def gamma_k_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            one_tensor = tf.ones(shape, dtype='float32')
            return self._gamma_init * one_tensor

        def gamma_s_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            one_tensor = tf.ones(shape, dtype='float32')
            return self._gamma_init * one_tensor

        gamma_k_shape = [1,] * 2  +  [num_channels//self.norm_groups, num_channels]
        gamma_s_shape = [3,] * 2  +  [num_channels, 1]
        beta_shape = [num_channels]
        beta_outside_shape = [num_channels]
        gamma_outside_shape = [num_channels]

        self.beta = self.add_weight(
            name='beta',
            shape=beta_shape,
            initializer=beta_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=True)

        self.gamma_k = self.add_weight(
            name='gamma_k',
            shape=gamma_k_shape,
            initializer=gamma_k_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=True)

        self.gamma_s = self.add_weight(
            name='gamma_s',
            shape=gamma_s_shape,
            initializer=gamma_s_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=True)
        
        self.beta_outside = self.add_weight(
            name='beta_outside',
            shape=beta_outside_shape,
            initializer=beta_outside_initializer,
            dtype=self.dtype,
            trainable=True)
        
        self.gamma_outside = self.add_weight(
            name='gamma_outside',
            shape=gamma_outside_shape,
            initializer=gamma_outside_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=True)
        
        center_zero_tensor = tf.constant([[1.,1.,1.],[1.,0,1.],[1.,1.,1.]])  # center zero restriction. set to reparam_offset will lead to 0.
        center_zero_tensor = tf.stack([center_zero_tensor] * num_channels, axis=2)  # build a [3, 3, #c] tensor
        self.center_zero_tensor = tf.expand_dims(center_zero_tensor, axis = -1) # build a [3, 3, #c, 1] tensor

        self.built = True

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        ndim = self._input_rank
        shape = self.gamma_k.get_shape().as_list()
        dilation_rate = [self.surround_dist]*(ndim-2)
        squared_inputs = tf.math.square(inputs)
        squared_input_groups =  tf.split(squared_inputs, self.norm_groups, -1)
        gamma_k_groups = tf.split(self.gamma_k, self.norm_groups, -1)

        # Compute normalization pool.

        # Pk for center group
        convolve_k = lambda inputs_i, gamma_k: tf.nn.convolution(inputs_i,
                                                                  gamma_k,
                                                                  strides=(1, 1),
                                                                  padding='SAME')
            
        Pk_groups= [convolve_k(i, k) for i,k in zip(squared_input_groups, gamma_k_groups)]
        Pk = tf.concat(Pk_groups, axis=3)
        
        gamma_s = tf.math.multiply(self.gamma_s, self.center_zero_tensor)
        Ps = tf.nn.depthwise_conv2d(squared_inputs,
                                    gamma_s,
                                    strides=[1,1,1,1],
                                    padding='SAME',
                                    dilations=dilation_rate)
        beta = self.beta + self._beta_min
        norm_pool_ks = tf.nn.bias_add(tf.math.add(Pk, Ps), beta, data_format='N'+'DHW' [-(ndim - 2):]+'C') # NHWC
        norm_pool_ks = tf.math.sqrt(norm_pool_ks)

        norm_pool = tf.math.reciprocal(norm_pool_ks)
        outputs = tf.multiply(inputs, norm_pool)
        outputs = outputs * self.gamma_outside + self.beta_outside
        outputs.set_shape(inputs.get_shape())

        return outputs

    def compute_output_shape(self, input_shape):
        channel_axis = self._channel_axis()
        input_shape = tensor_shape.TensorShape(input_shape)
        #if not 3 <= input_shape.ndim <= 5:
        #  raise ValueError('`input_shape` must be of rank 3 to 5, inclusive.')
        if input_shape[channel_axis].value is None:
            raise ValueError(
              'The channel dimension of `input_shape` must be defined.')
        return input_shape


# Weighted Nomalization - center only pool
class WN_c(tf.keras.layers.Layer):

    def get_config(self):
        config = {
            'surround_dist': self.surround_dist,
            'beta_min': self._beta_min,
            'beta_init': self._beta_init,
            'gamma_init': self._gamma_init,
            'data_format': self.data_format,
        }
        base_config = super(WN_c, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def __init__(self,
               surround_dist=1,
               beta_min=1e-6,
               beta_init = 1e-6,
               gamma_init=1/8,
               data_format='channels_last',
               name=None,
               trainable=True,
               **kwargs):
        super(WN_c, self).__init__(
            trainable=trainable,
            name=name)#,
            #**kwargs)
    # TODO: check group_n is consistent with the  number of channels
        self.surround_dist = surround_dist
        self._beta_min = beta_min
        self._beta_init = beta_init
        self._gamma_init = gamma_init
        self.data_format = data_format
        self._channel_axis()  # trigger ValueError early

    def _channel_axis(self):
        try:
            return {'channels_first': 1, 'channels_last': -1}[self.data_format]
        except KeyError:
            raise ValueError('Unsupported `data_format` for WN layer: {}.'.format(
            self.data_format))
    
    def build(self, input_shape):
        channel_axis = self._channel_axis()
        self.norm_groups = int(input_shape[channel_axis]/8)
        super(WN_c, self).build(input_shape)
        num_channels = input_shape[channel_axis]
        if num_channels is None:
            raise ValueError('The channel dimension of the inputs to `WN` '
                           'must be defined.')

        if input_shape[channel_axis] % self.norm_groups != 0:
            raise ValueError('The number of channels must be a multiple of '
                           'the normalization_groups.')
        
        self._input_rank = input_shape.ndims

        def beta_outside_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return tf.zeros(shape, dtype='float32')

        def gamma_outside_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return tf.ones(shape, dtype='float32')

        def beta_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return self._beta_init * tf.ones(shape, dtype='float32')

        def gamma_k_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            one_tensor = tf.ones(shape, dtype='float32')
            return self._gamma_init * one_tensor

        gamma_k_shape = [1,] * 2  +  [num_channels//self.norm_groups, num_channels]
        beta_shape = [num_channels]
        beta_outside_shape = [num_channels]
        gamma_outside_shape = [num_channels]

        self.beta = self.add_weight(
            name='beta',
            shape=beta_shape,
            initializer=beta_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=True)

        self.gamma_k = self.add_weight(
            name='gamma_k',
            shape=gamma_k_shape,
            initializer=gamma_k_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=True)

        self.beta_outside = self.add_weight(
            name='beta_outside',
            shape=beta_outside_shape,
            initializer=beta_outside_initializer,
            dtype=self.dtype,
            trainable=True)
        
        self.gamma_outside = self.add_weight(
            name='gamma_outside',
            shape=gamma_outside_shape,
            initializer=gamma_outside_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=True)


        self.built = True

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        ndim = self._input_rank
        shape = self.gamma_k.get_shape().as_list()
        dilation_rate = [self.surround_dist]*(ndim-2)
        squared_inputs = tf.math.square(inputs)
        squared_input_groups =  tf.split(squared_inputs, self.norm_groups, -1)
        gamma_k_groups = tf.split(self.gamma_k, self.norm_groups, -1)

        # Compute normalization pool.

        # Pk for center group
        convolve_k = lambda inputs_i, gamma_k: tf.nn.convolution(inputs_i,
                                                                  gamma_k,
                                                                  strides=(1, 1),
                                                                  padding='SAME')
            
        Pk_groups= [convolve_k(i, k) for i,k in zip(squared_input_groups, gamma_k_groups)]
        Pk = tf.concat(Pk_groups, axis=3)
        beta = self.beta + self._beta_min
        norm_pool_k = tf.nn.bias_add(Pk, beta, data_format='N'+'DHW' [-(ndim - 2):]+'C') # NHWC
        norm_pool_k = tf.math.sqrt(norm_pool_k)
        norm_pool = tf.math.reciprocal(norm_pool_k)
        outputs = tf.multiply(inputs, norm_pool)
        outputs = outputs * self.gamma_outside + self.beta_outside
        outputs.set_shape(inputs.get_shape())

        return outputs

    def compute_output_shape(self, input_shape):
        channel_axis = self._channel_axis()
        input_shape = tensor_shape.TensorShape(input_shape)
        #if not 3 <= input_shape.ndim <= 5:
        #  raise ValueError('`input_shape` must be of rank 3 to 5, inclusive.')
        if input_shape[channel_axis].value is None:
            raise ValueError(
              'The channel dimension of `input_shape` must be defined.')
        return input_shape


# Weighted Normalization - center and surround normalization pool, with fixed weights
class WN_s_fix(tf.keras.layers.Layer):

    def get_config(self):
        config = {
            'surround_dist': self.surround_dist,
            'beta_min': self._beta_min,
            'beta_init': self._beta_init,
            'gamma_init': self._gamma_init,
            'data_format': self.data_format,
        }
        base_config = super(WN_s_fix, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def __init__(self,
               surround_dist=4,
               beta_min=1e-6,
               beta_init = 1e-6,
               gamma_init=1/16,
               data_format='channels_last',
               name=None,
               trainable=True,
               **kwargs):
        super(WN_s_fix, self).__init__(
            trainable=trainable,
            name=name)#,
            #**kwargs)
        self.surround_dist = surround_dist
        self._beta_min = beta_min
        self._beta_init = beta_init
        self._gamma_init = gamma_init
        self.data_format = data_format
        self._channel_axis()  # trigger ValueError early

    def _channel_axis(self):
        try:
            return {'channels_first': 1, 'channels_last': -1}[self.data_format]
        except KeyError:
            raise ValueError('Unsupported `data_format` for WN layer: {}.'.format(
            self.data_format))
    
    def build(self, input_shape):
        channel_axis = self._channel_axis()
        self.norm_groups = int(input_shape[channel_axis]/8)
        super(WN_s_fix, self).build(input_shape)
        num_channels = input_shape[channel_axis]
        if num_channels is None:
            raise ValueError('The channel dimension of the inputs to `WN` '
                           'must be defined.')

        if input_shape[channel_axis] % self.norm_groups != 0:
            raise ValueError('The number of channels must be a multiple of '
                           'the normalization_groups.')
        
        self._input_rank = input_shape.ndims

        def beta_outside_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return tf.zeros(shape, dtype='float32')

        def gamma_outside_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return tf.ones(shape, dtype='float32')

        def beta_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return self._beta_init * tf.ones(shape, dtype='float32')

        def gamma_k_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            one_tensor = tf.ones(shape, dtype='float32')
            return self._gamma_init * one_tensor

        def gamma_s_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            one_tensor = tf.ones(shape, dtype='float32')
            return self._gamma_init * one_tensor

        gamma_k_shape = [1,] * 2  +  [num_channels//self.norm_groups, num_channels]
        gamma_s_shape = [3,] * 2  +  [num_channels, 1]
        beta_shape = [num_channels]
        beta_outside_shape = [num_channels]
        gamma_outside_shape = [num_channels]

        self.beta = self.add_weight(
            name='beta',
            shape=beta_shape,
            initializer=beta_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=True)

        self.gamma_k = self.add_weight(
            name='gamma_k',
            shape=gamma_k_shape,
            initializer=gamma_k_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=False)

        self.gamma_s = self.add_weight(
            name='gamma_s',
            shape=gamma_s_shape,
            initializer=gamma_s_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=False)
        
        self.beta_outside = self.add_weight(
            name='beta_outside',
            shape=beta_outside_shape,
            initializer=beta_outside_initializer,
            dtype=self.dtype,
            trainable=True)
        
        self.gamma_outside = self.add_weight(
            name='gamma_outside',
            shape=gamma_outside_shape,
            initializer=gamma_outside_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=True)
        
        center_zero_tensor = tf.constant([[1.,1.,1.],[1.,0,1.],[1.,1.,1.]])  # center zero restriction. set to reparam_offset will lead to 0.
        center_zero_tensor = tf.stack([center_zero_tensor] * num_channels, axis=2)  # build a [3, 3, #c] tensor
        self.center_zero_tensor = tf.expand_dims(center_zero_tensor, axis = -1) # build a [3, 3, #c, 1] tensor

        self.built = True

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        ndim = self._input_rank
        shape = self.gamma_k.get_shape().as_list()
        dilation_rate = [self.surround_dist]*(ndim-2)
        squared_inputs = tf.math.square(inputs)
        squared_input_groups =  tf.split(squared_inputs, self.norm_groups, -1)
        gamma_k_groups = tf.split(self.gamma_k, self.norm_groups, -1)

        # Compute normalization pool.

        # Pk for center group
        convolve_k = lambda inputs_i, gamma_k: tf.nn.convolution(inputs_i,
                                                                  gamma_k,
                                                                  strides=(1, 1),
                                                                  padding='SAME')
            
        Pk_groups= [convolve_k(i, k) for i,k in zip(squared_input_groups, gamma_k_groups)]
        Pk = tf.concat(Pk_groups, axis=3)
        
        gamma_s = tf.math.multiply(self.gamma_s, self.center_zero_tensor)
        Ps = tf.nn.depthwise_conv2d(squared_inputs,
                                    gamma_s,
                                    strides=[1,1,1,1],
                                    padding='SAME',
                                    dilations=dilation_rate)
        beta = self.beta + self._beta_min
        norm_pool_ks = tf.nn.bias_add(tf.math.add(Pk, Ps), beta, data_format='N'+'DHW' [-(ndim - 2):]+'C') # NHWC
        norm_pool_ks = tf.math.sqrt(norm_pool_ks)

        norm_pool = tf.math.reciprocal(norm_pool_ks)
        outputs = tf.multiply(inputs, norm_pool)
        outputs = outputs * self.gamma_outside + self.beta_outside
        outputs.set_shape(inputs.get_shape())

        return outputs

    def compute_output_shape(self, input_shape):
        channel_axis = self._channel_axis()
        input_shape = tensor_shape.TensorShape(input_shape)
        #if not 3 <= input_shape.ndim <= 5:
        #  raise ValueError('`input_shape` must be of rank 3 to 5, inclusive.')
        if input_shape[channel_axis].value is None:
            raise ValueError(
              'The channel dimension of `input_shape` must be defined.')
        return input_shape

# Weighted Normalization - center only normalization pool, with fixed weights
class WN_c_fix(tf.keras.layers.Layer):

    def get_config(self):
        config = {
            'surround_dist': self.surround_dist,
            'beta_min': self._beta_min,
            'beta_init': self._beta_init,
            'gamma_init': self._gamma_init,
            'data_format': self.data_format,
        }
        base_config = super(WN_c_fix, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def __init__(self,
               surround_dist=1,
               beta_min=1e-6,
               beta_init = 1e-6,
               gamma_init=1/8,
               data_format='channels_last',
               name=None,
               trainable=True,
               **kwargs):
        super(WN_c_fix, self).__init__(
            trainable=trainable,
            name=name)#,
            #**kwargs)
    # TODO: check group_n is consistent with the  number of channels
        self.surround_dist = surround_dist
        self._beta_min = beta_min
        self._beta_init = beta_init
        self._gamma_init = gamma_init
        self.data_format = data_format
        self._channel_axis()  # trigger ValueError early

    def _channel_axis(self):
        try:
            return {'channels_first': 1, 'channels_last': -1}[self.data_format]
        except KeyError:
            raise ValueError('Unsupported `data_format` for WN layer: {}.'.format(
            self.data_format))
    
    def build(self, input_shape):
        channel_axis = self._channel_axis()
        self.norm_groups = int(input_shape[channel_axis]/8)
        super(WN_c_fix, self).build(input_shape)
        num_channels = input_shape[channel_axis]
        if num_channels is None:
            raise ValueError('The channel dimension of the inputs to `WN` '
                           'must be defined.')

        if input_shape[channel_axis] % self.norm_groups != 0:
            raise ValueError('The number of channels must be a multiple of '
                           'the normalization_groups.')
        
        self._input_rank = input_shape.ndims

        def beta_outside_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return tf.zeros(shape, dtype='float32')

        def gamma_outside_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return tf.ones(shape, dtype='float32')

        def beta_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return self._beta_init * tf.ones(shape, dtype='float32')

        def gamma_k_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            one_tensor = tf.ones(shape, dtype='float32')
            return self._gamma_init * one_tensor

        gamma_k_shape = [1,] * 2  +  [num_channels//self.norm_groups, num_channels]
        beta_shape = [num_channels]
        beta_outside_shape = [num_channels]
        gamma_outside_shape = [num_channels]

        self.beta = self.add_weight(
            name='beta',
            shape=beta_shape,
            initializer=beta_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=True)

        self.gamma_k = self.add_weight(
            name='gamma_k',
            shape=gamma_k_shape,
            initializer=gamma_k_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=False)

        self.beta_outside = self.add_weight(
            name='beta_outside',
            shape=beta_outside_shape,
            initializer=beta_outside_initializer,
            dtype=self.dtype,
            trainable=True)
        
        self.gamma_outside = self.add_weight(
            name='gamma_outside',
            shape=gamma_outside_shape,
            initializer=gamma_outside_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=True)


        self.built = True

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        ndim = self._input_rank
        shape = self.gamma_k.get_shape().as_list()
        squared_inputs = tf.math.square(inputs)
        squared_input_groups =  tf.split(squared_inputs, self.norm_groups, -1)
        gamma_k_groups = tf.split(self.gamma_k, self.norm_groups, -1)

        # Compute normalization pool.

        # Pk for center group
        convolve_k = lambda inputs_i, gamma_k: tf.nn.convolution(inputs_i,
                                                                  gamma_k,
                                                                  strides=(1, 1),
                                                                  padding='SAME')
            
        Pk_groups= [convolve_k(i, k) for i,k in zip(squared_input_groups, gamma_k_groups)]
        Pk = tf.concat(Pk_groups, axis=3)
        beta = self.beta + self._beta_min
        norm_pool_k = tf.nn.bias_add(Pk, beta, data_format='N'+'DHW' [-(ndim - 2):]+'C') # NHWC
        norm_pool_k = tf.math.sqrt(norm_pool_k)
        norm_pool = tf.math.reciprocal(norm_pool_k)
        outputs = tf.multiply(inputs, norm_pool)
        outputs = outputs * self.gamma_outside + self.beta_outside
        outputs.set_shape(inputs.get_shape())

        return outputs

    def compute_output_shape(self, input_shape):
        channel_axis = self._channel_axis()
        input_shape = tensor_shape.TensorShape(input_shape)
        #if not 3 <= input_shape.ndim <= 5:
        #  raise ValueError('`input_shape` must be of rank 3 to 5, inclusive.')
        if input_shape[channel_axis].value is None:
            raise ValueError(
              'The channel dimension of `input_shape` must be defined.')
        return input_shape


####################################################################################
# Implementation of Divisive Normalization:
# Ren, Mengye, et al. "Normalizing the normalizers: Comparing and extending network 
# normalization schemes." arXiv preprint arXiv:1611.04520 (2016).
class DN(tf.keras.layers.Layer):
    
    def __init__(self,
               name=None,
               trainable=True,
               **kwargs):
        super(DN, self).__init__(
            trainable=trainable,
            name=name)#,
            #**kwargs)

    def build(self, input_shape):
        super(DN, self).build(input_shape)
        channel_axis = -1
        num_channels = input_shape[channel_axis]     
        self._input_rank = input_shape.ndims

        def beta_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return tf.zeros(shape, dtype='float32')

        def gamma_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return tf.ones(shape, dtype='float32')
         
        def sigma_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            return tf.ones(shape, dtype='float32')

        beta_shape = [num_channels]

        gamma_shape = [num_channels]

        self.beta = self.add_weight(
            name='beta',
            shape=beta_shape,
            initializer=beta_initializer,
            dtype=self.dtype,
            trainable=True)

        self.gamma = self.add_weight(
            name='gamma',
            shape=gamma_shape,
            initializer=gamma_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=True)
        
        self.sigma = self.add_weight(
            name='sigma',
            initializer=sigma_initializer,
            dtype=self.dtype,
            constraint=keras.constraints.NonNeg(),
            trainable=True)
        
        self.convfilter = tf.ones([3, 3, num_channels, 1]) / (9*num_channels)

        self.built = True

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        ndim = self._input_rank
        inputs_mean = tf.nn.conv2d(inputs, self.convfilter, strides=[1, 1, 1, 1], padding='SAME')
        inputs_center = inputs-inputs_mean
        squared_inputs = tf.math.square(inputs_center)
        squared_inputs_mean = tf.nn.conv2d(squared_inputs, self.convfilter, strides=[1, 1, 1, 1], padding='SAME')
        norm_pool = tf.math.sqrt(squared_inputs_mean+tf.math.square(self.sigma))
        outputs = inputs_center / norm_pool
        outputs = outputs * self.gamma + self.beta
        outputs.set_shape(inputs.get_shape())

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


#############################################################################

# Implementation of Generalized Divisive Normalization
# "Density Modeling of Images using a Generalized Normalization Transformation"
# Johannes Ballé, Valero Laparra, Eero P. Simoncelli
# "End-to-end Optimized Image Compression"
# Johannes Ballé, Valero Laparra, Eero P. Simoncelli
class GDN(tf.keras.layers.Layer):

    def get_config(self):
        config = {
              'beta_min': self._beta_min,
              'gamma_init': self._gamma_init,
              'reparam_offset': self._reparam_offset,
              'data_format': self.data_format,
        }
        base_config = super(GDN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def __init__(self,
               inverse=False,
               beta_min=1e-6,
               gamma_init=.1,
               reparam_offset=2**-18,
               data_format='channels_last',
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
        super(GDN, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer)#,
        #**kwargs)
        self.inverse = inverse
        self._beta_min = beta_min
        self._gamma_init = gamma_init
        self._reparam_offset = reparam_offset
        self.data_format = data_format
        self._channel_axis()  # trigger ValueError early
    #self.input_spec = base.InputSpec(min_ndim=3, max_ndim=5)

    def _channel_axis(self):
        try:
            return {'channels_first': 1, 'channels_last': -1}[self.data_format]
        except KeyError:
            raise ValueError('Unsupported `data_format` for GDN layer: {}.'.format(
                self.data_format))

    @staticmethod
    def _lower_bound(inputs, bound, name=None):

        with tf.name_scope('GDNLowerBound') as scope:
            inputs = tf.convert_to_tensor(inputs, name='inputs')
            bound = tf.convert_to_tensor(bound, name='bound')
        with tf.compat.v1.get_default_graph().gradient_override_map(
            {'Maximum': 'GDNLowerBound'}):
            return tf.maximum(inputs, bound, name=scope)

    @staticmethod
    def _lower_bound_grad(op, grad):

        inputs = op.inputs[0]
        bound = op.inputs[1]
        pass_through_if = tf.logical_or(inputs >= bound, grad < 0)
        return [tf.cast(pass_through_if, grad.dtype) * grad, None]

    def build(self, input_shape):
        channel_axis = self._channel_axis()
        input_shape = tf.TensorShape(input_shape)
        num_channels = input_shape.dims[channel_axis].value
        if num_channels is None:
            raise ValueError('The channel dimension of the inputs to `GDN` '
                       'must be defined.')
        self._input_rank = input_shape.ndims
    #self.input_spec = base.InputSpec(
    #    ndim=input_shape.ndims, axes={channel_axis: num_channels})

        pedestal = tf.constant(self._reparam_offset**2, dtype=self.dtype)
        beta_bound = tf.constant((self._beta_min + self._reparam_offset**2)**.5, dtype=self.dtype)
        gamma_bound = tf.constant(self._reparam_offset, dtype=self.dtype)

        def beta_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            pedestal = tf.constant(self._reparam_offset**2, dtype=self.dtype)
            return tf.sqrt(tf.ones(shape, dtype=dtype) + pedestal)

        def gamma_initializer(shape, dtype=None, partition_info=None):
            del partition_info  # unused
            assert len(shape) == 2
            assert shape[0] == shape[1]
            eye = tf.eye(shape[0], dtype=dtype)
            pedestal = tf.constant(self._reparam_offset**2, dtype=self.dtype)
            return tf.sqrt(self._gamma_init * eye + pedestal)

        beta = self.add_variable(
            'reparam_beta',
            shape=[num_channels],
            initializer=beta_initializer,
            dtype=self.dtype,
            trainable=True)
        beta = self._lower_bound(beta, beta_bound)
        self.beta = tf.square(beta) - pedestal

        gamma = self.add_variable(
            'reparam_gamma',
            shape=[num_channels, num_channels],
            initializer=gamma_initializer,
            dtype=self.dtype,
            trainable=True)
        gamma = self._lower_bound(gamma, gamma_bound)
        self.gamma = tf.square(gamma) - pedestal

        self.built = True

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        ndim = self._input_rank

        shape = self.gamma.get_shape().as_list()
        gamma = tf.reshape(self.gamma, (ndim - 2) * [1] + shape)

    # Compute normalization pool.
        if self.data_format == 'channels_first':
            norm_pool = tf.nn.convolution(
                tf.square(inputs),
                gamma,
                data_format='NC' + 'DHW' [-(ndim - 2):])
            if ndim == 3:
                norm_pool = tf.expand_dims(norm_pool, 2)
                norm_pool = tf.nn.bias_add(norm_pool, self.beta, data_format='NCHW')
                norm_pool = tf.squeeze(norm_pool, [2])
            elif ndim == 5:
                shape = tf.shape(norm_pool)
                norm_pool = tf.reshape(norm_pool, shape[:3] + [-1])
                norm_pool = tf.nn.bias_add(norm_pool, self.beta, data_format='NCHW')
                norm_pool = tf.reshape(norm_pool, shape)
            else:  # ndim == 4
                norm_pool = tf.nn.bias_add(norm_pool, self.beta, data_format='NCHW')
        else:  # channels_last
            norm_pool = tf.nn.convolution(tf.square(inputs), gamma)
            norm_pool = tf.nn.bias_add(norm_pool, self.beta, data_format='NHWC')
        norm_pool = tf.sqrt(norm_pool)

        if self.inverse:
            outputs = inputs * norm_pool
        else:
            outputs = inputs / norm_pool
        outputs.set_shape(inputs.get_shape())
        return outputs

    def compute_output_shape(self, input_shape):
        channel_axis = self._channel_axis()
        input_shape = tf.TensorShape(input_shape)
        if not 3 <= input_shape.ndim <= 5:
            raise ValueError('`input_shape` must be of rank 3 to 5, inclusive.')
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
              'The channel dimension of `input_shape` must be defined.')
        return input_shape

try:
    tf.RegisterGradient('GDNLowerBound')(GDN._lower_bound_grad)  # pylint:disable=protected-access
except:
    pass

#######################################################################################

# Implementation of Local Response Normalization
# Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with
# deep convolutional neural networks." Advances in neural information processing systems 
# 25 (2012): 1097-1105.
class LRN(tf.keras.layers.Layer):
    def get_config(self):
        config = {
            'alpha': self.alpha,
            'k': self.k,
            'beta': self.beta,
            'n': self.n,
            'data_format': self.data_format,
        }
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def __init__(self, 
                 alpha=1, 
                 k=1, 
                 beta=0.5, 
                 n=7, 
                 data_format="channels_last",
                 **kwargs
                ):
        if n % 2 == 0:
            raise NotImplementedError("LRN only works with odd n. n provided: " + str(n))
        super(LRN, self).__init__()
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        self.data_format = data_format

    def build(self, input_shape):
        super(LRN, self).build(input_shape)

    def call(self, X):
        half_n = self.n // 2
        return tf.nn.local_response_normalization(
                X,
                depth_radius=half_n,
                bias=self.k,
                alpha=self.alpha,
                beta=self.beta
                )
        
    def compute_output_shape(self, input_shape):
        return input_shape
#######################################################################

# Trivial layer for control
class no_norm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(no_norm, self).__init__()

    def call(self, inputs):
        return inputs
