import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class Layers(object):

    def __init__(self, parameters={}):

        self.num_params = 0
        self.initializer, self.parameters = {}, parameters

    def __initializer_random(self, shape, name=''):

        try: return self.initializer[name]
        except:
            try: initializer = tf.compat.v1.constant_initializer(self.parameters[name])
            except:
                try: stddev = np.sqrt(2/(shape[-2]+shape[-1]))
                except: stddev = np.sqrt(2/shape[-1])
                self.initializer[name] = tf.compat.v1.random_normal_initializer(\
                    mean=0.0, stddev=stddev, dtype=tf.dtypes.float32)
            return self.initializer[name]

    def __initializer_constant(self, shape, constant=0, name=''):

        try: return self.initializer[name]
        except:
            try: self.initializer[name] = tf.compat.v1.constant_initializer(self.parameters[name])
            except: self.initializer[name] = tf.compat.v1.constant_initializer(np.ones(shape)*constant)
            return self.initializer[name]

    def __get_variable(self, shape, constant=None, trainable=True, name=''):

        try: return self.parameters[name]
        except:
            try: initializer = self.__initializer_constant(shape=shape, constant=constant, name=name)
            except: initializer = self.__initializer_random(shape=shape, name=name)

            tmp_num = 1
            for num in shape:
                tmp_num *= num
            self.num_params += tmp_num
            self.parameters[name] = tf.compat.v1.get_variable(name=name, \
                shape=shape, initializer=initializer, trainable=trainable)

            return self.parameters[name]

    def activation(self, x, activation=None, name=''):

        if(activation is None): return x
        elif("sigmoid" == activation):
            return tf.compat.v1.nn.sigmoid(x, name='%s_sigmoid' %(name))
        elif("tanh" == activation):
            return tf.compat.v1.nn.tanh(x, name='%s_tanh' %(name))
        elif("relu" == activation):
            return tf.compat.v1.nn.relu(x, name='%s_relu' %(name))
        elif("lrelu" == activation):
            return tf.compat.v1.nn.leaky_relu(x, name='%s_lrelu' %(name))
        elif("elu" == activation):
            return tf.compat.v1.nn.elu(x, name='%s_elu' %(name))
        else: return x

    def maxpool(self, x, ksize, strides, padding, name='', verbose=True):

        y = tf.compat.v1.nn.max_pool(value=x, \
            ksize=ksize, strides=strides, padding=padding, name=name)

        if(verbose): print("Pool", x.shape, "->", y.shape)
        return y

    def batch_normalization(self, x, trainable=True, training=None, name='', verbose=True):

        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/layers/batch_normalizationalization
        shape_in = x.get_shape().as_list()[-1]

        beta = self.__initializer_constant(shape=[shape_in], constant=0.0, name='%s_beta' %(name))
        gamma = self.__initializer_constant(shape=[shape_in], constant=1.0, name='%s_gamma' %(name))
        mv_mean = self.__initializer_constant(shape=[shape_in], constant=0.0, name='%s_mv_mean' %(name))
        mv_var = self.__initializer_constant(shape=[shape_in], constant=1.0, name='%s_mv_var' %(name))

        y = tf.compat.v1.layers.batch_normalization(inputs=x, \
            beta_initializer=beta,
            gamma_initializer=gamma,
            moving_mean_initializer=mv_mean,
            moving_variance_initializer=mv_var, \
            training=training, trainable=trainable, name=name)

        if(verbose): print("BN", x.shape, ">", y.shape)
        return y

    def instance_normalization(self, x, batch_size, trainable=True, training=None, name='', verbose=True):
      with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        x_org = x
        shapes = x.get_shape().as_list()[1:]
        shape_in = x.get_shape().as_list()[-1]
        x = tf.split(x, batch_size, axis=0)   # [batch, other+dim] -> batch * [other_dim]
        
        # res = np.zeros(shapes)
        # res = np.tile(np.expand_dims(res, 0), [batch_size, 1, 1, 1])
        # res = tf.convert_to_tensor(res, dtype=tf.float32)
        res = []

        beta = self.__initializer_constant(shape=[shape_in], constant=0.0, name='%s_beta' %(name))
        gamma = self.__initializer_constant(shape=[shape_in], constant=1.0, name='%s_gamma' %(name))
        mv_mean = self.__initializer_constant(shape=[shape_in], constant=0.0, name='%s_mv_mean' %(name))
        mv_var = self.__initializer_constant(shape=[shape_in], constant=1.0, name='%s_mv_var' %(name))
        for i, instance in enumerate (x):
          y = tf.compat.v1.layers.batch_normalization(inputs=instance, \
              beta_initializer=beta,
              gamma_initializer=gamma,
              moving_mean_initializer=mv_mean,
              moving_variance_initializer=mv_var, \
              training=training, trainable=trainable, name=name)
          # print(res.shape)
          # print(y[0].shape)

          res.append(y[0])
          
        y = tf.stack(res, 0)
        # print(y.shape)

        if(verbose): print("IN", x_org.shape, ">", y.shape)
      
      return y


    # def layer_normalization(self, x, batch_size, trainable=True, training=None, name='', verbose=True):
    #   with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
    #     x_org = x
    #     shapes = x.get_shape().as_list()[1:]
    #     shape_in = x.get_shape().as_list()[-1]
    #     x = tf.split(x, batch_size, axis=0)   # [batch, other+dim] -> batch * [other_dim]
        
    #     # res = np.zeros(shapes)
    #     # res = np.tile(np.expand_dims(res, 0), [batch_size, 1, 1, 1])
    #     # res = tf.convert_to_tensor(res, dtype=tf.float32)
    #     res = []

    #     beta = self.__initializer_constant(shape=[1], constant=0.0, name='%s_beta' %(name))
    #     gamma = self.__initializer_constant(shape=[1], constant=1.0, name='%s_gamma' %(name))
    #     mv_mean = self.__initializer_constant(shape=[1], constant=0.0, name='%s_mv_mean' %(name))
    #     mv_var = self.__initializer_constant(shape=[1], constant=1.0, name='%s_mv_var' %(name))
    #     for i, instance in enumerate (x):
    #       channels = tf.split(x, shape_in, axis=-1)
    #       ch = []
    #       for j, channel in enumerate(channels):
    #       y = tf.compat.v1.layers.batch_normalization(inputs=channels, \
    #           beta_initializer=beta,
    #           gamma_initializer=gamma,
    #           moving_mean_initializer=mv_mean,
    #           moving_variance_initializer=mv_var, \
    #           training=training, trainable=trainable, name=name)

    #         ch.append(y[0])
    #       res.append(y[0])
          
    #     y = tf.stack(res, 0)
    #     # print(y.shape)

    #     if(verbose): print("IN", x_org.shape, ">", y.shape)

    def conv2d(self, x, stride, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], \
        padding='SAME', batch_norm=False, training=None, activation=None, name='', verbose=True, norm_type="IN"):

        w = self.__get_variable(shape=filter_size, \
            trainable=True, name='%s_w' %(name))
        b = self.__get_variable(shape=[filter_size[-1]], \
            trainable=True, name='%s_b' %(name))

        wx = tf.compat.v1.nn.conv2d(
            input=x,
            filter=w,
            strides=[1, stride, stride, 1],
            padding=padding,
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv' %(name),
        )
        print(wx.shape)
        print(b.shape)
        y = tf.math.add(wx, b, name='%s_add' %(name))
        if(verbose): print("Conv", x.shape, "->", y.shape)

        if(batch_norm): 
          if norm_type == "IN":
            # instance normalization
            y = tfa.layers.InstanceNormalization(axis=-1, 
                              center=True, 
                              scale=True,
                              beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(y)
          else:
            # layer normalization
            y = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])(y)
            

        return self.activation(x=y, activation=activation, name=name)

    def convt2d(self, x, stride, output_shape, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], \
        padding='SAME', batch_norm=False, training=None, activation=None, name='', verbose=True):

        for idx_os, _ in enumerate(output_shape):
            if(idx_os == 0): continue
            output_shape[idx_os] = int(output_shape[idx_os])

        w = self.__get_variable(shape=filter_size, \
            trainable=True, name='%s_w' %(name))
        b = self.__get_variable(shape=[filter_size[-2]], \
            trainable=True, name='%s_b' %(name))

        wx = tf.compat.v1.nn.conv2d_transpose(
            value=x,
            filter=w,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv_tr' %(name),
        )
        # print(wx.shape)
        # print(b.shape)
        y = tf.math.add(wx, b, name='%s_add' %(name))

        if(verbose): print("ConvT", x.shape, "->", y.shape)

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=True, training=training, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)

    def fc(self, x, units, use_bias=True, scope='fc'):
        with tf.compat.v1.variable_scope(scope):
            x = self.flatten(x)
            x = tf.compat.v1.layers.dense(x, units=units,
                                use_bias=use_bias)

            return x

    def flatten(self, x) :
      return tf.compat.v1.layers.flatten(x)

    def fully_connected(self, x, c_out, \
        batch_norm=False, training=None, activation=None, name='', verbose=True):

        c_in, c_out = x.get_shape().as_list()[-1], int(c_out)

        w = self.__get_variable(shape=[c_in, c_out], \
            trainable=True, name='%s_w' %(name))
        b = self.__get_variable(shape=[c_out], \
            trainable=True, name='%s_b' %(name))

        wx = tf.compat.v1.matmul(x, w, name='%s_mul' %(name))
        y = tf.math.add(wx, b, name='%s_add' %(name))
        if(verbose): print("FC", x.shape, "->", y.shape)

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=True, training=training, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)
