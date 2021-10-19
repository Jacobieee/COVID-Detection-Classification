"""
some code adapted from https://github.com/YeongHyeon/DGM-TF.
"""
import os
import numpy as np
import tensorflow as tf
import source.layers as lay
# import psutil
# from os import getpid
import tensorflow_addons as tfa
# import tensorflow.python.keras.utils as generic_utils
# from bert4keras.optimizers import *
from tensorflow.keras import layers

class DGM(object):

    def __init__(self, \
        height, width, channel, ksize, zdim, \
        learning_rate=1e-5, path='', verbose=True):
        tf.compat.v1.disable_eager_execution()
        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel, self.ksize, self.zdim = \
            height, width, channel, ksize, zdim
        self.learning_rate = learning_rate
        self.path_ckpt = path

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel], \
            name="x")
        self.y = tf.compat.v1.placeholder(tf.float32, [None, 1], \
            name="y")
        self.batch_size = tf.compat.v1.placeholder(tf.int32, shape=[], \
            name="batch_size")
        self.training = tf.compat.v1.placeholder(tf.bool, shape=[], \
            name="training")
        self.x_normals = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel], \
            name="x_normals")


        self.layer = lay.Layers()


        self.epoch = -1

        self.variables, self.losses = {}, {}
        self.__build_model(x=self.x, ksize=self.ksize, verbose=verbose)
        self.__build_loss()

        with tf.compat.v1.control_dependencies(self.variables['ops_d']):

            if self.epoch > 0 and self.epoch % 10 == 0:
              self.learning_rate /= 2

            self.optimizer_d = tf.compat.v1.train.AdamOptimizer( \
                self.learning_rate, beta1=0.5, name='Adam_d').minimize(\
                self.losses['loss_d'], var_list=self.variables['params_d'])

            # AdamW = extend_with_weight_decay(self.optimizer_d, 'AdamW_D')
            # self.optimizer_d = AdamW(weight_decay_rate=1e-4)
  
          #   self.optimizer_d = tfa.optimizers.AdamW(
          #     weight_decay=1e-4,
          #     learning_rate=self.learning_rate,
          #     beta_1=0.5,
          #     beta_2=0.999,
          #     name='Adam_d'
          # ).minimize(loss=self.losses['loss_d'], var_list=self.variables['params_d'], tape=tf.compat.v1.GradientTape.__enter__(self))

        with tf.compat.v1.control_dependencies(self.variables['ops_g']):
            
            self.optimizer_g = tf.compat.v1.train.AdamOptimizer( \
                self.learning_rate, beta1=0.5, name='Adam_g').minimize(\
                self.losses['loss_g'], var_list=self.variables['params_g'])

            
            # opt = tfa.optimizers.AdamW(
            #   weight_decay=1e-4,
            #   learning_rate=self.learning_rate,
            #   beta_1=0.5,
            #   beta_2=0.999,
            #   name='Adam_g'
            # ).minimize(loss=self.losses['loss_g'], var_list=self.variables['params_g'], tape=tf.compat.v1.GradientTape(gradient))


        tf.compat.v1.summary.scalar('DGM/loss_a', self.losses['loss_a'])
        tf.compat.v1.summary.scalar('DGM/loss_r', self.losses['loss_r'])
        tf.compat.v1.summary.scalar('DGM/loss_tv', self.losses['loss_tv'])
        tf.compat.v1.summary.scalar('DGM/loss_g', self.losses['loss_g'])
        tf.compat.v1.summary.scalar('DGM/loss_d', self.losses['loss_d'])
        self.summaries = tf.compat.v1.summary.merge_all()

        self.__init_session(path=self.path_ckpt)

    def step(self, x, y, x_normals, iteration=0, epoch=-1, training=False):
        # tf.compat.v1.reset_default_graph()
        feed_tr = {self.x:x, self.y:y, self.x_normals:x_normals, self.batch_size:x.shape[0], self.training:True}
        feed_te = {self.x:x, self.y:y, self.x_normals:x_normals, self.batch_size:x.shape[0], self.training:False}

        self.epoch = epoch

        summary_list = []
        #process = psutil.Process(getpid())
        #before = process.memory_percent()
        if(training):
            try:
                _, summaries = self.sess.run([self.optimizer_d, self.summaries], \
                    feed_dict=feed_tr, options=self.run_options, run_metadata=self.run_metadata)
                summary_list.append(summaries)
                # self.sess.close()

                _, summaries = self.sess.run([self.optimizer_g, self.summaries], \
                    feed_dict=feed_tr, options=self.run_options, run_metadata=self.run_metadata)
                summary_list.append(summaries)
                # self.sess.close()
            except:
                _, summaries = self.sess.run([self.optimizer_d, self.summaries], \
                    feed_dict=feed_tr)
                summary_list.append(summaries)
                # self.sess.close()

                _, summaries = self.sess.run([self.optimizer_g, self.summaries], \
                    feed_dict=feed_tr)
                summary_list.append(summaries)
                # self.sess.close()

            for summaries in summary_list:
                self.summary_writer.add_summary(summaries, iteration)
        #after1 = process.memory_percent()
        #print("MEMORY CHANGE 1 %.4f -> %.4f" % (before, after1))
        y_hat, loss_d, loss_g, mse, residue, adv_loss, tv, r, r1, r2, r3 = \
            self.sess.run([self.variables['y_hat'], self.losses['loss_d'], self.losses['loss_g'], self.losses['mse'], self.variables['a'], self.losses['loss_a'], self.losses['loss_tv'], self.losses['loss_r'], self.losses['1'], self.losses['2'], self.losses['3']], \
            feed_dict=feed_te)
        #after2 = process.memory_percent()
        #print("MEMORY CHANGE 2 %.4f -> %.4f" % (after1, after2))
        outputs = {'y_hat':y_hat, 'loss_d':loss_d, 'loss_g':loss_g, 'mse':mse, 'residue': residue, 'adv': adv_loss, 'tv': tv, 'r': r, 'r1': r1, 'r2': r2, 'r3': r3}
        # tf.contrib.keras.backend.clear_session()
        return outputs

    def save_parameter(self, model='model_checker', epoch=-1):

        self.saver.save(self.sess, os.path.join(self.path_ckpt, model))
        if(epoch >= 0): self.summary_writer.add_run_metadata(self.run_metadata, 'epoch-%d' % epoch)

    def load_parameter(self, model='model_checker'):

        path_load = os.path.join(self.path_ckpt, '%s.index' %(model))
        if(os.path.exists(path_load)):
            print("\nRestoring parameters")
            self.saver.restore(self.sess, path_load.replace('.index', ''))

    def confirm_params(self, verbose=True):

        print("\n* Parameter arrange")

        ftxt = open("list_parameters.txt", "w")
        for var in tf.compat.v1.trainable_variables():
            text = "Trainable: " + str(var.name) + str(var.shape)
            if(verbose): print(text)
            ftxt.write("%s\n" %(text))
        ftxt.close()

    def confirm_bn(self, verbose=True):

        print("\n* Confirm Batch Normalization")

        t_vars = tf.compat.v1.trainable_variables()
        for var in t_vars:
            if('bn' in var.name):
                tmp_x = np.zeros((1, self.height, self.width, self.channel))
                tmp_y = np.zeros((1, 1))
                values = self.sess.run(var, \
                    feed_dict={self.x:tmp_x, self.y:tmp_y, self.batch_size:1, self.training:False})
                if(verbose): print(var.name, var.shape)
                if(verbose): print(values)

    def loss_l1(self, x, reduce=None):

        distance = tf.compat.v1.reduce_mean(\
            tf.math.abs(x), axis=reduce)

        return distance

    def loss_l2(self, x, reduce=None):

        distance = tf.compat.v1.reduce_sum(\
            tf.math.sqrt(\
            tf.math.square(x) + 1e-9), axis=reduce)

        return distance


    def loss_square(self, x, reduce=None):

        distance = tf.compat.v1.reduce_sum(tf.math.square(x), axis=reduce) + 1e-9

        return distance
    

    def __init_session(self, path):

        try:
            sess_config = tf.compat.v1.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            # tf.compat.v1.disable_eager_execution()
            # tf.compat.v1.reset_default_graph()
            self.sess = tf.compat.v1.Session(config=sess_config)

            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()

            self.summary_writer = tf.compat.v1.summary.FileWriter(path, self.sess.graph)
            self.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            self.run_metadata = tf.compat.v1.RunMetadata()

            # tf.compat.v1.enable_eager_execution()
        except: pass

    def __build_loss(self):

        # Adversarial loss
        least_square_term_1 = tf.math.square(\
            self.variables['d_fake'] - tf.zeros_like(self.variables['d_fake']) + 1e-9)

        least_square_term_2 = tf.math.square(\
            self.variables['d2_fake'] - tf.zeros_like(self.variables['d2_fake']) + 1e-9)

        least_square_term_3 = tf.math.square(\
            self.variables['d3_fake'] - tf.zeros_like(self.variables['d3_fake']) + 1e-9)

        least_square_term = least_square_term_1 + least_square_term_2 + least_square_term_3

        self.losses['loss_a'] = tf.compat.v1.reduce_mean(0.5 * least_square_term)

        # Reconstruction losses
        self.losses['loss_r1'] = \
            self.loss_l1(self.variables['y_hat'] + self.variables['a'] - self.x, [1, 2, 3])
        self.losses['loss_r2'] = \
            self.loss_l1(self.variables['z_phat'] - self.x, [1, 2, 3])

        y_ext = tf.compat.v1.reshape(self.y, shape=[self.batch_size, 1, 1, 1], \
                name="y_ext")
        self.losses['loss_r3'] = \
            self.loss_l1(self.variables['a'] * y_ext, [1, 2, 3])

        # Total variation loss
        shift_i = tf.roll(self.variables['y_hat'], shift=[1], axis=[1])
        shift_j = tf.roll(self.variables['y_hat'], shift=[1], axis=[2])
        tv_i_term = \
            tf.math.abs(self.variables['y_hat'] - shift_i)
        tv_j_term = \
            tf.math.abs(self.variables['y_hat'] - shift_j)
        self.losses['loss_tv'] = \
            tf.compat.v1.reduce_mean(tv_i_term + tv_j_term, axis=[1, 2, 3])

        lambda_a, lambda_r, lambda_tv = 10, 10, 1 # in paper
        term_a = lambda_a * self.losses['loss_a']
        term_r = lambda_r * \
            (self.losses['loss_r1'] + self.losses['loss_r2'] + self.losses['loss_r3'])
        term_tv = lambda_tv * self.losses['loss_tv']
        self.losses['1'] = tf.compat.v1.reduce_mean(self.losses['loss_r1'])
        self.losses['2'] = tf.compat.v1.reduce_mean(self.losses['loss_r2'])
        self.losses['3'] = tf.compat.v1.reduce_mean(self.losses['loss_r3'])
        self.losses['loss_g'] = \
            tf.compat.v1.reduce_mean(term_a + term_r + term_tv)
        self.losses['loss_a'] = tf.compat.v1.reduce_mean(term_a)
        self.losses['loss_r'] = tf.compat.v1.reduce_mean(term_r)
        self.losses['loss_tv'] = tf.compat.v1.reduce_mean(term_tv)
        
        # rec = "| Adversarial Loss:%.3f | Reconstruction:%.3f | Total Variance:%.3f" % (l_a, l_r, l_tv)
        # with open("loss_record.txt", 'a') as f:
        #   f.write(f"{rec}\n")
        # For D optimization
        d_real_term = \
            self.loss_square(self.variables['d_real'] - (tf.zeros_like(self.variables['d_real'])+0.1), [1])
        d_fake_term = \
            self.loss_square(self.variables['d_fake'] - (tf.ones_like(self.variables['d_fake'])-0.1), [1])

        d2_real_term = \
            self.loss_square( self.variables['d2_real'] - (tf.zeros_like(self.variables['d2_real'])+0.1), [1])

        d2_fake_term = \
            self.loss_square( self.variables['d2_fake'] - (tf.ones_like(self.variables['d2_fake'])-0.1), [1])

        d3_real_term = \
            self.loss_square( self.variables['d3_real'] - (tf.zeros_like(self.variables['d3_real'])+0.1), [1])

        d3_fake_term = \
            self.loss_square( self.variables['d3_fake'] - (tf.ones_like(self.variables['d3_fake'])-0.1), [1])

        self.losses['loss_d'] = tf.compat.v1.reduce_mean(0.5 * tf.math.add_n([(d_real_term + d_fake_term), (d2_real_term + d2_fake_term), (d3_real_term + d3_fake_term)]))

        self.losses['mse'] = \
            tf.compat.v1.reduce_mean(self.loss_l2(self.variables['y_hat'] - self.x, [1, 2, 3]))

        self.variables['params_d'], self.variables['params_g'] = [], []
        for var in tf.compat.v1.trainable_variables():
            text = "Trainable: " + str(var.name) + str(var.shape)
            # print(text)
            if('dis_' in var.name): self.variables['params_d'].append(var)
            else: self.variables['params_g'].append(var)

        self.variables['ops_d'], self.variables['ops_g'] = [], []
        for ops in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS):
            if('dis_' in ops.name): self.variables['ops_d'].append(ops)
            else: self.variables['ops_g'].append(ops)

    def __build_model(self, x, ksize=3, verbose=True):

        print("\n-*-*- Flow 1 -*-*-")
        # Module 1: E_G
        self.variables['c_z'], _ = \
            self.__encoder(x=x, ksize=ksize, outdim=self.zdim, reuse=False, \
            name='enc_g', norm=True, verbose=verbose)
        # Module 2: D_G
        self.variables['y_hat'] = \
            self.__decoder(z=self.variables['c_z'], ksize=ksize, reuse=False, \
            name='gen_g', norm=True, verbose=verbose)
        # Module 3: D
        self.variables['d_real'] = \
            self.__D2(x=self.x_normals, ksize=ksize, outdim=1, reuse=False, \
            name='dis_1', norm=True, dnum=0)
        # Revisit D
        self.variables['d_fake'] = \
            self.__D2(x=self.variables['y_hat'], ksize=ksize, outdim=1, reuse=False, \
            name='dis_1', norm=True, dnum=0)

        self.variables['d2_real'] = \
            self.__D2(x=self.x_normals, ksize=ksize, outdim=1, reuse=False, \
            name='dis_2', norm=True, dnum=1)

        self.variables['d2_fake'] = \
            self.__D2(x=self.variables['y_hat'], ksize=ksize, outdim=1, reuse=True, \
            name='dis_2', norm=True, dnum=1)

        self.variables['d3_real'] = \
            self.__D2(x=self.x_normals, ksize=ksize, outdim=1, reuse=False, \
            name='dis_3', norm=True, dnum=2)

        self.variables['d3_fake'] = \
            self.__D2(x=self.variables['y_hat'], ksize=ksize, outdim=1, reuse=True, \
            name='dis_3', norm=True, dnum=2)



        print("\n-*-*- Flow 2 -*-*-")
        # Module 4: E_F
        self.variables['c_s'], _ = \
            self.__encoder(x=x, ksize=ksize, outdim=self.zdim, reuse=False, \
            name='enc_f', norm=False, verbose=verbose)
        # Module 5: D_F
        self.variables['a'] = \
            self.__decoder(z=self.variables['c_s'], ksize=ksize, reuse=False, \
            name='gen_f', norm=True, verbose=verbose)

        print("\n-*-*- Flow 3 -*-*-")
        # Vector joint
        self.variables['c_joint'] = \
            tf.concat([self.variables['c_z'], self.variables['c_s']], -1)
        # Module 6: D_J
        self.variables['z_phat'] = \
            self.__decoder(z=self.variables['c_joint'], style=self.variables['c_s'], ksize=ksize, reuse=False, \
            name='gen_j', norm=True, verbose=verbose, joint=True)


    def __D2(self, x, ksize=3, outdim=1, reuse=False, \
        name='D2', activation='relu', norm=True, dnum=0):

        
        with tf.compat.v1.variable_scope(name, reuse=reuse):
          
          if dnum != 0:
            for i in range(dnum):
              x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
             
          
          conv1 = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
          conv1 = layers.LeakyReLU(0.2)(conv1)
          
          nf = 64
          for n in range(1, 3):
            nf_prev = nf
            nf = min(nf * 2, 512)
            conv1 = layers.Conv2D(nf, (4, 4), strides=(2, 2), padding='same')(conv1)
            conv1 = tfa.layers.InstanceNormalization(axis=-1, 
                              center=True, 
                              scale=True,
                              beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(conv1)
            conv1 = layers.LeakyReLU(0.2)(conv1)

          ng_prev = nf
          nf = min(nf * 2, 512)
          conv1 = layers.Conv2D(nf, (4, 4), strides=(1, 1), padding='same')(conv1)
          conv1 = layers.ZeroPadding2D()(conv1)
          conv1 = tfa.layers.InstanceNormalization(axis=-1, 
                              center=True, 
                              scale=True,
                              beta_initializer="random_uniform",
                              gamma_initializer="random_uniform")(conv1)
          conv1 = layers.LeakyReLU(0.2)(conv1)

          conv1 = layers.Conv2D(1, (4, 4), strides=(1, 1), padding='same')(conv1)
          conv1 = layers.ZeroPadding2D()(conv1)

          conv1 = tf.keras.activations.sigmoid(conv1)

          return conv1

    def MLP(self, style, scope='MLP'):
        channel = 512
        with tf.compat.v1.variable_scope(scope) :
            x = style

            for i in range(2):
                x = self.layer.fc(x, channel, scope='FC_' + str(i))
                x = tf.compat.v1.nn.relu(x)

            mu_list = []
            var_list = []

            for i in range(8):
                mu = self.layer.fc(x, channel, scope='FC_mu_' + str(i))
                var = self.layer.fc(x, channel, scope='FC_var_' + str(i))

                mu = tf.compat.v1.reshape(mu, shape=[-1, 1, 1, channel])
                var = tf.compat.v1.reshape(var, shape=[-1, 1, 1, channel])

                mu_list.append(mu)
                var_list.append(var)

            return mu_list, var_list

    def adain(self, content, gamma, beta, epsilon=1e-5):
        # gamma, beta = style_mean, style_std from MLP

        c_mean, c_var = tf.compat.v1.nn.moments(content, axes=[1, 2], keep_dims=True)
        c_std = tf.sqrt(c_var + epsilon)

        return gamma * ((content - c_mean) / c_std) + beta



    def __encoder(self, x, ksize=3, outdim=1, reuse=False, \
        name='enc', activation='relu', norm=True, depth=4, verbose=True):

        with tf.compat.v1.variable_scope(name, reuse=reuse):

            conv0_1 = self.layer.conv2d(x=x, stride=2, padding='SAME', \
                    filter_size=[7, 7, self.channel, 64], batch_norm=norm, training=self.training, \
                    activation=activation, name="%s_conv%d_0" %(name, 1), verbose=verbose, norm_type="IN")
            conv0_2 = self.layer.conv2d(x=conv0_1, stride=2, padding='SAME', \
                    filter_size=[4, 4, 64, 64], batch_norm=norm, training=self.training, \
                    activation=activation, name="%s_conv%d_0" %(name, 2), verbose=verbose, norm_type="IN")
            conv0_3 = self.layer.conv2d(x=conv0_2, stride=2, padding='SAME', \
                    filter_size=[4, 4, 64, 128], batch_norm=norm, training=self.training, \
                    activation=activation, name="%s_conv%d_0" %(name, 3), verbose=verbose, norm_type="IN")
            conv0_4 = self.layer.conv2d(x=conv0_3, stride=2, padding='SAME', \
                    filter_size=[4, 4, 128, 256], batch_norm=norm, training=self.training, \
                    activation=activation, name="%s_conv%d_0" %(name, 4), verbose=verbose, norm_type="IN")
            conv0_5 = self.layer.conv2d(x=conv0_4, stride=2, padding='SAME', \
                    filter_size=[4, 4, 256, 512], batch_norm=norm, training=self.training, \
                    activation=activation, name="%s_conv%d_0" %(name, 5), verbose=verbose, norm_type="IN")
            print('----------------------')
            c_in, c_out = 512, 512
            x = conv0_5
            ret_lst = []
            ret_lst.append(conv0_5)
            for idx_d in range(depth):
                conv1 = self.layer.conv2d(x=x, stride=1, padding='SAME', \
                    filter_size=[ksize, ksize, c_in, c_out], batch_norm=norm, training=self.training, \
                    activation=activation, name="%s_conv%d_1" %(name, idx_d), verbose=verbose, norm_type="IN")
                conv2 = self.layer.conv2d(x=conv1, stride=1, padding='SAME', \
                    filter_size=[ksize, ksize, c_out, c_out], batch_norm=norm, training=self.training, \
                    activation=None, name="%s_conv%d_2" %(name, idx_d), verbose=verbose, norm_type="IN")
                x = conv2
                ret_lst.append(x)

            if outdim == 1:
                rs = tf.compat.v1.reshape(x, shape=[self.batch_size, int(7*7*c_out)], \
                    name="%s_rs" %(name))
                e = self.layer.fully_connected(x=rs, c_out=outdim, \
                    batch_norm=False, training=self.training, \
                    activation=None, name="%s_fc1" %(name), verbose=verbose)
                return e

            return x, ret_lst

    def __decoder(self, z, style=None, skip=None, ksize=3, reuse=False, \
        name='dec', activation='relu', norm=True, depth=4, verbose=True, joint=False):

        with tf.compat.v1.variable_scope(name, reuse=reuse):
            print('-------decoder--------')

            c_in = 1024 if joint else 512
            # c_in = 512
            c_out = 512
            if joint:
              mu, var = self.MLP(style)
            # if joint:
            #   rs = tf.compat.v1.reshape(z, shape=[self.batch_size, int(7*7*c_out)], \
            #         name="%s_rs" %(name))
            #   e = self.layer.fully_connected(x=rs, c_out=2, \
            #       batch_norm=False, training=self.training, \
            #       activation=None, name="%s_fc1" %(name), verbose=verbose)

            #   fc1 = self.layer.fully_connected(x=e, c_out=int(7*7*c_in), \
            #       batch_norm=norm, training=self.training, \
            #       activation=None, name="%s_fc2" %(name), verbose=verbose)
            #   rs = tf.compat.v1.reshape(fc1, shape=[self.batch_size, 7, 7, c_in], \
            #       name="%s_rs2" %(name))

            #   x = rs
            # else:
            
            x = z

            # if not joint and skip is not None:
            #   x = skip + x

            for idx_d in range(depth):
                idx = 2 * idx_d

                conv1 = self.layer.conv2d(x=x, stride=1, padding='SAME', \
                    filter_size=[ksize, ksize, c_in, c_out], batch_norm=norm, training=self.training, \
                    activation=activation, name="%s_conv%d_1" %(name, idx_d), verbose=verbose, norm_type="LN")
                if joint: 
                  conv1 = self.adain(conv1, mu[idx], var[idx])
                if c_in == 1024:
                  c_in = 512
                conv2 = self.layer.conv2d(x=conv1, stride=1, padding='SAME', \
                    filter_size=[ksize, ksize, c_out, c_out], batch_norm=norm, training=self.training, \
                    activation=None, name="%s_conv%d_2" %(name, idx_d), verbose=verbose, norm_type="LN")
                if joint:
                  conv2 = self.adain(conv2, mu[idx+1], var[idx+1])
                x = conv2
            print('----------------------')
            


            convt1 = tf.keras.layers.UpSampling2D(size=(2,2), data_format="channels_last")(x)
            print(f"convt1{convt1.shape}")
            conv1 = tf.keras.layers.Conv2D(512, (5,5), strides=(1, 1), activation=activation, padding='same')(convt1)
            conv1 = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])(conv1)

            convt2 = tf.keras.layers.UpSampling2D(size=(2,2), data_format="channels_last")(conv1)
            print(f"convt2{convt2.shape}")
            conv2 = tf.keras.layers.Conv2D(256, (5,5), strides=(1, 1), activation=activation, padding='same')(convt2)
            conv2 = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])(conv2)

            convt3 = tf.keras.layers.UpSampling2D(size=(2,2), data_format="channels_last")(conv2)
            print(f"convt3{convt3.shape}")
            conv3 = tf.keras.layers.Conv2D(128, (5,5), strides=(1, 1), activation=activation, padding='same')(convt3)
            conv3 = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])(conv3)

            convt4 = tf.keras.layers.UpSampling2D(size=(2,2), data_format="channels_last")(conv3)
            print(f"convt4{convt4.shape}")
            conv4 = tf.keras.layers.Conv2D(64, (5,5), strides=(1, 1), activation=activation, padding='same')(convt4)
            conv4 = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])(conv4)


            convt5 = tf.keras.layers.UpSampling2D(size=(2,2), data_format="channels_last")(conv4)
            print(f"convt5{convt5.shape}")

            conv5 = tf.keras.layers.Conv2D(1, (5,5), strides=(1, 1), activation='tanh', padding='same')(convt5)
 
            print('flow end')
            return conv5
