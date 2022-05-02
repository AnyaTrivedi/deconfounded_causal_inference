%tensorflow_version 1.x
import tensorflow as tf
import numpy as np 
import pandas as pd
import numpy.random as npr
from scipy import sparse
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed


class PPCA():
    def __init__(df, latent_dim, form):
        self.exposureDf = df.copy()
        self.exposureDf['rating'] = self.exposureDf['rating'].where(self.exposureDf['rating'] == 0, 1)
        self.nusers = self.exposureDf['userId'].nunique()
        self.nitems = self.exposureDf['itemId'].nunique()

        self.a_matrix = sparse.coo_matrix((self.exposureDf["rating"],(self.exposureDf["userId"],self.exposureDf["itemId"])),shape=(self.nusers,self.nitems))
        self.a_matrix = self.a_matrix.todense()
        self.latent_dim = latent_dim
        self.form = form


    def GetRowFactors(self):
        latent_dim =  self.latent_dim
        stddv_datapoints = 0.1
        num_datapoints, data_dim = self.a_matrix.shape
        a_matrix =  self.a_matrix

        # we allow both linear and quadratic model
        # for linear model x_n has mean z_n * W
        # for quadratic model x_n has mean b + z_n * W + (z_n**2) * W_2
        # quadractice model needs to change the checking step accordingly

        def ppca_model(data_dim, latent_dim, num_datapoints, stddv_datapoints, self.form):
            w = ed.Normal(loc=tf.zeros([latent_dim, data_dim]),
                        scale=tf.ones([latent_dim, data_dim]),
                        name="w")  # parameter
            z = ed.Normal(loc=tf.zeros([num_datapoints, latent_dim]),
                        scale=tf.ones([num_datapoints, latent_dim]), 
                        name="z")  # local latent variable / substitute confounder
            if self.form == "linear":
        #          x = ed.Normal(loc=tf.multiply(tf.matmul(z, w), a_matrix),
                x = ed.Normal(loc=tf.matmul(z, w),
                            scale=stddv_datapoints * tf.ones([num_datapoints, data_dim]),
                            name="x")  # (modeled) data
            elif self.form == "quadratic":
                b = ed.Normal(loc=tf.zeros([1, data_dim]),
                        scale=tf.ones([1, data_dim]),
                        name="b")  # intercept
                w2 = ed.Normal(loc=tf.zeros([latent_dim, data_dim]),
                        scale=tf.ones([latent_dim, data_dim]),
                        name="w2")  # quadratic parameter
        #          x = ed.Normal(loc=tf.multiply(b + tf.matmul(z, w) + tf.matmul(tf.square(z), w2), a_matrix),
                x = ed.Normal(loc=b + tf.matmul(z, w) + tf.matmul(tf.square(z), w2),                        
                            scale=stddv_datapoints * tf.ones([num_datapoints, data_dim]),
                            name="x")  # (modeled) data
            return x, (w, z)

        log_joint = ed.make_log_joint_fn(ppca_model)


        def variational_model(qb_mean, qb_stddv, qw_mean, qw_stddv, 
                            qw2_mean, qw2_stddv, qz_mean, qz_stddv):
            qb = ed.Normal(loc=qb_mean, scale=qb_stddv, name="qb")
            qw = ed.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
            qw2 = ed.Normal(loc=qw2_mean, scale=qw2_stddv, name="qw2")
            qz = ed.Normal(loc=qz_mean, scale=qz_stddv, name="qz")
            return qb, qw, qw2, qz


        log_q = ed.make_log_joint_fn(variational_model)

        def target(b, w, w2, z):
            """Unnormalized target density as a function of the parameters."""
            return log_joint(data_dim=data_dim,
                            latent_dim=latent_dim,
                            num_datapoints=num_datapoints,
                            stddv_datapoints=stddv_datapoints,
                            w=w, z=z, w2=w2, b=b, x=a_matrix)

        def target_q(qb, qw, qw2, qz):
            return log_q(qb_mean=qb_mean, qb_stddv=qb_stddv,
                        qw_mean=qw_mean, qw_stddv=qw_stddv,
                        qw2_mean=qw2_mean, qw2_stddv=qw2_stddv,
                        qz_mean=qz_mean, qz_stddv=qz_stddv,
                        qw=qw, qz=qz, qw2=qw2, qb=qb)

        qb_mean = tf.Variable(np.ones([1, data_dim]), dtype=tf.float32)
        qw_mean = tf.Variable(np.ones([latent_dim, data_dim]), dtype=tf.float32)
        qw2_mean = tf.Variable(np.ones([latent_dim, data_dim]), dtype=tf.float32)
        qz_mean = tf.Variable(np.ones([num_datapoints, latent_dim]), dtype=tf.float32)
        qb_stddv = tf.nn.softplus(tf.Variable(0 * np.ones([1, data_dim]), dtype=tf.float32))
        qw_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([latent_dim, data_dim]), dtype=tf.float32))
        qw2_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([latent_dim, data_dim]), dtype=tf.float32))
        qz_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([num_datapoints, latent_dim]), dtype=tf.float32))

        qb, qw, qw2, qz = variational_model(qb_mean=qb_mean, qb_stddv=qb_stddv,
                                            qw_mean=qw_mean, qw_stddv=qw_stddv,
                                            qw2_mean=qw2_mean, qw2_stddv=qw2_stddv,
                                            qz_mean=qz_mean, qz_stddv=qz_stddv)


        energy = target(qb, qw, qw2, qz)
        entropy = -target_q(qb, qw, qw2, qz)

        elbo = energy + entropy


        optimizer = tf.train.AdamOptimizer(learning_rate = 0.05)
        train = optimizer.minimize(-elbo)

        init = tf.global_variables_initializer()

        t = []

        num_epochs = 500

        with tf.Session() as sess:
            sess.run(init)

            for i in range(num_epochs):
                sess.run(train)
                if i % 5 == 0:
                    t.append(sess.run([elbo]))

                b_mean_inferred = sess.run(qb_mean)
                b_stddv_inferred = sess.run(qb_stddv)
                w_mean_inferred = sess.run(qw_mean)
                w_stddv_inferred = sess.run(qw_stddv)
                w2_mean_inferred = sess.run(qw2_mean)
                w2_stddv_inferred = sess.run(qw2_stddv)
                z_mean_inferred = sess.run(qz_mean)
                z_stddv_inferred = sess.run(qz_stddv)
                
        return z_mean_inferred, z_stddv_inferred


