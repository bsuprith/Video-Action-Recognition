"""
A collection of models we'll use to attempt to classify videos.
"""
import tensorflow as tf
from collections import deque
import sys

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048):
        """
        `model` = lrcn
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        load_model = tf.keras.models.load_model
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)        
        elif model == 'lrcn':
            print("Loading CNN-LSTM model.")
            self.input_shape = (seq_length, 80, 80, 3)
            self.model = self.lrcn()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = tf.keras.optimizers.Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

        print(self.model.summary())

    def lrcn(self):
        """Build a CNN into RNN.
        """
        def add_default_block(model, kernel_filters, init, reg_lambda):

            # conv
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(kernel_filters, (3, 3), padding='same',
                                             kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(l=reg_lambda))))
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('relu')))
            # conv
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(kernel_filters, (3, 3), padding='same',
                                             kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(l=reg_lambda))))
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('relu')))
            # max pool
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))))

            return model

        initialiser = 'glorot_uniform'
        reg_lambda  = 0.001

        model = tf.keras.models.Sequential()

        # first (non-default) block
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (7, 7), strides=(2, 2), padding='same',
                                         kernel_initializer=initialiser, kernel_regularizer=tf.keras.regularizers.l2(l=reg_lambda)),
                                  input_shape=self.input_shape))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('relu')))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3,3), kernel_initializer=initialiser, kernel_regularizer=tf.keras.regularizers.l2(l=reg_lambda))))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('relu')))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))))

        # 2nd-5th (default) blocks
        model = add_default_block(model, 64,  init=initialiser, reg_lambda=reg_lambda)
        model = add_default_block(model, 128, init=initialiser, reg_lambda=reg_lambda)
        model = add_default_block(model, 256, init=initialiser, reg_lambda=reg_lambda)
        model = add_default_block(model, 512, init=initialiser, reg_lambda=reg_lambda)

        # LSTM output head
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
        model.add(tf.keras.layers.LSTM(256, return_sequences=False, dropout=0.5))
        model.add(tf.keras.layers.Dense(self.nb_classes, activation='softmax'))

        return model
