import os
import glob
import keras
from keras_video import VideoFrameGenerator
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D, \
    TimeDistributed, GRU, Dense, Dropout, LSTM, Activation, Flatten
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam, RMSprop
from keras import regularizers


SIZE = (80, 80)
CHANNELS = 3
NUM_FRAMES = 30
BATCH_SIZE = 8

# class names
classes = [i.split(os.path.sep)[1] for i in glob.glob('videos\\*')]
classes.sort()
nb_classes = len(classes)

# video pattern
glob_pattern = 'videos\\{classname}\\*.mp4'

# for data augmentation
data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)

# Create video frame generator
train = VideoFrameGenerator(
    classes=classes, 
    glob_pattern=glob_pattern,
    nb_frames=NUM_FRAMES,
    split=.33, 
    shuffle=True,
    batch_size=BATCH_SIZE,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    transformation=data_aug,
    use_frame_cache=True)

valid = train.get_validation_generator()

def add_default_block(model, kernel_filters, init, reg_lambda):

    # conv
    model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                        kernel_initializer=init, kernel_regularizer=regularizers.l2(reg_lambda))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    # conv
    model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3), padding='same',
                                        kernel_initializer=init, kernel_regularizer=regularizers.l2(reg_lambda))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    # max pool
    model.add(TimeDistributed(MaxPool2D((2, 2), strides=(2, 2))))

    return model

if __name__ == "__main__":
    INSHAPE=(NUM_FRAMES,) + SIZE + (CHANNELS,)
        
    # GPU usage options
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    
    initialiser = 'glorot_uniform'
    reg_lambda  = 1e-4

    model = keras.Sequential()

    # first (non-default) block
    model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same',
                                        kernel_initializer=initialiser, kernel_regularizer=regularizers.l2(reg_lambda)),
                                input_shape=INSHAPE))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(32, (3,3), kernel_initializer=initialiser, kernel_regularizer=regularizers.l2(reg_lambda))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPool2D((2, 2), strides=(2, 2))))

    # 2nd-5th (default) blocks
    model = add_default_block(model, 64,  init=initialiser, reg_lambda=reg_lambda)
    model = add_default_block(model, 128, init=initialiser, reg_lambda=reg_lambda)
    model = add_default_block(model, 256, init=initialiser, reg_lambda=reg_lambda)
    model = add_default_block(model, 512, init=initialiser, reg_lambda=reg_lambda)

    # LSTM output head
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(
        optimizer,
        'categorical_crossentropy',
        metrics=['acc']
    )

    EPOCHS = 20
    # saving checkpoints
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(verbose=1, monitor='val_accuracy', mode='max'),
        keras.callbacks.ModelCheckpoint(
            'chkp\\weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            verbose=1, save_best_only=True),
    ]
    history = model.fit_generator(
        train,
        validation_data=valid,
        verbose=1,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.savefig('model_accuracy.png')
    plt.close()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.savefig('model_loss.png')
    plt.close()