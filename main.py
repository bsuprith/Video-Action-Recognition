import os
import glob
import keras
from keras_video import VideoFrameGenerator
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D, \
    TimeDistributed, GRU, Dense, Dropout, LSTM
import tensorflow as tf
import matplotlib.pyplot as plt


SIZE = (112, 112)
CHANNELS = 3
NUM_FRAMES = 10
BATCH_SIZE = 30

# class names
classes = [i.split(os.path.sep)[1] for i in glob.glob('videos\\*')]
classes.sort()

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

""" test = VideoFrameGenerator(
    classes=classes, 
    glob_pattern=test_pattern,
    nb_frames=NUM_FRAMES,         
    shuffle=True,
    batch_size=BATCH_SIZE,
    target_shape=SIZE,
    nb_channel=CHANNELS,    
    use_frame_cache=True) """


# Network
def build_convnet(shape=(112, 112, 3)):
    momentum = .9
    model = keras.Sequential()
    model.add(Conv2D(64, (3,3), input_shape=shape,
        padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    # flatten...
    model.add(GlobalMaxPool2D())
    return model

def action_model(shape=(5, 112, 112, 3), nbout=3):
    # (112, 112, 3) input shape
    convnet = build_convnet(shape[1:])    
    
    # then create our final model
    model = keras.Sequential()
    # add the convnet with (5, 112, 112, 3) shape
    model.add(TimeDistributed(convnet, input_shape=shape))
    # here, you can also use GRU or LSTM
    model.add(GRU(64))
    """ model.add(LSTM(64)) """
    # and finally, we make a decision network
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    return model

if __name__ == "__main__":
    INSHAPE=(NUM_FRAMES,) + SIZE + (CHANNELS,) # (5, 112, 112, 3)
        
    # GPU usage options
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    model = action_model(INSHAPE, len(classes))
    optimizer = keras.optimizers.Adam(0.001)
    model.compile(
        optimizer,
        'categorical_crossentropy',
        metrics=['acc']
    )

    EPOCHS = 20
    # saving checkpoints
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(verbose=1),
        keras.callbacks.ModelCheckpoint(
            'chkp\\weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            verbose=1),
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