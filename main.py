"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from models import ResearchModels
from data import DataSet
import time
import os.path
import pdb
import matplotlib.pyplot as plt

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100):
    # Helper: Save the model.
    callbacks = [
        ReduceLROnPlateau(verbose=1, monitor='val_acc', mode='max'),
        ModelCheckpoint(
            'chkp\\weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            verbose=1, save_best_only=True)
    ]

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size
    
    # Get generators.
    generator = data.frame_generator(batch_size, 'train', data_type)
    val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Fit!    
    history = rm.model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=callbacks,
        validation_data=val_generator,
        validation_steps=40,
        workers=4)

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_accuracy'])
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

def main():
    """These are the main training settings. Set each before running
    this file."""
    
    model = 'lrcn'
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 5
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 8
    nb_epoch = 300

    # Chose images or features and image shape based on network.
    data_type = 'images'
    image_shape = (80, 80, 3)

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()