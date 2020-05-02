"""
Class for managing our data.
"""
import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
#from processor import process_image
from keras.utils import to_categorical
import cv2
import skimage.io
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from pathlib import Path


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():

    def __init__(self, seq_length=40, class_limit=None, image_shape=(224, 224, 3)):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join(__location__,'data', 'sequences')
        self.max_frames = 300  # max number of frames a video can have for us to use it
        self.fps = 30  # video fps

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning.
        self.data = self.clean_data()

        self.image_shape = image_shape

    @staticmethod
    def get_data():
        """Load our data from file."""
        with open(os.path.join(__location__,'data', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)
        
        data = [item for item in data if len(item) > 0]
        return data

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if len(item) == 0:
                continue
            elif item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert len(label_hot) == len(self.classes)

        return label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences_in_memory(self, train_test, data_type):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Loading %d samples into memory for %sing." % (len(data), train_test))

        X, y = [], []
        for row in data:

            if data_type == 'images':
                frames = self.get_frames_for_sample(row)
                frames = self.rescale_list(frames, self.seq_length)

                # Build the image sequence
                sequence = self.build_image_sequence(frames)

            else:
                sequence = self.get_extracted_sequence(data_type, row)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    raise

            X.append(sequence)
            y.append(self.get_class_one_hot(row[1]))

        return np.array(X), np.array(y)

    @threadsafe_generator
    def frame_generator(self, batch_size, train_test, data_type):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample = random.choice(data)

                # Check to see if we've already saved this sequence.
                if data_type is "images":
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames, self.seq_length)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)
                else:
                    # Get the sequence from disk.
                    sequence = self.get_extracted_sequence(data_type, sample)

                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")

                X.append(sequence)
                y.append(self.get_class_one_hot(sample[1]))

            yield np.array(X), np.array(y)

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [self.process_image(x, self.image_shape) for x in frames]

    @staticmethod
    def process_image(frame, shape):
        """ img = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
        downsampled_img = cv2.resize(img, (shape[0], shape[1]), interpolation=cv2.INTER_CUBIC) """
        img = skimage.io.imread(frame)
        downsampled_img = resize(img, (shape[0], shape[1]), anti_aliasing=True)
        #img_array = np.ndarray.flatten(downsampled_img)
        return downsampled_img
    
    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) + \
            '-' + data_type + '.npy')
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_frames_by_filename(self, filename, data_type):
        """Given a filename for one of our samples, return the data
        the model needs to make predictions."""
        # First, find the sample row.
        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError("Couldn't find sample: %s" % filename)

        if data_type == "images":
            # Get and resample frames.
            """ frames = self.get_frames_for_sample(sample) """
            frames = self.get_frames_for_video(sample)
            frames = self.rescale_list(frames, self.seq_length)
            # Build the image sequence
            sequence = self.build_image_sequence(frames)
        else:
            # Get the sequence from disk.
            sequence = self.get_extracted_sequence(data_type, sample)

            if sequence is None:
                raise ValueError("Can't find sequence. Did you generate them?")

        return sequence

    def get_frames_newvid(self, filename):        
        
        # Get and resample frames.            
        self.generate_images(filename)
        frames = self.get_frames_for_video(filename)
        return frames, self.fps

    @staticmethod
    def generate_images(vidfile):
        src = os.path.join(__location__, vidfile)
        dest = os.path.join(__location__, 'Images', vidfile)
        Path(dest).mkdir(parents=True, exist_ok=True)

        vidcap = cv2.VideoCapture(vidfile + '.mp4')

        # Calculate fps of video
        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        if int(major_ver)  < 3 :
            fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
        else :
            fps = vidcap.get(cv2.CAP_PROP_FPS)
        
        
        count = 0
        while True:
            success,image = vidcap.read()
            if success:
                writefile = os.path.join(dest,"{}-%05d.jpg".format(vidfile) % (count+1))
                if os.path.exists(writefile):
                    continue
                else:
                    image = cv2.convertScaleAbs(image, alpha=((255.0 - np.amin(image))/(np.amax(image) - np.amin(image) + 1e-8)))
                    cv2.imwrite(writefile, image)     # save frame as JPEG file      
                    print('Read a new frame: ', success)
                    count += 1
            else:
                break
        vidcap.release()
    
    @staticmethod
    def get_frames_for_video(vidfile):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = os.path.join(__location__, 'Images', vidfile)
        images = sorted(glob.glob(os.path.join(path, vidfile + '*.jpg')))
        return images
    
    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = os.path.join(os.getcwd(),'STAIR_Actions_v1.1', sample[0], sample[1])
        filename = sample[2]
        images = sorted(glob.glob(os.path.join(path, filename + '*.jpg')))
        return images

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split(os.path.sep)
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        """ assert len(input_list) >= size """
        if len(input_list) < size:
            return None

        else:
            # Get the number to skip between iterations.
            skip = len(input_list) // size

            # Build our new output.
            output = [input_list[i] for i in range(0, len(input_list), skip)]

            # Cut off the last one if needed.
            return output[:size]

    def print_class_from_prediction(self, predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        # Get the prediction for each label.
        label_predictions = {}
        for i, label in enumerate(self.classes):
            label_predictions[label] = predictions[i]

        # Now sort them.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # And return the top N.        
        return sorted_lps[:nb_to_return]
