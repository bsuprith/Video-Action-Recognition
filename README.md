# Video-Action-Recognition

The file main.py contains the code for training the network to perform the task of recognizing the task of "reading a book" in videos.
The folder "videos" has to contain the videos from the dataset divided into 2 categories : "reading_book" and "nonreading".
The batch size and number of frames can be specified as arguments to the Video Frame Generator framework. The trained model is saved in the 'chkp' folder, with the latest model weights saved in 'model3.hdf5'.

The file test.py can be used for testing any new video that outputs a JSON file with the format {"timestamp", "required class prediction accuracy"}. The video path and folder containing the saved model have to be specified.
