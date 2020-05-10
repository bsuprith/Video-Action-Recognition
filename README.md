# Video-Action-Recognition

The file main.py contains the code for training the network to perform the task of recognizing the task of "reading a book" in videos.
The STAIR-Actions dataset needs to be downloaded to a folder called 'STAIR_Actions_v1.1'. The videos need to be separated into 'train' and 'test' categories by running the file 'move_files.py'. Images need to be extracted for the videos of the respective categories in the dataset by running 'extract_files.py'.
The batch size and number of frames can be specified as arguments to the Video Frame Generator framework. The trained model is saved in the 'chkp' folder, with the latest model weights saved in 'model8.hdf5'.

The file test.py can be used for testing any new video that outputs a JSON file with the format {"predicted action", "probability of predicted class"} for each second of the test video. The video name has to be specified in the file 'test.py'.
