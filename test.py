"""
Given a video path and a saved model (checkpoint), produce classification
predictions.

"""
from keras.models import load_model
from data import DataSet
import numpy as np
import os, json
import matplotlib.pyplot as plt

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def predict(data_type, seq_length, saved_model, image_shape, video_name, class_limit):
    model = load_model(saved_model)

    temporal_predictions = []
    temporal_labels = []
    temporal_prob = []
    fileJson = {}

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(seq_length=seq_length, class_limit=class_limit)
    else:
        data = DataSet(seq_length=seq_length, image_shape=image_shape,
            class_limit=class_limit)
    
    # Extract the sample from the data.
    vid_frames, vid_fps = data.get_frames_newvid(video_name)
    
    for i in np.arange(0, len(vid_frames), vid_fps):
        rescaled_list = data.rescale_list(vid_frames[i:np.amin([i+vid_fps-1, len(vid_frames)])], seq_length)
        if rescaled_list is None:
            continue
        # Generate the sequence
        sample = data.build_image_sequence(rescaled_list)
        prediction = model.predict(np.expand_dims(sample, axis=0))
        class_prediction = data.print_class_from_prediction(np.squeeze(prediction, axis=0), 1)
        # Store the top prediction
        temporal_predictions.append(class_prediction[0])

    fileJson[video_name] = temporal_predictions
    with open(video_name + ".json", "w") as outfile: 
        json.dump(str(fileJson), outfile)
    
    for tmp in np.arange(len(temporal_predictions)):
        temporal_labels.append(temporal_predictions[tmp][0])
        temporal_prob.append(temporal_predictions[tmp][1])

    
    # Plot of predicted labels with time
    
    plt.clf()
    xs = np.arange(0,len(temporal_labels),1)
    ys = temporal_prob
    plt.plot(xs,ys,'bo-')
        
    plt.title('Video action predictions')
    plt.ylabel('Probability')
    plt.xlabel('Time')
    for x,y in zip(xs,ys):
        label = "{}".format(temporal_labels[x])
        plt.annotate(label,
                    (x,y),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')

    plt.savefig('{}.png'.format(video_name+'_predictions'))
    plt.close()

def main():    
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d.
    model = 'lrcn'
    # Must be a weights file.
    saved_model = os.path.join(__location__, 'chkp\\weights.155-2.21.hdf5')
    # Sequence length must match the lengh used during training.
    seq_length = 5
    # Limit must match that used during training.
    class_limit = None

    # Demo file.
    # Do not include the extension.
    # Assumes it's in data/[train|test]/
    # It takes in the path to
    # an actual video file, extract frames, generate sequences, etc.
    video_name = 'a063-0542C'

    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    predict(data_type, seq_length, saved_model, image_shape, video_name, class_limit)

if __name__ == '__main__':
    main()