
import keras
from glob import glob
import os
import skvideo.io
import numpy as np
from keras.models import load_model
import ntpath, json

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

videoPath = "test_videos\\y_DxQaexwtw.mp4"
fileName = os.path.splitext(path_leaf(videoPath))[0]
 
# load model
model = load_model('chkp\\weights.20-0.75.hdf5')
# summarize model.
model.summary()

testFrames = skvideo.io.vread(videoPath, height = 112, width = 112)

i = 0
fileJson = {}
fileJson[fileName] = []
while(i<testFrames.shape[0]):
  test = testFrames[i:i+10,:,:,:]
  test = np.expand_dims(test, axis = 0)
  predict = model.predict(test)
  fileJson[fileName].append([str(i),str([predict[0][0],predict[0][1]]) ] )
  i += 10

with open("Test\\" + fileName + ".json", "w") as outfile: 
    json.dump(fileJson, outfile)