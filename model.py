# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from convautoencoder import ConvAutoencoder
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

from keras.utils import img_to_array
import os
from tqdm import tqdm
import re
import tensorflow as tf



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", type=int, default=8,
	help="# number of samples to visualize when decoding")
ap.add_argument("-o", "--output", type=str, default="output.png",
	help="path to output visualization file")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output plot file")
args = vars(ap.parse_args())

# initialize the number of epochs to train for and batch size
EPOCHS = 60
BS = 24
# load the dataset
print("[INFO] loading dataset...")

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)
# defining the size of the image
SIZE = 256

trainX = []
path = r"C:\Users\ashva\Desktop\Super_Resolution\Data\train"
files = os.listdir(path)
files = sorted_alphanumeric(files)
for i in tqdm(files):
    if i == '900.jpg':
        break
    else:
        img = cv2.imread(path + '/'+i,1)
        # open cv reads images in BGR format so we have to convert it to RGB

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #resizing image
        img = cv2.resize(img, (SIZE, SIZE))
        
        trainX.append(img_to_array(img))
trainX = np.array(trainX)


testX = []
path = r"C:\Users\ashva\Desktop\Super_Resolution\Data\test"
files = os.listdir(path)
files = sorted_alphanumeric(files)
for i in tqdm(files):
     if i == '100.jpg':
        break
     else:
        img = cv2.imread(path + '/'+i,1)

        #resizing image
        img = cv2.resize(img, (SIZE, SIZE))
        
        testX.append(img_to_array(img))
testX = np.array(testX)

# adds the Gaussian noise based on the mean and the standard deviation
def add_gaussian_noise(data):
    mean = (10, 10, 10)
    std = (50, 50, 50)
    row, col, channel = data.shape
    noise = np.random.normal(mean, std, (row, col, channel)).astype('uint8')
    return data + noise
def add_noise_to_dataset(data):
    count = 0
    end = len(data)
    output_data = []
    while count < end:
        output_data.append(add_gaussian_noise(data[count]))
        count+=1
    return np.array(output_data)


trainXNoisy = add_noise_to_dataset(trainX)
testXNoisy = add_noise_to_dataset(testX)


#img = img.astype('float32') / 255.0
testX = testX / 255
trainX = trainX / 255
trainXNoisy = trainXNoisy / 255
testXNoisy = testXNoisy / 255


gaussian_auto_encoder = ConvAutoencoder()
gaussian_history = gaussian_auto_encoder.fit(trainXNoisy, trainX, validation_split=0.1, epochs=EPOCHS, batch_size=BS)
gaussian_auto_encoder.save("savemodel")

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, gaussian_history.history["loss"], label="train_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# use the convolutional autoencoder to make predictions on the
# testing images, then initialize our list of output images
print("[INFO] making predictions...")

decoded = gaussian_auto_encoder.predict(testXNoisy)
outputs = None
# loop over our number of output samples
for i in range(0, args["samples"]):
	# grab the original image and reconstructed image

 original = (testX[i] * 255).astype("uint8")
 noisy = (add_gaussian_noise(testX[i] * 255)).astype("uint8")
 recon = (decoded[i] * 255).astype("uint8")
	# stack the original and reconstructed image side-by-side
 output = np.hstack([original, noisy, recon])
	# if the outputs array is empty, initialize it as the current
	# side-by-side image display
 if outputs is None:
		outputs = output
	# otherwise, vertically stack the outputs
 else:
	    outputs = np.vstack([outputs, output])
# save the outputs image to disk
cv2.imwrite(args["output"], outputs)
