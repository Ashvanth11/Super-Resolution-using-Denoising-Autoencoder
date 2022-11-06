# Super-Resolution-using-Denoising-Autoencoder
A TensorFlow based implementation of **Image Super-Resolution** via **Denoising Autoencoder**

Images were added with Gaussian noise and were sent into a Deep Convolutional Autoencoder which denoises the image and reconstructs it to a higher resolution.

## Dataset Used
The model was trained using **DIV2K dataset**

DIV2K dataset has the following structure:

1000 2K resolution images divided into: 800 images for training, 100 images for validation, 100 images for testing
