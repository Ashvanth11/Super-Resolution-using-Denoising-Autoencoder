# Super-Resolution-using-Denoising-Autoencoder
A TensorFlow based implementation of **Image Super-Resolution** via **Denoising Autoencoder**

Images were added with Gaussian noise and were sent into a Deep Convolutional Autoencoder which denoises the image and reconstructs it to a higher resolution.

## Dataset Used
The model was trained using **DIV2K dataset**

DIV2K dataset has the following structure:

1000 2K resolution images divided into: 800 images for training, 100 images for validation, 100 images for testing

## Training History
![plot](https://user-images.githubusercontent.com/85792473/200181943-4a9ffde0-053c-4d47-b853-2ce3bd243e43.png)


## Sample Output
input-noisy-output
![input-noisy-output](https://user-images.githubusercontent.com/85792473/200181970-bb581bf0-affb-4b78-864c-33b70b4e712e.png)

