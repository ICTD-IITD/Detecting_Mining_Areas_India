This repo contains the code and data for detection of surface mining activity

# Data
Data folder contains only the polygon shapes of mines. Satellite images of mining sites can be downlaoded using the data_download script provided. This script was made on Google Colab, thus the code here assumes the notebook is connected to Google Drive.

grId in the geojson mentions the corresponding group the mine belongs to. For Brazil and India these group ids are mentioned in a their corresponding geojsons

The below drive link contain satellite images for India and Brazil and their corresponding masks
https://drive.google.com/drive/folders/1XRCPxPTQcRNDJgqyDkbOcPv_WLHwI_9q?usp=sharing 

Paper from which data is taken: https://www.nature.com/articles/s41597-020-00624-w


# Data Conversion

The data downlaoded from data_download script is in .tif format. Currently, no libraries are available on HPC which can read .tif files, thus before training the segmentation model .tif files are converted to .npy format. _conversion.py_ can be used for this purpose.

_conversion.py_ also changes the shape of image from *num_channels* x *size_x* x *size_y* ==> *size_x* x *size_y* x *num_channels*, to make it compatible for cv2 operations.

# Unet Model

Unet is an image segmentation model which does a pixel level classification of image.
Original paper can be found here: https://arxiv.org/abs/1505.04597

unet.py script describes couple of UNet segmentation models. The script though is targeted to work on segmenting satellite images, it can be used for any type of segmentation.

Input:
Satellite images are taken in "npy" format and the corresponding masks are taken in ".png" format by default (can be changed as per requirement)
No arguments are taken while running the scripts.
Path to the folder containing satellite images and masks are needed to be set in the script only.


Output:
Saves the best model(default: best in terms of lowest val_loss) in ".h5" format.

Given script contains 2 models of Unet:

Unet():
	This model contains 7 down and 7 up blocks and a bottleneck layer. 
	Originally described here: https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow


Unetv2():
	This Unet model was inspired by https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/.

Model described in Unet() seems to work better than the one in Unetv2() 

