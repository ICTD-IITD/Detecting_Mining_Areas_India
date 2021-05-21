This repo contains the code and data for detection of surface mining activity

# Data
Data folder contains only the polygon shapes of mines. Satellite images of mining sites can be downlaoded using the data_download script provided. This script was made on Google Colab, thus the code here assumes the notebook is connected to Google Drive.

Paper from which data is taken: https://www.nature.com/articles/s41597-020-00624-w


# Data Conversion

The data downlaoded from data_download scripy is in .tif format. Currently, no libraries are available on HPC which can read .tif files, thus before training the segmentation model .tif files are converted to .npy format. __conversion.py__ can be used for this purpose.

# Unet Model

Unet is an image segmentation model which does a pixel level classification of image.
Original paper can be found here: https://arxiv.org/abs/1505.04597

unet.py script describes couple of UNet segmentation models. The script though is targeted to work on segmenting satellite images, it can be used for any type of segmentation.

Input:
Satellite images are taken in "npy" format and the corresponding masks are taken in ".png" format by default (can be changed as per requirement)
No arguments are taken while running the scripts.
Path to the folder containing satellite images and masks are needed to be set in the script only.

Defined functions in Script:
augment_data():
	randomly changing the orientation of input images to increase training images.

Convolution_layers: - down_block, up_block, bottleneck
	for down samping and upsampling.

Performance calculating functions: - jaccard_distance, weighted_binary_crossentropy, iou_score, mean_iou
	for calculation of accuracy and loss incured by the trained model.

Unet():
	Unet architecture is implemented for 7 down and 7 up layers to train the model. 
	Originally described here: https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow


Unetv2():
	Unet architecture inspired by https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/.

Model described in Unet() seems to work better than the one in Unetv2() 

Output of Script:
	Saves the best model(default: best in terms of lowest val_loss) in ".h5" format.
