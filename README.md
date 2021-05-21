This repo contains the code and data for detection of surface mining activity

# Data
Data folder contains only the polygon shapes of mines. Satellite images of mining sites can be downlaoded using the data_download script provided. This script was made on Google Colab, thus the code here assumes the notebook is connected to Google Drive.

Paper from which data is taken: https://www.nature.com/articles/s41597-020-00624-w


# Data Conversion

The data downlaoded from data_download scripy is in .tif format. Currently, no libraries are available on HPC which can read .tif files, thus before training the segmentation model .tif files are converted to .npy format. __conversion.py__ can be used for this purpose.

# Unet Model

===================================================================================================================

Final Script of Model:

This python script is a machine learning program which learns to detect mining site present on a satallite image using sample images and predicts mining site present on any satallite image.

Input:
It takes satallite images in tif formate. It takes the corresponding masks of each image to verify and validate its prediction.

We are giving 170 satallite images and their corresponding masks as input further we are augmenting it internally to increase the number of images for learning.

Defined functions in Script:
augment_data():
	randomly changing the orientation of input images to increase training images.

Convolution_layers: - down_block, upblock bottleneck
	for down samping and upsampling.

Performance calculating functions: - jaccard_distance, weighted_binary_crossentropy, iou_score, mean_iou
	for calculation of accuracy and loss incured by the trained model.

Unet():
	Unet architecture is implemented for 7 down and 7 up layers to train the model.

Unetv2():
	Unet architecture is implemented for higher number of down and up layers for alternate approach. 


Working of Script:
Initially, libraries are imported and training and validation images are loaded in different lists in numpy array formate.

Masks are also loaded in different list each for training and validation dateset in numpy array formate.

Next, defined Unet() function is assigned to a variable named model and then is complied with following parameters:

optimizer="Adamax", loss=['binary_crossentropy'], metrics=[iou_score,mean_iou]

Checkpoint is created and stored in a directory for saving learning model at different stages to counter any exceptional failure in the middle of the execution in future.

Next, model is trained on the list of training images and their corresponding masks in batch size of 8 and for 600 epochs.

At last from saved checkpoints best model is retrieved and evaluated for testing the accuracy of the final model.

Output of Script:
	It will give us a best model trained on input images which can be used for prediction of mining areas in a satallite images. This model will predict the mining area with accuracy shown by the best model saved in the checkpoint.
