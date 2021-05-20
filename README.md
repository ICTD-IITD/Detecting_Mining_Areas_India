This repo contains the code and data for detection of surface mining activity

# Data
Data folder contains only the polygon shapes of mines. Satellite images of mining sites can be downlaoded using the data_download script provided. This script was made on Google Colab, thus the code here assumes the notebook is connected to Google Drive.

Paper from which data is taken: https://www.nature.com/articles/s41597-020-00624-w


# Data Conversion

The data downlaoded from data_download scripy is in .tif format. Currently, no libraries are available on HPC which can read .tif files, thus before training the segmentation model .tif files are converted to .npy format. __conversion.py__ can be used for this purpose.

# Unet Model
