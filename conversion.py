import rasterio
import numpy as np
from matplotlib import pyplot as plt
import glob
import os

#Resizing images is optional, CNNs are ok with large images

images_path = []
#Capture training image info as a list
directory_path = "<path where .tf files are stored>"
for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
    images_path.append(img_path)
    img = rasterio.open(img_path).read()
    img_name = os.path.basename(img_path).split('.')[0]
    img = np.rollaxis(img, 1)
    img = np.rollaxis(img, 2,1)
    np.save('<path to store .npy files>'+ img_name +'.npy',img)
    if len(images_path)%10 == 0:
        print(len(images_path))