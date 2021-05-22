#!/usr/bin/env python
# coding: utf-8

# # Imports

#seg environment
import os
import sys
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
from keras import backend as K
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print("====== Imported Libraries =======")

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print("GPUS -> ", get_available_gpus())


def augment_data(img,img_lis,augment_list,mode):
    """
    flip:       0:"none",   1:"vertical_flip",  2:"horizontal_flip" 
    rotation:   0:"none",   1:"counter_clockwise",      2:"rotate_180", 3:"clockwise"
    """
    assert np.all(img==img_lis[-1])
    if mode == "train":
        x = random.randint(0,2)
        y = random.randint(0,3)
        augment_list.append((x,y))
    elif mode=="val":
        x,y = augment_list.pop(0)
    else:
        assert False

    if x==1:
        img_lis.append(cv2.flip(img,0))
    elif x==2:
        img_lis.append(cv2.flip(img,1))

    if y==1:
        rot = np.rot90(img)
        img_lis.append(rot)
    elif y==2:
        rot = np.rot90(img)
        rot = np.rot90(rot)
        img_lis.append(rot)
    elif y==3:
        rot = np.rot90(img)
        rot = np.rot90(rot)
        rot = np.rot90(rot)
        img_lis.append(rot)
    return



# # Data Loading

fp = './Data/data_v3/'
print("============ fp: ",fp," ============")
x_train = []
augment_list = []
x_val = []
train_path = []
val_path = []
SIZE_Y,SIZE_X = (1024,1024)
num_channels = 12

#loading npy files
for img_path in glob.glob(os.path.join(fp+"npy/train/", "*.npy")):
    img = np.load(img_path)
    img = np.nan_to_num(img)        #convert nan to 0 in img
    img = cv2.resize(img,(SIZE_Y,SIZE_X),interpolation = cv2.INTER_NEAREST)   #resizing
    train_path.append(img_path)
    x_train.append(img)
    augment_data(img,x_train,augment_list,"train")

x_train = np.array(x_train)
print("============ Loaded Train Images ==========")
print("Images Shape: ",x_train.shape)
print("Max in train_images: ",np.max(x_train))

#loading npy files
for img_path in glob.glob(os.path.join(fp+"npy/val/", "*.npy")):
    img = np.load(img_path)
    img = np.nan_to_num(img)        #convert nan to 0 in img
    img = cv2.resize(img,(SIZE_Y,SIZE_X),interpolation = cv2.INTER_NEAREST)   #resizing
    val_path.append(img_path)
    x_val.append(img)

x_val = np.array(x_val)
print("============ Loaded Validation Images ==========")
print("Images Shape: ",x_val.shape)
print("Max in train_images: ",np.max(x_val))

print("Not normalizing the images")

y_train = []
#loading masks
for img_path in train_path:
    img_name = os.path.basename(img_path).split('.')[0]
    mask_path = fp + "Masks/" + img_name.split('.')[0] + ".png"
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X),interpolation = cv2.INTER_NEAREST)
    mask = mask.astype('float32')
    mask = mask/255.0
    y_train.append(mask)
    augment_data(mask,y_train,augment_list,"val")


y_train = np.array(y_train)
y_train = np.expand_dims(y_train, axis=3)
print("============ Loaded Train Masks ==========")
print("Masks Shape: ",y_train.shape)
print("Max in train_masks: ",np.max(y_train))


# Validation
y_val = []
#loading masks
for img_path in val_path:
    img_name = os.path.basename(img_path).split('.')[0]
    mask_path = fp + "Masks/" + img_name.split('.')[0] + ".png"
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X),interpolation = cv2.INTER_NEAREST)
    mask = mask.astype('float32')
    mask = mask/255.0
    y_val.append(mask)
y_val = np.array(y_val)
y_val = np.expand_dims(y_val, axis=3)
print("============ Loaded Validation Masks ==========")
print("Masks Shape: ",y_val.shape)
print("Max in train_masks: ",np.max(y_val))



def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


# # Custom Loss function

def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def weighted_binary_crossentropy(y_true, y_pred):

    # Calculate the binary crossentropy
    b_ce = K.binary_crossentropy(y_true, y_pred)
    
    # print(b_ce)
    
    # Apply the weights
    weight_vector = y_true * 10 + (1. - y_true) * 1
    weighted_b_ce = weight_vector * b_ce

    # Return the mean error
    return K.mean(weighted_b_ce)

def iou_score(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

def mean_iou(y_true, y_pred):
    y_pred_zero = K.cast(K.less_equal(y_pred,.5),dtype='float32')
    y_true_zero = K.cast(K.less_equal(y_pred,.5),dtype='float32')
    inter = K.sum(K.sum(K.squeeze(y_true_zero * y_pred_zero, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true_zero + y_pred_zero, axis=3), axis=2), axis=1) - inter
    zero_ans = K.mean((inter + K.epsilon()) / (union + K.epsilon()))
    
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    one_ans = K.mean((inter + K.epsilon()) / (union + K.epsilon()))
    return (zero_ans + one_ans)/2.

# # Unet Model

def Unet():
    f = [16, 32, 64, 128, 256, 512, 1024, 2048]
    inputs = keras.layers.Input((SIZE_Y, SIZE_X, num_channels))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) 
    c2, p2 = down_block(p1, f[1]) 
    c3, p3 = down_block(p2, f[2]) 
    c4, p4 = down_block(p3, f[3]) 
    c5, p5 = down_block(p4, f[4]) 
    c6, p6 = down_block(p5, f[5]) 
    c7, p7 = down_block(p6, f[6]) 
    
    bn = bottleneck(p7, f[7])
    
    u1 = up_block(bn, c7, f[6]) 
    u2 = up_block(u1, c6, f[5]) 
    u3 = up_block(u2, c5, f[4])
    u4 = up_block(u3, c4, f[3])
    u5 = up_block(u4, c3, f[2])
    u6 = up_block(u5, c2, f[1])
    u7 = up_block(u6, c1, f[0])
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u7)
    model = keras.models.Model(inputs, outputs)
    return model

def Unetv2():
    input_layer = keras.layers.Input((SIZE_Y, SIZE_X, 12))

    #Block1
    bl1 = keras.layers.Conv2D(64, kernel_size=(3,3), padding="same", strides=1, activation="relu")(input_layer)

    #Down-Block2
    bl2_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl1)
    bl2_2 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl2_1)
    bl2_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl2_2)
    bl2_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl2_3)   
    pl2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),padding='same')(bl2_4)
#     print("Block2: ",pl2.shape)
    
    #Down-Block3
    bl3_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(pl2)
    bl3_2 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl3_1)
    bl3_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl3_2)
    bl3_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl3_3)
    bl3_5 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl3_4)
    bl3_6 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl3_5)  
    pl3 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),padding='same')(bl3_6)
#     print("Block3: ",pl3.shape)
    
    
    #Down-Block3.5 (3_5)
    bl3_5_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(pl3)
    bl3_5_2 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl3_5_1)
    bl3_5_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl3_5_2)
    bl3_5_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl3_5_3)
    bl3_5_5 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl3_5_4)
    bl3_5_6 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl3_5_5)  
    pl3_5 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),padding='same')(bl3_5_6)
#     print("Block3.5: ",pl3_5.shape)
    
    
    #Down-Block4
    bl4_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(pl3_5)
    bl4_2 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl4_1)
    bl4_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl4_2)
    bl4_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl4_3)
    bl4_5 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl4_4)
    bl4_6 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl4_5)
    pl4 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),padding='same')(bl4_6)
#     print("Block4: ",pl4.shape)
    
    
    #Down-Block4.5 (4_5)
    bl4_5_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(pl4)
    bl4_5_2 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl4_5_1)
    bl4_5_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl4_5_2)
    bl4_5_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl4_5_3)
    bl4_5_5 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl4_5_4)
    bl4_5_6 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl4_5_5)
    pl4_5 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),padding='same')(bl4_5_6)
#     print("Block4.5: ",pl4_5.shape)
    
    
    #Down-Block5
    bl5_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(pl4_5)
    bl5_2 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl5_1)
    bl5_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl5_2)
    bl5_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl5_3)
    bl5_5 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl5_4)
    bl5_6 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl5_5)  
    pl5 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),padding='same')(bl5_6)
#     print("Block5: ",pl5.shape)
    
    
    #Down-Block6
    bl6_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(pl5)
    bl6_2 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl6_1)
    bl6_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl6_2)
    bl6_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl6_3)   
    bl6_5 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl6_4)
    bl6_6 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl6_5)    
    pl6 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),padding='same')(bl6_6)
#     print("Block6: ",pl6.shape)
    
    

    #Up-Block7
    bl7_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(pl6)
    bl7_2 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl7_1)
    bl7_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl7_1)
    bl7_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl7_1)   
    bl7_5 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl7_1)
    bl7_6 = keras.layers.Conv2DTranspose(64, kernel_size=(3,3), strides=(2, 2), padding='same',output_padding=1,activation=None)(bl7_1)


    #Up-Block8
    bl8 = keras.layers.Concatenate()([bl7_6, bl6_4])
    bl8_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl8)
    bl8_2 = keras.layers.Conv2D(96, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl8_1)
    bl8_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl8_2)
    bl8_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl8_3)   
    bl8_5 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl8_4)
    bl8_6 = keras.layers.Conv2DTranspose(64, kernel_size=(3,3), strides=(2, 2), padding='same',output_padding=1,activation=None)(bl8_5)


    #Up-Block9
    bl9 = keras.layers.Concatenate()([bl8_6, bl5_4])
    bl9_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl9)
    bl9_2 = keras.layers.Conv2D(96, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl9_1)
    bl9_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl9_2)
    bl9_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl9_3)   
    bl9_5 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl9_4)
    bl9_6 = keras.layers.Conv2DTranspose(64, kernel_size=(3,3), strides=(2, 2), padding='same',output_padding=1,activation=None)(bl9_5)
    
    #Up-Block9.5 (9_5)
    bl9_5_0 = keras.layers.Concatenate()([bl9_6, bl4_5_4])
    bl9_5_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl9_5_0)
    bl9_5_2 = keras.layers.Conv2D(96, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl9_5_1)
    bl9_5_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl9_5_2)
    bl9_5_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl9_5_3)   
    bl9_5_5 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl9_5_4)
    bl9_5_6 = keras.layers.Conv2DTranspose(64, kernel_size=(3,3), strides=(2, 2), padding='same',output_padding=1,activation=None)(bl9_5_5)


    #Up-Block10
    bl10 = keras.layers.Concatenate()([bl9_5_6, bl4_4])
    bl10_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl10)
    bl10_2 = keras.layers.Conv2D(96, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl10_1)
    bl10_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl10_2)
    bl10_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl10_3)   
    bl10_5 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl10_4)
    bl10_6 = keras.layers.Conv2DTranspose(64, kernel_size=(3,3), strides=(2, 2), padding='same',output_padding=1,activation=None)(bl10_5)
    
    #Up-Block10.5 (10_5)
    bl10_5_0 = keras.layers.Concatenate()([bl10_6, bl3_5_4])
    bl10_5_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl10_5_0)
    bl10_5_2 = keras.layers.Conv2D(96, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl10_5_1)
    bl10_5_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl10_5_2)
    bl10_5_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl10_5_3)   
    bl10_5_5 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl10_5_4)
    bl10_5_6 = keras.layers.Conv2DTranspose(64, kernel_size=(3,3), strides=(2, 2), padding='same',output_padding=1,activation=None)(bl10_5_5)


    #Up-Block11
    bl11 = keras.layers.Concatenate()([bl10_5_6, bl3_4])
    bl11_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl11)
    bl11_2 = keras.layers.Conv2D(96, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl11_1)
    bl11_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl11_2)
    bl11_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl11_3)   
    bl11_5 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl11_4)
    bl11_6 = keras.layers.Conv2DTranspose(64, kernel_size=(3,3), strides=(2, 2), padding='same',output_padding=1,activation=None)(bl11_5)


    #Up-Block12
    bl12 = keras.layers.Concatenate()([bl11_6, bl2_2])
    bl12_1 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl12)
    bl12_2 = keras.layers.Conv2D(96, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl12_1)
    bl12_3 = keras.layers.BatchNormalization(momentum=0.01,trainable=False,beta_initializer='zeros', gamma_initializer='ones')(bl12_2)
    bl12_4 = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), activation="relu")(bl12_3)
    output_layer = keras.layers.Conv2D(1, kernel_size=(1,1), padding='same', strides=(1,1), activation="sigmoid")(bl12_4)

    model = keras.models.Model(input_layer, output_layer)
    return model



model = Unet()

#Sequential model
model.compile(optimizer="Adamax", loss=['binary_crossentropy'], metrics=[iou_score,mean_iou])
print("Unet summary")
model.summary()

#Distributing work load on multiple GPUs
""" Use if more than 1 GPU is avaialble """
# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():
#     model = UNet()
#     model.compile(optimizer="Adamax", loss=['binary_crossentropy'], metrics=[iou_score,mean_iou])
#     model.summary()
#     .... copy below code of  model_checkpoint_callback and model.fit() 

checkpoint_filepath = './checkpoint_temp/'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
# mcp_save = ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min',save_weights_only=True)
# reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1, min_delta=1e-4, mode='min')

print("DeepUnet model + Starting with pre training")
print("Saving in checkpoint_temp + min val_loss")
print("Loss function: binary_crossentropy")
print("Metric = iou_score (only for positive labels) + mean_iou")
print("******************* Adamax ***********************")
# print("Using earlyStopping, mcp_save,reduce_lr_loss")

model.fit(
    x=x_train,
    y=y_train,
    batch_size=8,
    epochs=600,
    verbose=1,
    validation_data=(x_val, y_val),
    callbacks=[model_checkpoint_callback]
)



# Saving the Best Model

model = Unet()
model.compile(optimizer="Adamax", loss=['binary_crossentropy'], metrics=["acc"])
model.summary()

checkpoint_filepath = './checkpoint_temp/'
model.load_weights(checkpoint_filepath)

print("Validation:")
model.evaluate(x_val,y_val)

print("Train:")
model.evaluate(x_train,y_train)

model_name = 'x2_Adamax.h5'
model.save("./Data/saved_models/"+model_name)
print("Model saved: "+"./Data/saved_models/"+model_name)

