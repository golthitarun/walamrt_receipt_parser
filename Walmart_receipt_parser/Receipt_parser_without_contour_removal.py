
# coding: utf-8

# In[1]:

import cv2
import pytesseract 
import numpy as np
from PIL import *
import PIL.Image as Image,ImageDraw
import imutils
from scipy import ndimage
import sys,os
from scipy.ndimage.filters import rank_filter
import pandas as pd


# In[2]:

src = "/home/tharunn/Documents/project/images"


# In[3]:

# Method used to read data from the images by using cv2 and tesseract

def get_string(path):

    #To Read the image
    img = cv2.imread(path)
    x,y,z = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #To dialate and erode the images
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    
    #To apply adaptive Thresholding to the images so as to improve accuracy.
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    # To rotate the images horizontally so as to extract accurate text.
    if(x<y):
        img = ndimage.rotate(img, -90)
    cv2.imwrite(src+"/rotated.jpg", img)
    result = pytesseract.image_to_string(Image.open(src+"/rotated.jpg"))
    return result


# In[4]:

#To read the image from a directory, Method I used to check the the image processing method.
def Read_Dir(path,dest):
    for path,subirs,files in os.walk(path):
        for image in files:
            dest_file = os.path.join(dest,image+".txt")
            try:
                f = open(dest_file,'w')
                util_path = os.path.join(path,image)
                string = get_string(util_path)
                print >>f,string
            except e:
                print e
                continue


# In[5]:

#To read images as listed in the csv files and store them in a output location. 
def read_data(row):
    path = os.path.join(src,row['EXT_ID']+".jpg")
    dest_path = os.path.join(dest,row['EXT_ID']+".txt")
    f = open(dest_path, 'w')
    string = get_string(path)
    print >>f,string


# In[ ]:

#To run the algorithm on all the traina nd test data. 
path = "/home/tharunn/Documents/project/test"
dest = "/home/tharunn/Documents/project/test"
train = pd.read_csv("/home/tharunn/Documents/project/training_data.csv")
test = pd.read_csv("/home/tharunn/Documents/project/test_data.csv")
test.apply(read_data, axis=1, raw=True)
test.apply(read_data, axis=1, raw=True) 

