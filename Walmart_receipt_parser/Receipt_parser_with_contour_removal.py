
# coding: utf-8

# In[3]:

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


# In[4]:

src = "/home/tharunn/Documents/project/images"


# In[5]:

# Method used to Dilate using an NxN '+' sign shape. ary is np.uint8.
def dilate(ary, N, iterations): 
    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[(N-1)/2,:] = 1
    dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)

    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[:,(N-1)/2] = 1
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
    return dilated_image




# In[6]:

def intersect_crops(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)


# In[7]:

# Method used to union two crops. 
def union_crops(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


# In[8]:

#Method used to crop the given area.
def crop_area(crop):
    x1, y1, x2, y2 = crop
    return max(0, x2 - x1) * max(0, y2 - y1)


# In[9]:

#Calculate bounding box & the number of set pixels for each contour.
def props_for_contours(contours, ary):
    c_info = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        c_im = np.zeros(ary.shape)
        cv2.drawContours(c_im, [c], 0, 255, -1)
        c_info.append({
            'x1': x,
            'y1': y,
            'x2': x + w - 1,
            'y2': y + h - 1,
            'sum': np.sum(ary * (c_im > 0))/255
        })
    return c_info


# In[10]:

#Method used to Slightly expand the crop to get full contours.This will expand to include any contours it currently intersects, but will
#not expand past a border.
def pad_crop(crop, contours, edges, border_contour, pad_px=15):
    bx1, by1, bx2, by2 = 0, 0, edges.shape[0], edges.shape[1]
    if border_contour is not None and len(border_contour) > 0:
        c = props_for_contours([border_contour], edges)[0]
        bx1, by1, bx2, by2 = c['x1'] + 5, c['y1'] + 5, c['x2'] - 5, c['y2'] - 5

    def crop_in_border(crop):
        x1, y1, x2, y2 = crop
        x1 = max(x1 - pad_px, bx1)
        y1 = max(y1 - pad_px, by1)
        x2 = min(x2 + pad_px, bx2)
        y2 = min(y2 + pad_px, by2)
        return crop
    
    crop = crop_in_border(crop)

    c_info = props_for_contours(contours, edges)
    changed = False
    for c in c_info:
        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
        this_area = crop_area(this_crop)
        int_area = crop_area(intersect_crops(crop, this_crop))
        new_crop = crop_in_border(union_crops(crop, this_crop))
        if 0 < int_area < this_area and crop != new_crop:
            #print '%s -> %s' % (str(crop), str(new_crop))
            changed = True
            crop = new_crop

    if changed:
        return pad_crop(crop, contours, edges, border_contour, pad_px)
    else:
        return crop


# In[11]:

# Method used to Find a crop which strikes a good balance of coverage/compactness.
 #   Returns an (x1, y1, x2, y2) tuple.
def find_optimal_components_subset(contours, edges):
    c_info = props_for_contours(contours, edges)
    c_info.sort(key=lambda x: -x['sum'])
    total = np.sum(edges) / 255
    area = edges.shape[0] * edges.shape[1]

    c = c_info[0]
    del c_info[0]
    this_crop = c['x1'], c['y1'], c['x2'], c['y2']
    crop = this_crop
    covered_sum = c['sum']

    while covered_sum < total:
        changed = False
        recall = 1.0 * covered_sum / total
        prec = 1 - 1.0 * crop_area(crop) / area
        f1 = 2 * (prec * recall / (prec + recall))
        #print '----'
        for i, c in enumerate(c_info):
            this_crop = c['x1'], c['y1'], c['x2'], c['y2']
            new_crop = union_crops(crop, this_crop)
            new_sum = covered_sum + c['sum']
            new_recall = 1.0 * new_sum / total
            new_prec = 1 - 1.0 * crop_area(new_crop) / area
            new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)

            # Add this crop if it improves f1 score,
            # _or_ it adds 25% of the remaining pixels for <15% crop expansion.
            # ^^^ very ad-hoc! make this smoother
            remaining_frac = c['sum'] / (total - covered_sum)
            new_area_frac = 1.0 * crop_area(new_crop) / crop_area(crop) - 1
            if new_f1 > f1 or (
                    remaining_frac > 0.25 and new_area_frac < 0.15):
                '''print '%d %s -> %s / %s (%s), %s -> %s / %s (%s), %s -> %s' % (
                        i, covered_sum, new_sum, total, remaining_frac,
                        crop_area(crop), crop_area(new_crop), area, new_area_frac,
                        f1, new_f1)'''
                crop = new_crop
                covered_sum = new_sum
                del c_info[i]
                changed = True
                break

        if not changed:
            break

    return crop


# In[12]:

def find_components(edges, max_components=16):
   
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.
    count = 21
    dilation = 5
    n = 1
    while count > 16:
        n += 1
        dilated_image = dilate(edges, N=3, iterations=n)
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
    #print dilation
    #Image.fromarray(edges).show()
    #Image.fromarray(255 * dilated_image).show()
    return contours


# In[13]:

def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))


# In[14]:

# Method used to remove everything outside a border contour.
def remove_border(contour, ary):
    # Use a rotated rectangle (should be a good approximation of a border).
    # If it's far from a right angle, it's probably two sides of a border and
    # we should use the bounding box instead.
    c_im = np.zeros(ary.shape)
    r = cv2.minAreaRect(contour)
    degs = r[2]
    if angle_from_right(degs) <= 10.0:
        box = cv2.cv.BoxPoints(r)
        box = np.int0(box)
        cv2.drawContours(c_im, [box], 0, 255, -1)
        cv2.drawContours(c_im, [box], 0, 0, 4)
    else:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

    return np.minimum(c_im, ary)


# In[15]:

# Method used to find the border components. 
def find_border_components(contours, ary):
    borders = []
    area = ary.shape[0] * ary.shape[1]
    for i, c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        if w * h > 0.5 * area:
            borders.append((i, x, y, x + w - 1, y + h - 1))
    return borders


# In[16]:

#Method used to Shrink im until its longest dimension is <= max_dim.
def downscale_image(im, max_dim=2048):
    a, b = im.size
    if max(a, b) <= max_dim:
        return 1.0, im

    scale = 1.0 * max_dim / max(a, b)
    new_im = im.resize((int(a * scale), int(b * scale)), Image.ANTIALIAS)
    return scale, new_im


# In[17]:

# MEthod used to remoe contours from the image. This method takes help from all above methods. 
def remove_contours(path):
    
    scale, im = downscale_image(Image.open(path))
    edges = cv2.Canny(np.array(im), 100, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    borders = find_border_components(contours, edges)
    
    border_contour = None
    if len(borders):
        border_contour = contours[borders[0][0]]
        edges = remove_border(border_contour, edges)
    edges = 255 * (edges > 0).astype(np.uint8)
    
    maxed_rows = rank_filter(edges, -4, size=(1, 20))
    maxed_cols = rank_filter(edges, -4, size=(20, 1))
    debordered = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
    edges = debordered
    
    contours = find_components(edges)
    if len(contours) == 0:
        #print '%s -> (no text!)' % path
        return
    
    crop = find_optimal_components_subset(contours, edges)
    crop = pad_crop(crop, contours, edges, border_contour)
    
    crop = [int(x / scale) for x in crop]
    text_im = Image.open(path).crop(crop)
    
    text_im.save(src+"/crop.jpg")


# In[18]:

def get_string(path):
    remove_contours(path)
    img = cv2.imread(path_crop_image) #Here the path of the cropped image should be provided. 
    x,y,z = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    if(x<y):
        img = ndimage.rotate(img, -90)
    cv2.imwrite(src+"/rotated.jpg", img)
    
      
    result = pytesseract.image_to_string(Image.open(src+"/rotated.jpg"))
    return result


# In[19]:

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


# In[20]:

def read_data(row):
    path = os.path.join(src,row['EXT_ID']+".jpg")
    dest_path = os.path.join(dest,row['EXT_ID']+".txt")
    f = open(dest_path, 'w')
    string = get_string(path)
    print >>f,string


# In[22]:


path = "/home/tharunn/Documents/project/test"
dest = "/home/tharunn/Documents/project/test"
train = pd.read_csv("/home/tharunn/Documents/project/training_data.csv")
test = pd.read_csv("/home/tharunn/Documents/project/test_data.csv")
test.apply(read_data, axis=1, raw=True)
test.apply(read_data, axis=1, raw=True) 


# In[ ]:




# In[ ]:



