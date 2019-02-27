from PIL import Image
import os
import numpy as np
from keras.utils import np_utils

def normalizeImg(im):
    arr = np.array(im)
    arr = arr.astype('float')
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def resizeImg(im, target_size):
    old_size = im.size
    ratio = float(target_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (target_size, target_size))
    new_im.paste(im, ((target_size-new_size[0])//2, (target_size-new_size[1])//2))
    return new_im

def processImg(im, target_size):
    norm_im_arr = normalizeImg(im)
    norm_im = Image.fromarray(norm_im_arr.astype('uint8'),'RGB')
    new_im = resizeImg(norm_im, target_size)
    return new_im

def createProcessedDirs(src, tgt):
    for (dirpath, dirnames, filenames) in os.walk(src, topdown=False):
        for dirname in dirnames:
            tgtPath = os.path.join(tgt, dirname)
            if not os.path.exists(tgtPath):
                os.makedirs(tgtPath)

def processAllImages(src, tgt, targetimagesize):
    createProcessedDirs(src, tgt)
    
    for (dirpath, dirnames, filenames) in os.walk(src, topdown=False):
        for filename in filenames:
            file = os.path.join(dirpath, filename)
            label = dirpath.split('/')[-1]
            outputFile = os.path.join(tgt, label, filename)
            print(file + "=>" + outputFile)
            im = Image.open(file)
            new_im = processImg(im, targetimagesize)
            new_im_arr = np.array(new_im)
            np.save(outputFile, new_im_arr)

def load(tgt):
    imgs = []
    labels = []
    
    for (dirpath, dirnames, filenames) in os.walk(tgt, topdown=False):
        for filename in filenames:
            file = os.path.join(dirpath, filename)
            im_arr = np.load(file)
            label = dirpath.split('/')[-1]
            imgs.append(im_arr.flatten())
            labels.append(label)
    return imgs, labels

def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))