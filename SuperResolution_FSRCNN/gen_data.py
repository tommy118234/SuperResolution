# -*- coding: utf-8 -*-
import h5py
import numpy as np
import glob 
import cv2


def modcrop(img, modulo):
    sz = np.shape(img)
    sz = sz - np.mod(sz, modulo)
    return img[0 : sz[0] - 1, 0 : sz[1] - 1]


def store2hdf5(filename, data, labels):

    f = h5py.File(filename, "w")

    f.create_dataset('data', data=data)
    f.create_dataset('label', data=labels)
    f.close()

folder = './Train/'
savepath = './examples/SRCNN/train.h5'
size_input = 11
size_label = 19
scale = 3
stride = 4

max_sample = 30000
data = np.reshape(np.array([]), (size_input, size_input, 1, 0))
label = np.reshape(np.array([]), (size_label, size_label, 1, 0))
data = np.zeros((size_input, size_input, 1, max_sample))
label = np.zeros((size_label, size_label, 1, max_sample))
padding = abs(size_input - size_label) / 2
count = 0

filepaths = glob.glob(folder + '*.bmp')

for f in filepaths:

    image = cv2.imread(f)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    image = image[:, :, 0] / 255.0

    im_label = modcrop(image, scale)
    sz = np.shape(im_label)
    im_input = cv2.resize(im_label, (sz[1]/scale,sz[0]/scale), interpolation=cv2.INTER_CUBIC)
    sz = np.shape(im_input)

    """
        TODO #1:
    	Generate training data pairs.
    """
    for x in range(0, sz[0]-size_input, stride):
        for y in range(0, sz[1]-size_input, stride):

            locx = int(scale * (x + np.floor((size_input - 1) / 2)) - np.floor((size_label + scale) / 2 - 1))
            locy = int(scale * (y + np.floor((size_input - 1) / 2)) - np.floor((size_label + scale) / 2 - 1))

            subim_input = im_input[x: x + size_input, y: y + size_input]
            subim_label = im_label[locx: locx + size_label, locy: size_label+locy]

            count += 1
            data[:, :, 0, count-1] = subim_input
            label[:, :, 0, count-1] = subim_label
"""
TODO #1:
Randomly permute the data pairs.
"""
order = np.random.permutation(range(count-1))
data = data[:, :, :, order]
label = label[:, :, :, order]

data = np.transpose(data, (3, 2, 1, 0))
label = np.transpose(label, (3, 2, 1, 0))

store2hdf5(savepath, data, label)