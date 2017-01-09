# -*- coding: utf-8 -*-
import caffe
import numpy as np
import glob
import cv2
# config paths
model_path = "examples/SRCNN/SRCNN_test.prototxt"
param_path = "examples/SRCNN/SRCNN_iter_10000.caffemodel"
folder = './Test/Set5/'
img_path = glob.glob(folder + '*.bmp')
shrink = 12
net = caffe.Net(model_path, param_path, caffe.TEST)
count = 0
total_psnr = 0
# PSNR
def psnr(img1, img2, hei, wid):
    mse = np.sum((img1 - img2) ** 2)/(hei*wid)
    return 10 * np.log10(1.0 / mse)
for f in img_path:
    count += 1
    img = cv2.imread(f)
    sz = np.shape(img)
    img = cv2.resize(cv2.resize(img, (sz[1]/3, sz[0]/3), interpolation=cv2.INTER_AREA), (sz[1]+shrink, sz[0]+shrink),
                     interpolation=cv2.INTER_CUBIC)
    out = np.zeros((sz[0], sz[1], 3))
# super resolution for each channel
    for i in range(0, 3):
        ch = img[:, :, i] / 255.0
        net.blobs['data'].reshape(1, 1, sz[0]+shrink, sz[1]+shrink)
        net.blobs['data'].data[...] = np.reshape(ch, (1, 1, sz[0]+shrink, sz[1]+shrink))
        net.forward()
        x = net.blobs['conv3'].data[...]
        out[:, :, i] = np.squeeze(x)
# save outputs
    img = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('input_'+str(count)+'.png', img)
    cv2.imwrite('output_'+str(count)+'.png', out * 255)
# calculate psnr for each img
    psnr_single = np.mean(psnr(out, img/255.0, sz[0], sz[1]))
    print psnr_single
    total_psnr += psnr_single
print ("Mean PSNR for Set 5 is:"+str(total_psnr/count))


folder = './Test/Set14/'
img_path = glob.glob(folder + '*.bmp')
for f in img_path:
    count += 1
    img = cv2.imread(f)
    sz = np.shape(img)
    img = cv2.resize(cv2.resize(img, (sz[1]/3, sz[0]/3), interpolation=cv2.INTER_AREA), (sz[1]+shrink, sz[0]+shrink),
                     interpolation=cv2.INTER_CUBIC)
    out = np.zeros((sz[0], sz[1], 3))
# super resolution for each channel
    for i in range(0, 3):
        ch = img[:, :, i] / 255.0
        net.blobs['data'].reshape(1, 1, sz[0]+shrink, sz[1]+shrink)
        net.blobs['data'].data[...] = np.reshape(ch, (1, 1, sz[0]+shrink, sz[1]+shrink))
        net.forward()
        x = net.blobs['conv3'].data[...]
        out[:, :, i] = np.squeeze(x)
# save outputs
    img = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite('input_'+str(count)+'.png', img)
    #cv2.imwrite('output_'+str(count)+'.png', out * 255)
# calculate psnr for each img
    psnr_single = np.mean(psnr(out, img/255.0, sz[0], sz[1]))
    print psnr_single
    total_psnr += psnr_single
print ("Mean PSNR for Set 14 is:"+str(total_psnr/count))