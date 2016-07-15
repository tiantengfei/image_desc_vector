import os
from PIL import Image
import numpy as np
from random import shuffle


def get_files(base, filename):
    f = os.path.join(base, filename)

    onlyfiles = [os.path.join(f, fi) for fi in os.listdir(f)]
    # if os.path.isfile(os.path.join(f, "fi"))]

    # print onlyfiles

    return onlyfiles


def get_image_with_lable(*image_data):
    (label1, im1), (label2, im2) = image_data
    image1_data = im1.resize((64, 64))
    image2_data = im2.resize((64, 64))

    # print np.array(image1_data).shape, np.array(image2_data).shape
    m1 = np.transpose(np.array(image1_data), (1, 0, 2))
    m2 = np.transpose(np.array(image2_data), (1, 0, 2))

    # print m1.shape, m2.shape
    m3 = np.concatenate((m1, m2), axis=0)

    print m3.shape
    label = '0' if label1 == label2 else '1'
    return label, m3.tostring()


def get_shape(image):
    image_1 = image.resize((64, 64))
    im1 = np.array(image_1)
    return im1.shape


def write_image_to_files():
    images = [get_files("scooter", str(i)) for i in range(1, 5)]
    # print "images:", len(images)
    ims = [(i, Image.open(im)) for i in range(4) for im in images[i] if Image.open(im).mode == 'RGB']
    print "ims:", len(ims)

    shuffle(ims)
    num = 0
    for i in range(len(ims)):
        for j in range(i + 1, len(ims)):
            image_1 = get_shape(ims[i][1])
            image_2 = get_shape(ims[j][1])
            im1 = np.array(image_1)
            im2 = np.array(image_2)
            if im2.shape == im1.shape:
                num = num + 1
    print 'num:', num
    # print ims
    data_list = [get_image_with_lable(ims[i], ims[j])
                 for i in range(len(ims)) for j in range(i + 1, len(ims))
                 if get_shape(ims[i][1]) == get_shape(ims[j][1])]

    print(len(data_list))

    files = [open('image_%d.bin' % i, 'wb') for i in range(4)]

    count = 0
    # with open("image_test.bin", "wb") as f:
    #   for label, image in data_list:
    #      f.write(label)
    #      f.write(image)
    # *      count = count + 1

    for label, image in data_list:
        files[count / 5000].write(label)
        files[count / 5000].write(image)
        count = count + 1
    print 'count:', count
    for f in files:
        f.close()


def read_images():
    with open("image_test.bin", "rb") as f:
        while True:
            label = f.read(1)
            if label == '':
                break
            im = f.read(24576)
            im = np.fromstring(im, dtype=np.uint8).reshape(128, 64, 3)

            print label, im.shape


write_image_to_files()
#read_images()
# get_files("scooter", "2")
