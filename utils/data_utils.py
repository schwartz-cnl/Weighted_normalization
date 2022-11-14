import tensorflow as tf
import numpy as np
import os
import cv2
from utils.aug_utils import *
import config_ws as c
def load_list(list_path, image_root_path):
    images = []
    labels = []
    with open(list_path, 'r') as f:
        for line in f:
            line = line.replace('\n', '').split(' ')
            images.append(os.path.join(image_root_path, line[0]))
            labels.append(int(line[1]))
    return images, labels

def load_image(image_path, label, augment=False, crop_10=False):
    """
    In training, it is highly recommended to set the augment to true.
    In test, the standard 10-crop test [1] is provided for fair comparison.
    [1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
    """
    image = cv2.imread(image_path.numpy().decode()).astype(np.float32)

    if augment:
        image = random_aspect(image)
        image = random_size(image)
        image = random_crop(image)
        image = random_flip(image)
        image = random_hsv(image)
        image = random_pca(image)
    else:
        image = random_size(image, target_size=256)
        if crop_10:
            image = test_10_crop(image)
        else:
            image = center_crop(image)

    image = normalize(image)

    label_one_hot = np.zeros(c.category_num)
    label_one_hot[label] = 1.0

    return image, label_one_hot

def train_iterator(list_path=c.train_list_path):
    images, labels = load_list(list_path, c.train_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(len(images))
    dataset = dataset.repeat()
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y, True, False], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(c.batch_size)
    it = dataset.__iter__()
    return it

def test_iterator(list_path=c.test_list_path):
    images, labels = load_list(list_path, c.test_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y, False, False], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(c.batch_size)
    it = dataset.__iter__()
    return it

def test_10_crop_iterator(list_path=c.test_list_path):
    images, labels = load_list(list_path, c.test_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y, False, True], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    it = dataset.__iter__()
    return it

# used for train keras ResNet models.
def define_shape(x,y):
    x.set_shape(c.input_shape)
    y.set_shape((c.category_num,))
    return (x,y)

def train_dataset(list_path=c.train_list_path):
    images, labels = load_list(list_path, c.train_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(len(images))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y, True, False], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(define_shape)
    dataset = dataset.batch(c.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def test_dataset(list_path=c.test_list_path, batch_size=c.batch_size):
    images, labels = load_list(list_path, c.test_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y, False, False], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(define_shape)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

####################################################################
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
#     phi = np.arctan2(y, x)
    return rho

def RMS_contrast(I):
    mean_I = np.mean(I)
    return (np.sqrt(np.sum((I-mean_I)**2)/I.size))

def change_slope_rgb(image, alpha=0, contrast=76.5):
    alpha_image = np.stack((change_slop_channel(image[:,:,0], alpha),
                     change_slop_channel(image[:,:,1], alpha),
                     change_slop_channel(image[:,:,2], alpha)),axis=2)
    rms_contrast_r = RMS_contrast(alpha_image[:,:,0])
    rms_contrast_g = RMS_contrast(alpha_image[:,:,1])
    rms_contrast_b = RMS_contrast(alpha_image[:,:,2])
    rms_contrast = np.sqrt(rms_contrast_r**2 + rms_contrast_g**2 + rms_contrast_b**2)
    return alpha_image/rms_contrast*contrast + 128*(1-contrast/rms_contrast)

def change_slop_channel(image, alpha=0):
    im_max = np.max(image)
    im_min = np.min(image)
    
    size_image = image.shape
    L = np.min(size_image)
    fft_image = np.fft.fftshift(np.fft.fft2(image))
    abs_im = np.abs(fft_image)
    freq = np.arange(-L/2,L/2)
    x, y = np.meshgrid(freq, freq)
    ro = cart2pol(x, y)
    ro = np.round(ro)

    abs_im_av = np.zeros((L,L))
    for r in range(L):
        idx = ro==r
        temp = np.mean(abs_im[idx])
        abs_im_av[idx] = temp

    zeroslope = fft_image/abs_im_av
    filter_image = zeroslope*((1+ro)**float(-alpha))
    filter_image_shiftback = np.fft.ifftshift(filter_image)
    image_new = np.real(np.fft.ifft2(filter_image_shiftback))
    f = (image_new-np.min(image_new))/(np.max(image_new)-np.min(image_new))*(im_max-im_min)+im_min
    return f.astype(np.float32)

def load_image_slope(image_path, label, augment=False, crop_10=False, alpha=0):
    """
    In training, it is highly recommended to set the augment to true.
    In test, the standard 10-crop test [1] is provided for fair comparison.
    [1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
    """
    image = cv2.imread(image_path.numpy().decode()).astype(np.float32)

    if augment:
        image = random_aspect(image)
        image = random_size(image)
        image = random_crop(image)
        image = change_slope_rgb(image,alpha)
        image = random_flip(image)
        image = random_hsv(image)
        image = random_pca(image)
    else:
        image = random_size(image, target_size=256)
        if crop_10:
            # image = change_slope_rgb(image,alpha)
            image = test_10_crop(image)
            for i in range(10):
                image[i,...] = change_slope_rgb(image[i,...],alpha)
        else:
            image = center_crop(image)
            image = change_slope_rgb(image,alpha)

    image = normalize(image)

    label_one_hot = np.zeros(c.category_num)
    label_one_hot[label] = 1.0

    return image, label_one_hot

def train_dataset_slope(list_path=c.train_list_path, alpha=0):
    images, labels = load_list(list_path, c.train_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(len(images))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image_slope, inp=[x, y, True, False, alpha], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(define_shape)
    dataset = dataset.batch(c.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def test_dataset_slope(list_path=c.test_list_path, alpha=0, batch_size=c.batch_size):
    images, labels = load_list(list_path, c.test_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image_slope, inp=[x, y, False, False, alpha], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(define_shape)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def test_10_crop_iterator_slope(list_path=c.test_list_path, alpha=0):
    images, labels = load_list(list_path, c.test_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image_slope, inp=[x, y, False, True, alpha], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    it = dataset.__iter__()
    return it

if __name__ == '__main__':
    # Annotate the 'normalize' function for visualization
    # it = train_iterator()
    it = test_10_crop_iterator('../data/validation_label.txt')
    images, labels = it.next()
    # print(np.shape(images), np.shape(labels))
    for i in range(10):
        print(np.where(labels[i].numpy() != 0))
        cv2.imshow('show', images[i].numpy().astype(np.uint8))
        cv2.waitKey(0)
