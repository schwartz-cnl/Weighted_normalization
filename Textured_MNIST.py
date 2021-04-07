# generated Textured MNIST dataset.
# Textures are from Brodatz Textures and synthesized using a method by
# Portilla, Javier, and Eero P. Simoncelli. "A parametric texture model
# based on joint statistics of complex wavelet coefficients." International
# journal of computer vision 40.1 (2000): 49-70.

import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

texture_list = ['D1','D3','D6','D9','D15','D18','D20','D40','D73','D74','D95','D110']
texture = []
for folder in texture_list:
    for im in range(1,51):
        texture.append(Image.open('naturalistic/'+folder+'/naturalisticTexture_0'+str(im).zfill(2)+'.jpg'))

# Some textures are very similar. Exclude those "incompatible" choices.
compatible = [[2,3,6,7],
              [2,5,6,7,],
              [0,1,3,4,5,6,8,10,11],
              [2,6,7],
              [2,7],
              [1,2,4,6],
              [0,1],
              [0,1,3,4,5,8,9,10],
              [2,7],
              [2,7],
              [2,7,11],
              [2,7]
             ]

max_shift = 36
im_no = 18
textured_x_train = np.empty([len(x_train),226,226],dtype = 'uint8')
for im_no in range(len(x_train)):
    im = np.zeros((226,226))
    texture_class = np.random.randint(12)
    digit_texture_class = np.random.randint(len(compatible[texture_class]))
    digit_texture_class = compatible[texture_class][digit_texture_class]  
    texture_no = np.random.randint(50)
    digit_texture_no = np.random.randint(50)
    im_digit= Image.fromarray(x_train[im_no,:,:])
    im_digit = im_digit.resize((160,160))
    (x_shift,y_shift) = np.random.randint(max_shift,size=(2, 1))
    im[15+x_shift[0]:175+x_shift[0], 15+y_shift[0]:175+y_shift[0]] = im_digit
    (texture_x_shift, texture_y_shift) = np.random.randint(30,size=(2,1))
    digit_texture = texture[digit_texture_class][digit_texture_no].crop((texture_x_shift[0], texture_y_shift[0], texture_x_shift[0]+226, texture_y_shift[0]+226))
    digit = (np.array(im))*np.array(digit_texture)/255
    (texture_x_shift, texture_y_shift) = np.random.randint(30,size=(2,1))
    background_texture = texture[texture_class][texture_no].crop((texture_x_shift[0], texture_y_shift[0], texture_x_shift[0]+226, texture_y_shift[0]+226))
    mixed_im = np.array(background_texture)*((255-np.array(im))/255)+digit
    mixed_im = mixed_im.astype('uint8')
    textured_x_train[im_no,:,:] = mixed_im
    if im_no%1000 == 0:
        print('im_no: '+str(im_no)+'/'+str(len(x_train)))
np.save('x_train_one_instance.npy',textured_x_train)
np.save('y_train_one_instance.npy',y_train)


textured_x_test = np.empty([len(x_test),226,226],dtype = 'uint8')
for im_no in range(len(x_test)):
        im = np.zeros((226,226))
        texture_class = np.random.randint(12)
        digit_texture_class = np.random.randint(len(compatable[texture_class]))
        digit_texture_class = compatable[texture_class][digit_texture_class]  
        texture_no = np.random.randint(50)
        digit_texture_no = np.random.randint(50)
        im_digit= Image.fromarray(x_test[im_no,:,:])
        im_digit = im_digit.resize((160,160))
        (x_shift,y_shift) = np.random.randint(max_shift,size=(2, 1))
        im[15+x_shift[0]:175+x_shift[0], 15+y_shift[0]:175+y_shift[0]] = im_digit
        (texture_x_shift, texture_y_shift) = np.random.randint(30,size=(2,1))
        digit_texture = texture[digit_texture_class][digit_texture_no].crop((texture_x_shift[0], texture_y_shift[0], texture_x_shift[0]+226, texture_y_shift[0]+226))
        digit = (np.array(im))*np.array(digit_texture)/255
        (texture_x_shift, texture_y_shift) = np.random.randint(30,size=(2,1))
        background_texture = texture[texture_class][texture_no].crop((texture_x_shift[0], texture_y_shift[0], texture_x_shift[0]+226, texture_y_shift[0]+226))
        mixed_im = np.array(background_texture)*((255-np.array(im))/255)+digit
        mixed_im = mixed_im.astype('uint8')
        textured_x_test[im_no,:,:] = mixed_im
        if im_no%1000 == 0:
            print('im_no: '+str(im_no)+'/'+str(len(x_test)))

np.save('x_test_one_instance.npy',textured_x_test)
np.save('y_test_one_instance.npy',y_test)