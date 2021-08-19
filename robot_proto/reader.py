# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 20:47:56 2021

@author: caiy3
"""


import tensorflow as tf
import os
import cv2
import numpy as np
class DataReader:
    def __init__(self,batch_size,hw_root,img_size=(64,64)):
        self.hw_root=hw_root
        self.batch_size=batch_size
        self.label_tf=self.read_labels
        self.img_size=img_size
    def read_labels(self):
        label_list=list()
        dirs=[name for name in os.listdir(self.hw_root) if os.path.isdir(self.hw_root+'/'+name)]
        self.n_classes=len(dirs)
        for dir_name in dirs:
            label = dir_name
            for img_name in os.listdir(self.hw_root+'/'+dir_name):
                img_path=self.hw_root+'/'+dir_name+'/'+img_name
                label_list.append(img_path+' '+label)
        self.size=len(label_list)
        label_list_tf=tf.convert_to_tensor(label_list,dtype=tf.string)
        [label_tf]=tf.train.slice_input_producer([label_list_tf])
        return label_tf
    def __len__(self):
        return self.size
    def get_img_and_label(self):
        kvs = tf.string_split([self.label_tf],delimiter=' ')
        img_path=kvs.values[0]
        label_str=kvs.values[1]
        label_tf=tf.string_to_number(label_str,out_type=tf.int32)
        img_data_tf=tf.read_file(img_path)
        img_tf=tf.image.decode_jpeg(img_data_tf,channels=3)
        def resize_with_padding(img):
            im_h,im_w,im_c=img.shape
            wh=max(im_h,im_w)
            wh_img=np.ones((wh,wh,im_c),dtype=np.uint8)*255
            left=int((wh-im_w)/2)
            top=int((wh-im_h)/2)
            wh_img[top:top+im_h,left:left+im_w]=img
            wh_img=cv2.resize(wh_img,self.img_size)
            return wh_img
        img_tf=tf.py_func(resize_with_padding,[img_tf],tf.uint8)
        w,h=self.img_size
        img_tf.set_shape((h,w,3))
        return img_tf,label_tf
    def get(self):
        img_tf,label_tf=self.get_img_and_label()
        img_tf=tf.cast(img_tf,tf.float32)
        label_tf=tf.one_hot(label_tf,self.n_classes)
        img_batch,label_batch = tf.train.shuffle_batch_join([[img_tf,label_tf]],
                                                            capacity=1000,min_after_dequeue=10)
        return img_batch, label_batch

        
        
        
        