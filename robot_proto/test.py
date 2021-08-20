# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 00:53:14 2021

@author: caiy3
"""


import tensorflow as tf
import numpy as np
import cv2
from mobilenetv2 import *
from reader import *
def inference(input_tf,n_classes):
    net = MobileNetV2(n_classes=n_classes,depth_rate=1.0,is_training=False)
    output = net.build_graph(input_tf)
    return output

def main():
    datasets=DataReader(batch_size=1, hw_root='datasets/test')
    img_tf,label_tf=datasets.get_img_and_label()
    img_tf = tf.cast(img_tf,tf.float32)
    label_tf=tf.expand_dims(tf.one_hot(label_tf, 2),axis=0)
    logit_tf=inference(tf.expand_dims(img_tf,axis=0),2)
    correct_pred=tf.equal(tf.argmax(logits_tf,1),tf.argmax(label_tf,1))
    accuracy_tf=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    saver=tf.train.Saver()
    gpu_options=tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess,'model1_1/model-70000')
        acc_sum=0
        for i in range(len(datasets)):
            acc=sess.run(accuracy_tf)
            acc_sum=acc_cum+acc
            if i>0 and i%100==0:
                print("accuracy{}".format(acc_sum*1/i)
            coord.request_stop()
            coord.join(threads)
if __name__=='__main__':
    main()
    