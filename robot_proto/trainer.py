# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 22:52:43 2021

@author: caiy3
"""


import tensorflow as tf
import numpy as np
import cv2
from mobilenetv2 import *
from reader import *
BATCH_SIZE=10
MAX_STEPS=70000
RETRAIN_STEP=None
def inference(input_tf,n_classes):
    net = MobileNetV2(n_classes=n_classes,depth_rate=1.0,is_training=True)
    output = net.build_graph(input_tf)
    return output

def load_model(sess,step):
    if step is None:
        return 0
    else:
        saver = tf.train.Saver()
        saver.restore(sess,'model/model-%d'%step)
        print('load model from model/model-%d'%step)
        return step

def get_loss(logits,labels):
    loss =tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
    loss = tf.reduce_mean(loss)
    return loss

def main():
    start_step=0
    if not RETRAIN_STEP is None:
        start_step = RETRAIN_STEP
    global_step= tf.Variable(start_step,trainable=False,name='global_step2')
    datasets= DataReader(batch_size= BATCH_SIZE, hw_root='datasets/train')
    img_batch,onehot_batch=datasets.get()
    logits_tf=inference(img_batch,datasets.n_classes)
    loss_tf=get_loss(logits_tf,onehot_batch)
    correct_pred=tf.equal(tf.argmax(logits_tf,1),tf.argmax(onehot_batch,1))
    accuracy_tf=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    decay_steps=int(len(datasets)/BATCH_SIZE*3)
    lr_tf=tf.train.exponential_decay(0.01,global_step,decay_steps,0.94,staircase=True)
    opt =tf.train.AdamOptimizer(lr_tf,name="Adam6")
    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op=opt.minimize(loss_tf,global_step=global_step)
    saver=tf.train.Saver()
    with tf.name_scope('scalar1') as scope:
        tf.summary.scalar('loss1',loss_tf)
        tf.summary.scalar('accuracy1',accuracy_tf) 
        tf.summary.scalar('lr1',lr_tf)
        summaries=tf.get_collection(tf.GraphKeys.SUMMARIES,scope)
        summary_op=tf.summary.merge(summaries)
        summary_writer=tf.summary.FileWriter('./model',graph=tf.get_default_graph())
    gpu_options=tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        load_model(sess, RETRAIN_STEP)
        for i in range(start_step+1,MAX_STEPS):
            _,loss,accuracy,summary,lr=sess.run([train_op,loss_tf,summary_op,lr_tf])
            summary_writer.add_summary(summary,i)
            print('step:%d loss=%f accuracy = %f lr=%f'%(i,loss,accuracy,lr))
            if i%100==0:
                saver.save(sess,'model/model-%d'%i)
        coord.request_stop()
        coord.join(threads)

if __name__=='__main__':
    main()

    