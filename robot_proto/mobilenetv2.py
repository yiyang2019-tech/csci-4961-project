# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 20:17:21 2021

@author: caiy3
"""


import tensorflow as tf
from ops import *
class MobileNetV2:
    def __init__(self,n_classes,depth_rate=1.0,is_training=True):
        self.n_classes=n_classes
        self.depth_rate=depth_rate
        self.is_training=is_training
    def depth(self,n,rate,min_depth=8):
        return make_divisible(n*rate,divisor=8,min_value=min_depth)
    def parseNet(self,x,layer_list):
        for i, layer in enumerate(layer_list):
            if(len(layer))==4:
                op,k,s,out_c=layer
            elif len(layer)==5:
                op,k,s,out_c,expand_rate=layer
            else:
                raise "Unknown layer="+str(layer)
            out_c=self.depth(out_c, self.depth_rate)
            if op==conv:
                x=conv(x,'Conv_%d'%i,k=k,s=s,out_c=out_c,is_training=self.is_training)
            elif op==expanded_conv:
                x=expanded_conv(x,'Conv_%d'%i,k=k,s=s,out_c=out_c,
                                expand_rate=expand_rate,is_training=self.is_training)
        return x
    def build_graph(self,input_tf):
        layer_list=[[conv,3,2,32],
                    [expanded_conv,3,1,16,1],
                    [expanded_conv,3,2,24,6],
                    [expanded_conv,3,1,24,6],
                    [expanded_conv,3,2,32,6],
                    [expanded_conv,3,1,32,6],
                    [expanded_conv,3,1,32,6],
                    [expanded_conv,3,2,64,6],
                    [expanded_conv,3,1,64,6],
                    [expanded_conv,3,1,64,6],
                    [expanded_conv,3,1,64,6],
                    [expanded_conv,3,1,96,6],
                    [expanded_conv,3,1,96,6],
                    [expanded_conv,3,1,96,6],
                    [expanded_conv,3,2,160,6],
                    [expanded_conv,3,1,160,6],
                    [expanded_conv,3,1,160,6],
                    [expanded_conv,3,1,320,6],
                    [conv,1,1,1280]
                    ]
        x=self.parseNet(input_tf, layer_list)
        x=global_pool(x)
        if self.is_training:
            x=tf.nn.dropout(x,keep_prob=0.90)
        x=conv(x,'Conv_20',k=1,s=1,out_c=self.n_classes,with_bias=True,use_bn=False,
               relu=None,is_training=self.is_training)
        x=tf.squeeze(x,[1,2])
        return x

