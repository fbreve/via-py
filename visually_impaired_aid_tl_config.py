# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:59:14 2023

@author: fbrev
"""

class via_config():
    
    def __init__(self):
           
        self.FAST_RUN = False
        self.IMAGE_CHANNELS=3
        
        self.ALT_DENSE = False # True, False
        self.POOLING = 'avg' # None, 'avg', 'max'
        self.DATA_AUG = False # True, False
        self.DATA_AUG_MULT = 1 # >=1
        self.BATCH_SIZE = 16 # Note: IJCNN results were performed with 4
        self.DATASET_PATH = "../via-dataset/images/"
        self.MODEL_TYPE_LIST = ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 
                'ResNet152','ResNet50V2', 'ResNet101V2', 'ResNet152V2', "InceptionV3",
                'InceptionResNetV2', 'MobileNet', 'DenseNet121', 'DenseNet169',
                'DenseNet201', 'NASNetMobile', 'MobileNetV2',
                'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 
                'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
                'EfficientNetB6', 'EfficientNetB7', ]
                # NASNetLarge was removed because it is apparently too much for our GPU (GTX 1080 Ti)
        
        # The historical DONALD and PRECISION profiles did not define this
        # attribute, although the main training script requires it.
        # The final index in the model list is included by Python's slice
        # because MODEL_TYPE_START is converted from 1-based to 0-based.
        self.MODEL_TYPE_END = len(self.MODEL_TYPE_LIST)

        import socket
        hostname = socket.gethostname()
        
        if hostname=='DONALD':
            self.CONF_NUMBER = 900
            self.FINE_TUN = True
            self.MULTI_OPTIMIZER = False
            self.OPTIMIZER = 'RMSprop' # 'RMSprop', 'Adam', etc. Note: IJCNN results with rmsprop
            self.REST_BEST_W = False
            self.MODEL_TYPE_START = 18
            
        elif hostname=='PRECISION':
            self.CONF_NUMBER = 13
            self.FINE_TUN = True
            self.MULTI_OPTIMIZER = True
            self.OPTIMIZER = 'Adam' # 'RMSprop', 'Adam', etc. Note: IJCNN results with rmsprop
            self.REST_BEST_W = False
            self.MODEL_TYPE_START = 1            
            
        elif hostname=='SNOOPY': 
            self.CONF_NUMBER = 8
            self.FINE_TUN = False
            self.MULTI_OPTIMIZER = False
            self.OPTIMIZER = 'RMSprop'
            self.REST_BEST_W = False
            self.MODEL_TYPE_START = 24
            self.MODEL_TYPE_END = 24
        
        else:
            # Portable fallback: process only one architecture by default.
            # These values are not part of the published experiment profiles.
            self.CONF_NUMBER = 0
            self.FINE_TUN = False
            self.MULTI_OPTIMIZER = False
            self.OPTIMIZER = 'RMSprop'
            self.REST_BEST_W = False
            self.MODEL_TYPE_START = 1
            self.MODEL_TYPE_END = 1
            print("No host-specific VIA configuration for %s; using local defaults." % hostname)