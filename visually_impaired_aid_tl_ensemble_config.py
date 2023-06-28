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
        
        import socket
        hostname = socket.gethostname()
        
        if hostname=='DONALD':
            self.MODEL_TYPE = 'InceptionV3'
            self.CONF = 'E'
            self.FINE_TUN = True
            self.MULTI_OPTIMIZER = True
            self.OPTIMIZER = 'RMSprop' # 'RMSprop', 'Adam', etc. Note: IJCNN results with rmsprop
            self.REST_BEST_W = False
            
        elif hostname=='PRECISION':
            self.MODEL_TYPE = 'MobileNet'
            self.CONF = 'F'
            self.FINE_TUN = True
            self.MULTI_OPTIMIZER = True
            self.OPTIMIZER = 'Adam' # 'RMSprop', 'Adam', etc. Note: IJCNN results with rmsprop
            self.REST_BEST_W = False
            
        elif hostname=='SNOOPY': 
            self.MODEL_TYPE = 'MobileNet'
            self.CONF = 'B'
            self.FINE_TUN = False
            self.MULTI_OPTIMIZER = False
            self.OPTIMIZER = 'Adam'
            self.REST_BEST_W = False
        
        else:
            print("ERROR: There is no configuration defined for this host.")        
            import sys
            sys.exit()