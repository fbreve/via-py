# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:00:58 2019

@author: fbrev

based on visually-impaired-aid-tl-finetune

Required packages:
    - numpy
    - pandas
    - tensorflow / tensorflow-gpu
    - pillow
    - scikit-learn
"""

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
import random
#import psutil
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

#FAST_RUN = False
#IMAGE_WIDTH=128
#IMAGE_HEIGHT=128
#IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

ALT_DENSE = False # True, False
POOLING = 'avg' # None, 'avg', 'max'
DATA_AUG = False # True, False
N_REPEATS = 5
BATCH_SIZE = 16
MULTI_OPTIMIZER = True

def create_model(model_type):
    # load model and preprocessing_function    
    if model_type=='VGG16':
        image_size = (224, 224)
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        model = VGG16(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='VGG19':
        image_size = (224, 224)
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
        model = VGG19(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='Xception':
        image_size = (299, 299)
        from tensorflow.keras.applications.xception import Xception, preprocess_input
        model = Xception(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet50':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
        model = ResNet50(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet101':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
        model = ResNet101(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='ResNet152':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
        model = ResNet152(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))    
    elif model_type=='ResNet50V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
        model = ResNet50V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet101V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input
        model = ResNet101V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='ResNet152V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
        model = ResNet152V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='InceptionV3':
        image_size = (299, 299)
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        model = InceptionV3(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='InceptionResNetV2':
        image_size = (299, 299)
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
        model = InceptionResNetV2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='MobileNet':
        image_size = (224, 224)
        from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
        model = MobileNet(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))       
    elif model_type=='DenseNet121':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
        model = DenseNet121(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
    elif model_type=='DenseNet169':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
        model = DenseNet169(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,)) 
    elif model_type=='DenseNet201':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
        model = DenseNet201(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
    elif model_type=='NASNetLarge':
        image_size = (331, 331)
        from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
        model = NASNetLarge(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
    elif model_type=='NASNetMobile':
        image_size = (224, 224)
        from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input
        model = NASNetMobile(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
    elif model_type=='MobileNetV2':
        image_size = (224, 224)
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input        
        model = MobileNetV2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))                 
    elif model_type=='EfficientNetB0':
        image_size = (224, 224)
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
        model = EfficientNetB0(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB1':
        image_size = (240, 240)
        from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input
        model = EfficientNetB1(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB2':
        image_size = (260, 260)
        from tensorflow.keras.applications.efficientnet import EfficientNetB2, preprocess_input
        model = EfficientNetB2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB3':
        image_size = (300, 300)
        from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
        model = EfficientNetB3(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB4':
        image_size = (380, 380)
        from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
        model = EfficientNetB4(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB5':
        image_size = (456, 456)
        from tensorflow.keras.applications.efficientnet import EfficientNetB5, preprocess_input
        model = EfficientNetB5(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB6':
        image_size = (528, 528)
        from tensorflow.keras.applications.efficientnet import EfficientNetB6, preprocess_input
        model = EfficientNetB6(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB7':
        image_size = (600, 600)
        from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
        model = EfficientNetB7(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    else: print("Error: Model not implemented.")

    preprocessing_function = preprocess_input
    
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from tensorflow.keras.models import Model
                 
	# add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
 
    initializer = tf.keras.initializers.HeUniform(seed=SEED)
    if ALT_DENSE==False:
        class1 = Dense(128, activation='relu', kernel_initializer=initializer)(flat1)
    else:
        dense1 = Dense(256, activation='relu', kernel_initializer=initializer)(flat1)
        class1 = Dropout(0.2)(dense1)

    output = Dense(2, activation='softmax')(class1)
   
    # define new model
    model = Model(inputs=model.inputs, outputs=output)

    # compile model
    
    if MULTI_OPTIMIZER==True:
        optimizers = [
            tf.keras.optimizers.Adam(learning_rate=1e-5),
            tf.keras.optimizers.Adam(learning_rate=1e-3)
            ]
        optimizers_and_layers = [(optimizers[0], model.layers[0:-4]), (optimizers[1], model.layers[-4:])]
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    else:    
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()
       
    return model, preprocessing_function, image_size
    
# Main

model_type_list = ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152',  
                   'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'InceptionV3', 'InceptionResNetV2',
                   'MobileNet', 'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile',
                   'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                   'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']

for model_index, model_type in enumerate(model_type_list):
    model, preprocessing_function, image_size = create_model(model_type)
    
    print(model_type)    
    
    wait = input("Press Enter to continue.")
    