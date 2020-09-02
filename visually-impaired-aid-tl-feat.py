# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 20:17:15 2020

@author: fbrev
"""

import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
#print(os.listdir("dataset/"))

IMAGE_CHANNELS=3

POOLING = 'avg' # None, 'avg', 'max'

def load_data():
    
    filenames = os.listdir("dataset/")
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'clear':
            categories.append(1)
        else:
            categories.append(0)
    
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    return df

def create_model(model_type):
    # load model and preprocessing_function
    image_size = (224, 224)
    if model_type=='VGG16':
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        model = VGG16(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='VGG19':
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
        model = VGG19(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    else: print("Error: Model not implemented.")
    
    preprocessing_function = preprocess_input
    
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.models import Model

    output = Flatten()(model.layers[-1].output)   
    model = Model(inputs=model.inputs, outputs=output)

    return model, preprocessing_function, image_size

def extract_features(df, model, preprocessing_function, image_size):

    df["category"] = df["category"].replace({1: 'clear', 0: 'non-clear'}) 
           
    datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocessing_function
    )
    
    total = df.shape[0]
    batch_size = 4
    
    generator = datagen.flow_from_dataframe(
        df, 
        "dataset/", 
        x_col='filename',
        y_col='category',
        class_mode='categorical',
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    features = model.predict(generator, steps=np.ceil(total/batch_size))
    
    return features
   
# Main
    
model_type_list = ('VGG16', 'VGG19')

df = load_data()

np.savetxt("via-labels.txt", df.category, fmt="%s")
   
for model_type in model_type_list:    
    
    model, preprocessing_function, image_size = create_model(model_type)
    features = extract_features(df, model, preprocessing_function, image_size)
        
    # criar nome de arquivo vgg16.data e usa-lo
    filename = "via-" + model_type + "-data.txt"
    np.savetxt(filename, features, fmt="%s")