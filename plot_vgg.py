# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 00:30:13 2020

@author: fbrev
"""

ALT_DENSE = True # True, False
POOLING = 'avg' # None, 'avg', 'max'
DATA_AUG = False # True, False
DATA_AUG_MULT = 1 # >=1
FINE_TUN = 2 # 0, 1, 2, 3 (amount of trainable blocks from the output)
OPTIMIZER = 'sgd' # 'rmsprop', 'adam', 'sgd', etc.
MODEL_TYPE = 'VGG16'
IMAGE_CHANNELS = 3

image_size = (224, 224)
if MODEL_TYPE=='VGG16':
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    model = VGG16(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
elif MODEL_TYPE=='VGG19':
    from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
    model = VGG19(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
else: print("Error: Model not implemented.")

preprocessing_function = preprocess_input

from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model

# mark loaded layers as not trainable (except where fine-tuning is enabled)
for layer in model.layers:
    if FINE_TUN > 0 and 'block5' in layer.name:
        layer.trainable = True
    if FINE_TUN > 1 and 'block4' in layer.name:
        layer.trainable = True
    if FINE_TUN > 2 and 'block3' in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False

	# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
 
if ALT_DENSE==False:
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
else:
    dense1 = Dense(512, activation='relu', kernel_initializer='he_uniform')(flat1)
    dropout1 = Dropout(0.25)(dense1)
    dense2 = Dense(512, activation='relu', kernel_initializer='he_uniform')(dropout1)
    class1 = Dropout(0.25)(dense2)

output = Dense(2, activation='softmax')(class1)
   
# define new model
model = Model(inputs=model.inputs, outputs=output)
# compile model
model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])
	
model.summary()

from tensorflow.keras.utils import plot_model
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False, rankdir='TB', dpi=300)