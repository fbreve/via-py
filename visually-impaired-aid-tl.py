# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:00:58 2019

@author: fbrev

based on visually-impaired-aid-cv
transfer learning version

"""

import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import random
#print(os.listdir("../via-dataset/images/"))

os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

FAST_RUN = False
#IMAGE_WIDTH=128
#IMAGE_HEIGHT=128
#IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

ALT_DENSE = False # True, False
POOLING = 'avg' # None, 'avg', 'max'
DATA_AUG = False # True, False
DATA_AUG_MULT = 1 # >=1
FINE_TUN = True
OPTIMIZER = 'rmsprop' # 'rmsprop', 'adam', 'sgd', etc.
DATASET_PATH = "../via-dataset/images/"

def load_data():
    
    filenames = os.listdir(DATASET_PATH)
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
    
    # mark loaded layers as not trainable
    if FINE_TUN == False:
        for layer in model.layers:
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
    
    return model, preprocessing_function, image_size

def train_test_model(train_df, test_df, model, preprocessing_function, image_size):
   
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    earlystop = EarlyStopping(patience=10)
    
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

    callbacks = [earlystop, learning_rate_reduction]

    train_df["category"] = train_df["category"].replace({1: 'clear', 0: 'non-clear'}) 
    test_df["category"] = test_df["category"].replace({1: 'clear', 0: 'non-clear'})        

    train_df, validate_df = train_test_split(train_df, test_size=0.20, shuffle=True,
                                             stratify=train_df.category, random_state=SEED)
    
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    #train_df['category'].value_counts().plot.bar()
    
    #validate_df['category'].value_counts().plot.bar()
    
    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]
    total_test = test_df.shape[0]
    batch_size=4
       
    if DATA_AUG==True:
        train_datagen = ImageDataGenerator(
            rotation_range=15,    
            #rescale=1./255,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            preprocessing_function=preprocessing_function
        )
    else:
        train_datagen = ImageDataGenerator(
            #rescale=1./255,
            preprocessing_function=preprocessing_function
        )
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        DATASET_PATH, 
        x_col='filename',
        y_col='category',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,
    )
    
    validation_datagen = ImageDataGenerator(
        #rescale=1./255,
        preprocessing_function=preprocessing_function
    )
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        DATASET_PATH, 
        x_col='filename',
        y_col='category',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,        
    )

    #import psutil
    
    epochs=3 if FAST_RUN else 50
    model.fit(
        train_generator, 
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate/batch_size,
        #steps_per_epoch=total_train,
        steps_per_epoch= total_train * DATA_AUG_MULT / batch_size,
        callbacks=callbacks
        #use_multiprocessing=True,
        #workers=psutil.cpu_count()
    )


    test_datagen = ImageDataGenerator(
        #rescale=1./255,
        preprocessing_function=preprocessing_function
    )
    test_generator = test_datagen.flow_from_dataframe(
         test_df, 
         DATASET_PATH, 
         x_col='filename',
         y_col='category',
         class_mode='categorical',
         target_size=image_size,
         batch_size=batch_size,
         shuffle=False
    )
        
    _, acc = model.evaluate(test_generator, steps=np.ceil(total_test/batch_size))

    return acc
    
# Main
    
model_type_list = ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 
        'ResNet152','ResNet50V2', 'ResNet101V2', 'ResNet152V2', "InceptionV3",
        'InceptionResNetV2', 'MobileNet', 'DenseNet121', 'DenseNet169',
        'DenseNet201', 'NASNetMobile', 'MobileNetV2',
        'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 
        'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
        'EfficientNetB6', 'EfficientNetB7', ]

         # NASNetLarge was removed because it is apparently too much for our GPU (GTX 1080 Ti)

# get hostname for log-files
import socket
hostname = socket.gethostname()

# create filenames
log_filename = "via-tl-" + hostname + ".log"
csv_filename = "via-tl-" + hostname + ".csv"

# write log header
with open(log_filename,"a+") as f_log:
    f_log.write("Machine: %s\n" % hostname)
    from datetime import datetime
    now = datetime.now()
    f_log.write(now.strftime("Date: %d/%m/%Y Time: %H:%M:%S\n"))
    f_log.write("Alternative Dense Layer: %s\n" % ALT_DENSE)
    f_log.write("Pooling Application Layer: %s\n" % POOLING)
    f_log.write("Data Augmentation: %s\n" % DATA_AUG)  
    f_log.write("Data Augmentation Multiplier: %s\n" % DATA_AUG_MULT)
    f_log.write("Fine Tuning: %s\n" % FINE_TUN)
    f_log.write("Optimizer: %s\n\n" %OPTIMIZER)    
   
for model_type in model_type_list:
       
    df = load_data()
    
    model, preprocessing_function, image_size = create_model(model_type)
    
    # save weights before training the model
    ini_weights = model.get_weights()
    
    # creating folds for cross-validation
    from sklearn.model_selection import RepeatedKFold
    kfold_n_splits = 10
    kfold_n_repeats = 1
    kf = RepeatedKFold(n_splits=kfold_n_splits, n_repeats=kfold_n_repeats, random_state=SEED)
    kf.split(df)
    
    # record model type in the log file
    with open(log_filename,"a+") as f_log:
        f_log.write("Model Type: %s\n" % model_type)            
        
    # vector to hold each fold accuracy
    cvscores = []
    
    # enumerate allow the usage of the index for prints
    for index, [train, test] in enumerate(kf.split(df)):
        # set the weights to their initial state before each training
        model.set_weights(ini_weights)
        train_df = df.loc[train]
        test_df = df.loc[test]
        acc = train_test_model(train_df,test_df,model,preprocessing_function,image_size)
        cvscores.append(acc)
        
        # print results to screen
        print("\nModel: %s Fold: %i of %i Acc: %.2f%%" % (model_type, index+1, kfold_n_splits*kfold_n_repeats, (acc*100)))
        print("Mean: %.2f%% (+/- %.2f%%)\n" % (np.mean(cvscores)*100, np.std(cvscores)*100))
        
        #record log file
        with open(log_filename,"a+") as f_log:
            f_log.write("Fold: %i of %i Acc: %.2f%% Mean: %.2f%% (+/- %.2f%%)\n" % (
            index+1, kfold_n_splits*kfold_n_repeats, (acc*100), np.mean(cvscores)*100, np.std(cvscores)*100)) 
        
        #record individual results to csv file
        with open(csv_filename,"a+") as f_csv:
            f_csv.write("%.4f" % acc)
            if index+1<kfold_n_splits*kfold_n_repeats: f_csv.write(", ")
            else: f_csv.write("\n")