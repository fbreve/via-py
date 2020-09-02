# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:00:58 2019

@author: fbrev

based on visually-impaired-aid-eval
cross-validation version

"""

import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
#print(os.listdir("dataset/"))

FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
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


def create_model():
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
    
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes
    
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    
    model.summary()
    
    return model

def train_test_model(train_df, test_df, model):
   
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

    train_df, validate_df = train_test_split(train_df, test_size=0.20)
    
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    #train_df['category'].value_counts().plot.bar()
    
    #validate_df['category'].value_counts().plot.bar()
    
    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]
    total_test = test_df.shape[0]
    batch_size=4
    
    train_datagen = ImageDataGenerator(
        #rotation_range=15,    
        rescale=1./255,
        #shear_range=0.1,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #width_shift_range=0.1,
        #height_shift_range=0.1
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        DATASET_PATH, 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        DATASET_PATH, 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    #import psutil
    
    epochs=3 if FAST_RUN else 50
    model.fit_generator(
        train_generator, 
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate/batch_size,
        #steps_per_epoch=total_train,
        steps_per_epoch=total_train/batch_size,
        callbacks=callbacks,
        #use_multiprocessing=True,
        #workers=psutil.cpu_count()
    )


    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
         test_df, 
         DATASET_PATH, 
         x_col='filename',
         y_col='category',
         class_mode='categorical',
         target_size=IMAGE_SIZE,
         batch_size=batch_size,
         shuffle=False
    )
        
    _, acc = model.evaluate_generator(test_generator, steps=np.ceil(total_test/batch_size))

    return acc
    
# Main
    
# Seed to make it reproducible
np.random.seed(seed=1980)

df = load_data()

model = create_model()

# save weights before training the model
ini_weights = model.get_weights()

# creating folds for cross-validation
from sklearn.model_selection import RepeatedKFold
kfold_n_splits = 10
kfold_n_repeats = 10
kf = RepeatedKFold(n_splits=kfold_n_splits, n_repeats=kfold_n_repeats)
kf.split(df)

# get hostname for log-files
import socket
hostname = socket.gethostname()

# create filenames
log_filename = "via-tl-" + hostname + ".log"
csv_filename = "via-tl-res-" + hostname + ".csv"

# vector to hold each fold accuracy
cvscores = []

# enumerate allow the usage of the index for prints
for index, [train, test] in enumerate(kf.split(df)):
    # set the weights to their initial state before each training
    model.set_weights(ini_weights)
    train_df = df.loc[train]
    test_df = df.loc[test]
    acc = train_test_model(train_df,test_df,model)
    cvscores.append(acc)
    
    # print results to screen
    print("\nFold: %i of %i Acc: %.2f%%" % (index+1, kfold_n_splits*kfold_n_repeats, (acc*100)))
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