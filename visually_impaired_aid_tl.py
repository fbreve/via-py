# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:00:58 2019

@author: fbrev

based on visually-impaired-aid-cv
transfer learning version

Required packages:
    tensorflow
    pandas
    scikit-learn
    pillow
    tensorflow-addons
"""

import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import random
from multiprocessing import Process, Queue
#print(os.listdir("../via-dataset/images/"))

from visually_impaired_aid_tl_config import via_config
conf = via_config()

os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

def load_data():
    
    filenames = os.listdir(conf.DATASET_PATH)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'clear':
            categories.append(0)
        else:
            categories.append(1)
    
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    return df

def create_model(model_type):
    model_dic = {
        'VGG16': ['vgg16', 224, 224],
        'VGG19': ['vgg19', 224, 224],
        'Xception': ['xception', 299, 299],
        'ResNet50': ['resnet', 224, 224],
        'ResNet101': ['resnet', 224, 224],
        'ResNet152': ['resnet', 224, 224],
        'ResNet50V2': ['resnet_v2', 224, 224],
        'ResNet101V2': ['resnet_v2', 224, 224],
        'ResNet152V2': ['resnet_v2', 224, 224],
        'InceptionV3': ['inception_v3', 299, 299],
        'InceptionResNetV2': ['inception_resnet_v2', 299, 299],
        'MobileNet': ['mobilenet', 224, 224],
        'DenseNet121': ['densenet', 224, 224],
        'DenseNet169': ['densenet', 224, 224],
        'DenseNet201': ['densenet', 224, 224],
        'NASNetLarge': ['nasnet', 331, 331],
        'NASNetMobile': ['nasnet', 224, 224],
        'MobileNetV2': ['mobilenet_v2', 224, 224],
        'EfficientNetB0': ['efficientnet', 224, 224],
        'EfficientNetB1': ['efficientnet', 240, 240],
        'EfficientNetB2': ['efficientnet', 260, 260],
        'EfficientNetB3': ['efficientnet', 300, 300],
        'EfficientNetB4': ['efficientnet', 380, 380],
        'EfficientNetB5': ['efficientnet', 456, 456],
        'EfficientNetB6': ['efficientnet', 528, 528],
        'EfficientNetB7': ['efficientnet', 600, 600],
    }
    
    model_module = getattr(tf.keras.applications,model_dic[model_type][0])
    model_function = getattr(model_module,model_type)
    image_size = tuple(model_dic[model_type][1:])
    model = model_function(weights='imagenet', include_top=False, pooling=conf.POOLING, input_shape=image_size + (conf.IMAGE_CHANNELS,))
    preprocessing_function = getattr(model_module,'preprocess_input')
    
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from tensorflow.keras.models import Model
    
    # mark loaded layers as not trainable
    if conf.FINE_TUN == False:
        for layer in model.layers:
            layer.trainable = False
	# add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    
    initializer = tf.keras.initializers.HeUniform(seed=SEED)
 
    if conf.ALT_DENSE==False:
        class1 = Dense(128, activation='relu', kernel_initializer=initializer)(flat1)
    else:
        dense1 = Dense(512, activation='relu', kernel_initializer=initializer)(flat1)
        dropout1 = Dropout(0.25)(dense1)
        dense2 = Dense(512, activation='relu', kernel_initializer=initializer)(dropout1)
        class1 = Dropout(0.25)(dense2)

    output = Dense(2, activation='softmax')(class1)
    
    # define new model
    model = Model(inputs=model.inputs, outputs=output, name=model_type)    
	
    compile_model(model)
    
    model.summary()
    
    return model, preprocessing_function, image_size

def compile_model(model):
    
    # compile model

    opt_func = getattr(tf.keras.optimizers, conf.OPTIMIZER)
       
    if conf.MULTI_OPTIMIZER==True:
        optimizers = [
            opt_func(learning_rate=1e-5),
            opt_func(learning_rate=1e-3)
            ]
        optimizers_and_layers = [(optimizers[0], model.layers[0:-4]), (optimizers[1], model.layers[-4:])]
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    else:    
        optimizer = opt_func(learning_rate=1e-3)   
        # note: IJCNN results didn't have a learning rate set
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

def train_test_model(train_df, test_df, model, preprocessing_function, image_size, split, batch_size):
   
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    earlystop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=conf.REST_BEST_W
    )
   
    if conf.MULTI_OPTIMIZER==True:            
        callbacks = [earlystop]
    else:
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)        
        callbacks = [earlystop, learning_rate_reduction]
              
    train_df["category"] = train_df["category"].replace({0: 'clear', 1: 'non-clear'}) 
    test_df["category"] = test_df["category"].replace({0: 'clear', 1: 'non-clear'})        

    train_df, validate_df = train_test_split(train_df, test_size=0.20, shuffle=True,
                                             stratify=train_df.category, random_state=SEED)
    
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    #train_df['category'].value_counts().plot.bar()
    
    #validate_df['category'].value_counts().plot.bar()
    
    #total_train = train_df.shape[0]
    #total_validate = validate_df.shape[0]
    #total_test = test_df.shape[0]
       
    if conf.DATA_AUG==True:
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
        
    validation_datagen = ImageDataGenerator(
        #rescale=1./255,
        preprocessing_function=preprocessing_function
    )        
       
    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        conf.DATASET_PATH, 
        x_col='filename',
        y_col='category',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,
        seed=SEED,
    )
    
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        conf.DATASET_PATH, 
        x_col='filename',
        y_col='category',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,    
        seed=SEED,
    )

    #import psutil
    
    epochs=3 if conf.FAST_RUN else 50

    model.fit(
        train_generator, 
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=2,
    )


    test_datagen = ImageDataGenerator(
        #rescale=1./255,
        preprocessing_function=preprocessing_function
    )
    test_generator = test_datagen.flow_from_dataframe(
         test_df, 
         conf.DATASET_PATH, 
         x_col='filename',
         y_col='category',
         class_mode='categorical',
         target_size=image_size,
         batch_size=batch_size,
         shuffle=False,
         seed=SEED,
    )
    
    #_, acc = model.evaluate(test_generator, steps=np.ceil(total_test/batch_size))
    
    predictions = model.predict(
        test_generator,
        verbose=2,
    )    
    
    # save predictions to file inside the 'pred' subfolder   
    pred_filename = "./pred/conf" + str(conf.CONF_NUMBER) +  "/pred-" + model.name + "-split=" + str(split) + ".csv"
    os.makedirs(os.path.dirname(pred_filename), exist_ok=True)
    np.savetxt(pred_filename, predictions, delimiter=',')    
    
    # ground truth
    y_true = test_df["category"].replace({'clear':0, 'non-clear':1})
    # get hard labels from the predictions
    y_pred = np.argmax(predictions,1)
    # calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    acc = np.trace(cm) / np.sum(cm) #accuracy

    return acc

def run_tf(q, model_type, train_df, test_df, index, batch_size):

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)    

    try:    
        model, preprocessing_function, image_size = create_model(model_type)
        acc = train_test_model(train_df,test_df,model,preprocessing_function,image_size, index+1, batch_size)
        q.put(acc)
    except:
        q.put(-1)
        import sys
        sys.exit(1)        

# Main
if __name__ == "__main__":
    
    # get hostname for log-files
    import socket
    hostname = socket.gethostname()
    
    # create filenames
    log_filename = "via-tl-conf" + str(conf.CONF_NUMBER) + '-' + hostname + ".log"
    csv_filename = "via-tl-conf" + str(conf.CONF_NUMBER) + '-' + hostname + ".csv"
    
    # write log header
    with open(log_filename,"a+") as f_log:
        f_log.write("Machine: %s\n" % hostname)
        from datetime import datetime
        now = datetime.now()
        f_log.write(now.strftime("Date: %d/%m/%Y Time: %H:%M:%S\n"))
        f_log.write("Config. Number: %i\n" % conf.CONF_NUMBER)
        f_log.write("Alternative Dense Layer: %s\n" % conf.ALT_DENSE)
        f_log.write("Pooling Application Layer: %s\n" % conf.POOLING)
        f_log.write("Data Augmentation: %s\n" % conf.DATA_AUG)  
        f_log.write("Data Augmentation Multiplier: %s\n" % conf.DATA_AUG_MULT)
        f_log.write("Fine Tuning: %s\n" % conf.FINE_TUN)
        f_log.write("Multi Optimizer: %s\n" % conf.MULTI_OPTIMIZER)
        f_log.write("Optimizer: %s\n" % conf.OPTIMIZER)    
        f_log.write("Batch Size: %s\n" % conf.BATCH_SIZE)
        f_log.write("Restore Best Weights: %s\n\n" % conf.REST_BEST_W)
           
    df = load_data()

    model_type_list = conf.MODEL_TYPE_LIST[conf.MODEL_TYPE_START-1:conf.MODEL_TYPE_END]

    for model_type in model_type_list:                  
        
        # creating folds for cross-validation
        from sklearn.model_selection import RepeatedKFold
        kfold_n_splits = 10
        kfold_n_repeats = 5
        kf = RepeatedKFold(n_splits=kfold_n_splits, n_repeats=kfold_n_repeats, random_state=SEED)
        kf.split(df)
        
        # record model type in the log file
        with open(log_filename,"a+") as f_log:
            f_log.write("Model Type: %s\n" % model_type)            
            
        # vector to hold each fold accuracy
        cvscores = []

        batch_size = conf.BATCH_SIZE
        
        # enumerate allow the usage of the index for prints
        for index, [train, test] in enumerate(kf.split(df)):    
            train_df = df.loc[train]
            test_df = df.loc[test]            
            # Since TensorFlow is leaking memory and crashing randomly, I decided
            # to run it inside a process until it is successful. 
            err_count = 0
            while True:
                # the queue is used to get the return
                q = Queue()
                # start the process to run TensorFlow
                p = Process(target=run_tf, args=(q, model_type, train_df, test_df, index, batch_size))
                p.start()
                # get the return from the process
                acc = q.get()
                p.join()
                # check for abnormal process termination
                if p.exitcode != 0:                                
                    err_count = err_count + 1
                    # trying up to three times
                    if err_count < 3:
                        print("TensorFlow crashed. Retrying...\n")
                        with open(log_filename,"a+") as f_log:
                            f_log.write("TensorFlow crashed. Retrying...\n")
                    # after three times, lower batch_size
                    else:
                        if batch_size > 1:
                            err_count = 0
                            batch_size = round(batch_size / 2)
                            print("TensorFlow crashed. Retrying with batch_size=%i.\n" % batch_size)
                            with open(log_filename,"a+") as f_log:
                                f_log.write("TensorFlow crashed. Retrying with batch_size=%i.\n" % batch_size)
                        # after three fails with batch_size = 1, we give up on this model
                        else:
                            print("Giving up on this model...\n")
                            with open(log_filename,"a+") as f_log:
                                f_log.write("Giving up on this model...\n")
                            err_count = -1
                            break
                    continue
                else:
                    break
                
            # checkin if we give up on this model and going to the next one.
            if err_count == -1:
                break
                       
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