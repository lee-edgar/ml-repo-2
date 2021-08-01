from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses, Model
import cv2
import numpy as np
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import sklearn.metrics
import seaborn as sns
from keras.models import load_model
from tqdm.notebook import tqdm
from skimage.color import rgb2gray
from skimage.transform import resize
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.optimizers import *
import time
from tensorflow.keras.callbacks import * 
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.notebook import tqdm
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import *
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.applications import *
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet

def norm_digit(img):
    m = cv2.moments(img)
    cx = m['m10'] / m['m00']
    cy = m['m01'] / m['m00']
    h, w = img.shape[:2]
    aff = np.array([[1, 0, w/2 - cx], [0, 1, h/2 - cy]], dtype=np.float32)
    dst = cv2.warpAffine(img, aff, (0, 0))
    return dst

def img_preprocess(imgs):
  images = []
  bar_total = tqdm(imgs)
  for file in bar_total:
          
          img = cv2.imread(file, 0)
          img = np.where(img == 255, 0, img)
          img = norm_digit(img)
          clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
          cl1 = clahe.apply(img)
              
          images.append(resize(cl1, (224,224,3)))
          
  images = np.array(images)
  return images

def makelabel(labels):
  encoder = LabelEncoder()
  encoder.fit(labels)
  labels_encoded = encoder.transform(labels)
  labels_encoded[:3], encoder.classes_
  return labels_encoded

def Basic_work():
    total_dir = "/content/OCT_small"
    train_dir = "/content/OCT_small/train"
    test_dir = "/content/OCT_small/test"
    classes = os.listdir(train_dir)
    dataset = {
        "img_path" : [], 
        "class" : [], 
        "split" : []
    }
    for split in os.listdir(total_dir):
        for where in os.listdir(total_dir + "/" + split):
            for image in glob.glob(total_dir+"/"+split+"/"+where+"/*.jpeg"):
                dataset["img_path"].append(image)
                dataset["class"].append(where)
                dataset["split"].append(split)
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state = 77)

    train_df = dataset[dataset["split"] == "train"]
    test_df = dataset[dataset["split"] == "test"]
    valid_df = dataset[dataset["split"] == "validation"]

    X_train = train_df["img_path"]
    y_train = train_df["class"]

    X_test = test_df["img_path"]
    y_test = test_df["class"]

    X_val = valid_df["img_path"]
    y_val = valid_df["class"]

    X_train = img_preprocess(X_train)
    X_test = img_preprocess(X_test)
    X_val = img_preprocess(X_val)

    y_train = makelabel(y_train)
    y_test = makelabel(y_test)
    y_val = makelabel(y_val)
    return X_train, X_test, X_val, y_train, y_test, y_val


def Resnet50_model(IMG_SIZE):
    mymodel = ResNet50(weight="imagenet",
        include_top = False,
        input_shape = (IMG_SIZE, IMG_SIZE,3),
        pooling = "avg"
    )
    testmodel = models.Sequential()
    testmodel.add(mymodel)
    testmodel.add(Dense(4, activation="softmax"))
    testmodel.summary()
    
    return mymodel, testmodel

def MobileNet_model(IMG_SIZE):
    mymodel = MobileNet(weight="imagenet",
        include_top = False,
        input_shape = (IMG_SIZE, IMG_SIZE,3),
        pooling = "avg"
    )
    testmodel = models.Sequential()
    testmodel.add(mymodel)
    testmodel.add(Dense(4, activation="softmax"))
    testmodel.summary()
    
    return mymodel, testmodel


def create_model(model_name, img_size, batch_size):
    mymodel = model_name(weight="imagenet",
        include_top = False,
        input_shape = (img_size, img_size,3),
        pooling = "avg"
    )
    testmodel = models.Sequential()
    testmodel.add(mymodel)
    testmodel.add(Dense(4, activation="softmax"))
    testmodel.summary()

    optimizers = tf.keras.optimizers.Adam(learning_rate = 0.001)
    testmodel.compile(optimizer=optimizers,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    rlr_cb = ReduceLROnPlateau(monitor='val_accuracy',factor=0.3, patience=10, mode='max',verbose=1)
    ely_cb = EarlyStopping(monitor='val_accuracy', patience=20, mode='max', verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='min', baseline=None, restore_best_weights=False)

    start_time = time.time()
    hist=testmodel.fit(X_train, y_train.reshape(-1, 1),
                epochs=200, verbose=1, batch_size = 140,
                validation_data = (X_test, y_test.reshape(-1, 1)), shuffle=True,
                callbacks=[rlr_cb, ely_cb])
    acc = hist.history['accuracy'], val_acc = hist.history['val_accuracy'], loss = hist.history['loss'], val_loss = hist.history['val_loss']
    print('fit time : ', time.time() - start_time )

    
    return mymodel, testmodel, hist

def Performance_evaluation(y_val, pred):
    testmodel.evaluate(X_test, y_test)
    pred = testmodel.predict(X_val)
    pred = np.argmax(pred, axis=1)
    weighted = accuracy_score(y_val, pred), recall_score(y_val, pred, average='weighted'), precision_score(y_val, pred, average='weighted'), f1_score(y_val, pred, average='weighted')
    micro = accuracy_score(y_val, pred), recall_score(y_val, pred, average='micro'), precision_score(y_val, pred, average='micro'), f1_score(y_val, pred, average='micro')
    macro = accuracy_score(y_val, pred), recall_score(y_val, pred, average='macro'), precision_score(y_val, pred, average='macro'), f1_score(y_val, pred, average='macro')
    NoneAVG = accuracy_score(y_val, pred), recall_score(y_val, pred, average=None), precision_score(y_val, pred, average=None), f1_score(y_val, pred, average=None)
    confunsion = confusion_matrix(y_val, pred)
    return weighted, micro, macro, NoneAVG, confunsion


