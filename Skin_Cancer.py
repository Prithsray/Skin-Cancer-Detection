import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications import ConvNeXtLarge,ConvNeXtXLarge
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten,MaxPooling2D,Dropout,BatchNormalization
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
import keras
from sklearn.metrics import auc
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.preprocessing.image import  ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam, SGD,RMSprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt




plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 16

#Variable defining
SAMPLE_PER_CATEGORY = 1314
SEED = 42
WIDTH = 128
HEIGHT = 128
DEPTH = 3
INPUT_SHAPE = (WIDTH, HEIGHT, DEPTH)

data_dir = '/content/drive/MyDrive/data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
whole_dir= os.path.join(data_dir,'Train+test')

CATEGORIES = ['benign', 'malignant']
NUM_CATEGORIES = len(CATEGORIES)
NUM_CATEGORIES
for category in CATEGORIES:
    print('{} {} images'.format(category, len(os.listdir(os.path.join(train_dir, category)))))

train = []
for category_id, category in enumerate(CATEGORIES):
    for file in os.listdir(os.path.join(train_dir, category)):
        train.append(['train/{}/{}'.format(category, file), category_id, category])
train = pd.DataFrame(train, columns=['file', 'category_id', 'category'])
train.shape

#train.head(2)

train = pd.concat([train[train['category'] == c][:SAMPLE_PER_CATEGORY] for c in CATEGORIES])
train = train.sample(frac=1)
train.index = np.arange(len(train))
train.shape

#train

for category in CATEGORIES:
    print('{} {} images'.format(category, len(os.listdir(os.path.join(test_dir, category)))))

test = []
for category_id, category in enumerate(CATEGORIES):
    for file in os.listdir(os.path.join(test_dir, category)):
        test.append(['test/{}/{}'.format(category, file), category_id, category])
test = pd.DataFrame(test, columns=['file', 'category_id', 'category'])
#test.shape

#test

for category in CATEGORIES:
    print('{} {} images'.format(category, len(os.listdir(os.path.join(whole_dir, category)))))

whole_data = []
for category_id, category in enumerate(CATEGORIES):
    for file in os.listdir(os.path.join(whole_dir, category)):
        whole_data.append(['Train+test/{}/{}'.format(category, file), category_id, category])
whole_data = pd.DataFrame(whole_data, columns=['file', 'category_id', 'category'])
whole_data.shape

#whole_data

#whole_data.head(2)



def define_model():
	# load model
	model = ConvNeXtLarge(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(512, activation='relu', kernel_initializer='he_uniform')(flat1)
 
	class2 = BatchNormalization()(class1)
	class3 = Dense(512, activation='relu', kernel_initializer='he_uniform')(class2)
	class4 = Dense(512, activation='relu', kernel_initializer='he_uniform')(class3)
	class5 = Dense(512,activation='relu', kernel_initializer='he_uniform')(class4)
	class6 = Dense(512,activation='relu', kernel_initializer='he_uniform')(class5)
  
	#class2 = MaxPooling2D(31,31,32)(class3)
	output = Dense(2, activation='sigmoid')(class6)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = Adam(learning_rate=0.0009)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', keras.metrics.Precision(),keras.metrics.Recall(),keras.metrics.AUC(),keras.metrics.TruePositives(),keras.metrics.TrueNegatives(),keras.metrics.FalseNegatives(),keras.metrics.FalsePositives()])
	return model




def Cross_validation(images, epochs, cross_validation_folds):
    print("Train Model")
     
    datagen_train = ImageDataGenerator(featurewise_center= True,
                                       zoom_range=[0.5,1.0],
                                       featurewise_std_normalization=True,
                                       rotation_range=30, 
                                       fill_mode='nearest',
                                       width_shift_range=0.2, 
                                       height_shift_range=0.2,
                                       horizontal_flip=True, 
                                       vertical_flip=True)
    
    datagen_valid = ImageDataGenerator(featurewise_center= True,
                                       zoom_range=[0.5,1.0],
                                       featurewise_std_normalization=True,
                                       )
        
    print("Cross validation")
    kfold = StratifiedKFold(n_splits=cross_validation_folds, shuffle=True)
    cvscores = []
    iteration = 1
    
    t = images.category_id
    
    for train_index, test_index in kfold.split(np.zeros(len(t)), t):

        print("======================================")
        print("Iteration = ", iteration)

        iteration = iteration + 1

        train = images.loc[train_index]
        test = images.loc[test_index]

        print("======================================")
        
        model = define_model()

        print("======================================")
        
        train_generator = datagen_train.flow_from_dataframe(dataframe=train,
                                                  directory="/content/drive/MyDrive/data",
                                                  x_col="file",
                                                  y_col="category",
                                                  batch_size=64,
                                                  shuffle=True,
                                                  color_mode='rgb',
                                                  target_size=(224, 224));
        valid_generator=datagen_valid.flow_from_dataframe(dataframe=test,
                                                  directory="/content/drive/MyDrive/data",
                                                  x_col="file",
                                                  y_col="category",
                                                  batch_size=64,
                                                  shuffle=False,
                                                  color_mode='rgb',
                                                  target_size=(224, 224));
        
        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

        #Trains the model on data generated batch-by-batch by a Python generator
        history = model.fit_generator(train_generator, validation_data = valid_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_steps=STEP_SIZE_VALID, epochs=epochs,verbose=1)
        
        _,acc,precison,rcall,aucc,truepo,truene,falsepo,falsene = model.evaluate(valid_generator, steps=STEP_SIZE_VALID, verbose=0)

        ac= (truepo+truene)/(truene+truepo+falsene+falsepo)
        print("Accuarcy " , ac*100)
        cvscores.append(ac * 100)
        pre=truepo/(truepo+falsepo)
        print("Precision-> ",pre)
        recall=truepo/(truepo + falsene)
        print("Recall-> ",recall)
        specificity=truene/(truene+falsepo)
        print("Specificity -> ",specificity)
        f1= 2*pre*recall/(pre+recall)
        print("F1-Score -> ",f1)
        fpr= falsepo/(falsepo+truene)
        auc = (1/2)- (fpr/2) + (recall/2)
        print("AUC -> ",auc)

        
    accuracy = np.mean(cvscores);
    std = np.std(cvscores);
    print("Accuracy: " ,accuracy)
    return accuracy



'''
trainModelDF(
    whole_data,
    epochs=10,
    cross_validation_folds = 10,
)'''




datagen_train = ImageDataGenerator(featurewise_center= True,
                                    #zoom_range=[0.5,1.0],
                                   # featurewise_std_normalization=True,
                                    #rotation_range=30, 
                                    #fill_mode='nearest',
                                    #width_shift_range=0.2, 
                                    #height_shift_range=0.2,
                                    #horizontal_flip=True, 
                                    #vertical_flip=True
                                   )

datagen_valid = ImageDataGenerator(featurewise_center= True)

model = define_model()

    
train_generator = datagen_train.flow_from_dataframe(dataframe=train,
                                              directory="/content/drive/MyDrive/data",
                                              x_col="file",
                                              y_col="category",
                                              batch_size=64,
                                              shuffle=True,
                                              color_mode='rgb',
                                              target_size=(224,224));
valid_generator=datagen_valid.flow_from_dataframe(dataframe=test,
                                              directory="/content/drive/MyDrive/data",
                                              x_col="file",
                                              y_col="category",
                                              batch_size=64,
                                              shuffle=False,
                                              color_mode='rgb',
                                              target_size=(224, 224));
    
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

    #Trains the model on data generated batch-by-batch by a Python generator
history = model.fit_generator(train_generator, validation_data = valid_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_steps=STEP_SIZE_VALID, epochs=25,verbose=1)
    
_,acc,precison,rcall,aucc,truepo,truene,falsepo,falsene = model.evaluate(valid_generator, steps=STEP_SIZE_VALID, verbose=1)

ac= (truepo+truene)/(truene+truepo+falsene+falsepo)
print("Accuarcy " , ac*100)
pre=truepo/(truepo+falsepo)
print("Precision-> ",pre)
recall=truepo/(truepo + falsene)
print("Recall-> ",recall)
specificity=truene/(truene+falsepo)
print("Specificity -> ",specificity)
f1= 2*pre*recall/(pre+recall)
print("F1-Score -> ",f1)
fpr= falsepo/(falsepo+truene)
auc = (1/2)- (fpr/2) + (recall/2)
print("AUC -> ",auc)

