import pandas as pd
import numpy as np
np.random.seed(2019) #important to set the seed before importing keras

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import seaborn as sns



#load the dataset
df = pd.read_csv("HR_comma_sep.csv")

df.rename(columns={'sales':'department'},inplace=True)


df = pd.get_dummies(df,columns=['department','salary'])

x,y = df.drop('left',axis=1).values, df.left.values

# let's do a training-test split for validation later on

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=2017)


# let's convert our output variable into categorical format for keras

num_classes = np.max(y_train)+1
y_train = to_categorical(y_train,num_classes)
y_test = to_categorical(y_test,num_classes)


n_cols = x_train.shape[1]

#set-up early-stopping monitor
early_stopping_monitor = EarlyStopping(patience=5)


model = Sequential()
model.add(Dense(50,activation='relu',input_shape=(n_cols,)))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))

#compile
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#fit
history = model.fit(x_train,y_train,epochs=20,verbose=1,
                    validation_split=0.2,callbacks=[early_stopping_monitor],
                    shuffle=False)

model.summary()

#plot training and validation los
plt.plot(history.history['loss'],'r',label='training')
plt.plot(history.history['val_loss'],'b',label='validation')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

#plot training and validation accuracy
plt.figure(figsize=(8,7))
plt.plot(history.history['acc'],'r',label='training')
plt.plot(history.history['val_acc'],'b',label='validation')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

# so we see that our baseline model has quite good accuracy just after 20 epochs

#let's evaluate our model to predict on our hold-out data

print "Model Evaluation on test dataset [loss,accuracy] ",model.evaluate(x_test,y_test)
