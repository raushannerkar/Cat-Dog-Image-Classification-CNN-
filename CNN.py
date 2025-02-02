import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import matplotlib.pyplot as plt
import random


#Dataset Creation
#Loading the datasets
X_train = np.loadtxt('input.csv', delimiter=',')
Y_train = np.loadtxt('labels.csv', delimiter=',')
X_test = np.loadtxt('input_test.csv', delimiter=',')
Y_test = np.loadtxt('labels_test.csv', delimiter=',')

#Reshaping X_train and Y_train
X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)
X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)

#Dividing it by 255 to get smaller values
X_train = X_train / 255.0
Y_train = Y_train / 255.0

#Randomly plotting an image
#idx = random.randint(0, len(X_train))
#plt.imshow(X_train[idx, :])
#plt.show()


#Model Building: Type 1
model = Sequential([
    #First Argument: Number of filters
    #Second Argument: Filter size (3x3 matrix)
    #Third Argument: Activation Function
    #Fourth Argument: Size of the image in pixels along with the RGB status
    Conv2D(32, (3,3), activation='relu', input_shape = (100,100,3)),
    #Just mention filter size
    MaxPooling2D((2,2)),
    #Input shape is only for the first convo2D
    Conv2D(32, (3,3), activation='relu'),
    #No arguments
    Flatten(),
    #First Argument: Number of neurons
    Dense(64, activation='relu'),
    #This is the output layer that is why only 1 neuron
    Dense(1, activation='sigmoid')
])

#Compilation of the model
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics= ['accuracy'])

#Training the model
#model.fit(X_train, Y_train, epochs=10, batch_size=64)

#Model Evaluation
model.evaluate(X_test, Y_test)

#Making Predictions
idx2 = random.randint(0, len(Y_test))
plt.imshow(X_test[idx2, :])
plt.show()

y_pred = model.predict(X_test[idx2, :].reshape(1,100,100,3))
y_pred = y_pred > 0.5

if(y_pred == 0):
    pred = 'dog'
else:
    pred = 'cat'
print("The model has predicted: ", pred)

#Model Building: Type 2
'''
model = Sequential([
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (100,100,3)))
model.add(MaxPooling2D((2,2)))
model.add(Convo2D(32, (3,3), activation = 'relu'))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid))
])
'''