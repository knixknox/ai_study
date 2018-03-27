# -*- coding: utf-8 -*-

import csv
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
import matplotlib.pylab as plt

trainHistorySize = 50
lottoData = []

with open('lotto.csv', 'r') as fileHandle:
    fileObject = csv.reader(fileHandle)
    lottoStrList = list(fileObject)
    for row in lottoStrList:
        lottoData.append(list(map(int, row)))

lottoData = np.subtract(lottoData, 1)
lottoCatData = np_utils.to_categorical(lottoData)
lottoCatData = lottoCatData[:, 0, :] + lottoCatData[:, 1, :] + lottoCatData[:, 2, :]\
               + lottoCatData[:, 3, :] + lottoCatData[:, 4, :] + lottoCatData[:, 5, :]

with open('catdata.csv', 'w', newline='') as fileHandle:
    fileObject = csv.writer(fileHandle)
    for row in lottoCatData:
        fileObject.writerow(row)
        
trainData = []
labelData = []
for index in range(len(lottoCatData) - trainHistorySize):
    trainData.append(lottoCatData[index:index + trainHistorySize].reshape(-1))
    labelData.append(lottoCatData[index + trainHistorySize])

trainData = np.array(trainData)
labelData = np.array(labelData)
inputData = lottoCatData[len(lottoCatData) - trainHistorySize:len(lottoCatData)].reshape(1, 45*trainHistorySize)

with open('traindata.csv', 'w', newline='') as fileHandle:
    fileObject = csv.writer(fileHandle)
    for row in trainData:
        fileObject.writerow(row)

with open('labeldata.csv', 'w', newline='') as fileHandle:
    fileObject = csv.writer(fileHandle)
    for row in labelData:
        fileObject.writerow(row)

with open('inputdata.csv', 'w', newline='') as fileHandle:
    fileObject = csv.writer(fileHandle)
    for row in inputData:
        fileObject.writerow(row)

model = Sequential()
model.add(Dense(units=450, input_dim=45*trainHistorySize, activation='relu'))
model.add(Dense(units=450, activation='relu'))
model.add(Dense(units=45, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainData, labelData, epochs=100, batch_size=100)
model.save('mymodel.h5')

result = model.predict(inputData)
ranking = 45 - np.unique(result, return_inverse=True)[1]

for num in range(45):
    print('{0:2d} :\t{1:3.7f}%\t-> {2:2d}'.format(num + 1, result[0, num] * 100.0, ranking[num]))
    # print(num + 1, ":\t", result[0, num], '\t->', ranking[num])

plt.plot(result.transpose())

