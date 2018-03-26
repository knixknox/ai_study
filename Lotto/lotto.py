# -*- coding: utf-8 -*-

import csv
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pylab as plt

lottoData = []

with open('lotto.csv', 'r') as fileHandle:
    fileObject = csv.reader(fileHandle)
    lottoStrList = list(fileObject)
    for row in lottoStrList:
        lottoData.append(list(map(int, row)))

lottoData = np.subtract(lottoData, 1)
lottoCatData = np_utils.to_categorical(lottoData)
lottoCatData = lottoCatData[:, 0, :] + lottoCatData[:, 1, :] + lottoCatData[:, 2, :]\
               + lottoCatData[:, 3, :] + lottoCatData[:, 4, :]

trainData = []
labelData = []
trainHistorySize = 50
for index in range(len(lottoCatData) - trainHistorySize - 1):
    trainData.append(lottoCatData[index:index + trainHistorySize].reshape(-1))
    labelData.append(lottoCatData[index + trainHistorySize + 1])

trainData = np.array(trainData)
labelData = np.array(labelData)

model = Sequential()
model.add(Dense(units=45*10, input_dim=45*trainHistorySize, activation='relu'))
model.add(Dense(units=45, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(trainData, labelData, epochs=100, batch_size=8)

predictInput = lottoCatData[len(lottoCatData) - trainHistorySize:len(lottoCatData)].reshape(1, 45*trainHistorySize)
result = model.predict(predictInput)
print(result)
print(np.unique(result, return_inverse=True))
plt.plot(result.transpose())

