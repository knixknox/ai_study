# -*- coding: utf-8 -*-

from keras.utils import np_utils
import numpy as np
import collections
import csv

def Load(fileName, trainSize = 10):
    lottoData = []

    with open(fileName, 'r') as fileHandle:
        fileObject = csv.reader(fileHandle)
        lottoStrList = list(fileObject)
        for row in lottoStrList:
            lottoData.append(list(map(int, row)))

    lottoData = np.subtract(lottoData, 1)
    lottoCatData = np_utils.to_categorical(lottoData)
    lottoCatData = lottoCatData[:, 0, :] + lottoCatData[:, 1, :] + lottoCatData[:, 2, :] \
                   + lottoCatData[:, 3, :] + lottoCatData[:, 4, :] + lottoCatData[:, 5, :]

    trainData = []
    labelData = []
    for index in range(len(lottoCatData) - trainSize):
        trainData.append(lottoCatData[index:index + trainSize].reshape(-1))
        labelData.append(lottoCatData[index + trainSize])

    trainData = np.array(trainData)
    labelData = np.array(labelData)
    inputData = lottoCatData[len(lottoCatData) - trainSize:len(lottoCatData)].reshape(1, 45*trainSize)
    returnData = collections.namedtuple('returnData', ['trainData', 'labelData', 'inputData'])
    return returnData(trainData, labelData, inputData)


if __name__ == '__main__':
    trainSize = 10
    data = Load('lotto.csv', trainSize)

    with open('traindata.csv', 'w', newline='') as fileHandle:
        fileObject = csv.writer(fileHandle)
        for row in data.trainData:
            fileObject.writerow(row)

    with open('labeldata.csv', 'w', newline='') as fileHandle:
        fileObject = csv.writer(fileHandle)
        for row in data.labelData:
            fileObject.writerow(row)

    with open('inputdata.csv', 'w', newline='') as fileHandle:
        fileObject = csv.writer(fileHandle)
        for row in data.inputData:
            fileObject.writerow(row)

