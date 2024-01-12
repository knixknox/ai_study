# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import matplotlib.pylab as plt
import lotto_data


trainSize = 100
data = lotto_data.Load('lotto.csv', trainSize)

model = Sequential()
model.add(Dense(units=4500, input_dim=45*trainSize, activation='relu'))
model.add(Dense(units=4500, activation='relu'))
model.add(Dense(units=45, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data.trainData, data.labelData, epochs=10000, batch_size=50)
model.save('mymodel.h5')

result = model.predict(data.inputData)
ranking = 45 - np.unique(result, return_inverse=True)[1]

for num in range(45):
    print('{0:2d} :\t{1:3.7f}%\t-> {2:2d}'.format(num + 1, result[0, num] * 100.0, ranking[num]))

plt.plot(result.transpose())

