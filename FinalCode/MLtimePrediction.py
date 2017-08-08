"""
Created on Mon Jul 31 08:54:16 2017

@author: Eric Lin
Naive Bayes for station probability distribution.
"""
import os
import pickle

from datetime import datetime
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import numpy as np

os.chdir('C:/Users/GGSAdminLab/Documents/Eric Lin/Metro Data Project/FinalCode')


'''
Weather
'''
pickle_weather_in = open('meterologyJantoJun.pickle', 'rb')
weather_day_lst = pickle.load(pickle_weather_in)    #index 2 is avg temp, 5 is avg dew pt, 8 is avg humidity, 11 is avg sea lvl press., 14 is avg visibility, 17 is avg wind, 19 is precip., 20 is events

weatherDayLst_avg = []
for month in weather_day_lst:
    tempMonth = []
    for i in month:
        tempLst = [i[2], i[5], i[8], i[11], i[14], i[17], i[19]]
        if 'Rain' in i[20] and 'Snow' in i[20] and 'Thunderstorm' in i[20]:             #0 is nothing, 1 is rain, 2 is fog, 3 is Thunderstorm, 4 is snow, 5 is rain & thunderstorm, 6 is fog, rain, & snow, 7 is rain, snow, & thunderstorm, 
            tempLst.append(7)    
        elif 'Rain' in i[20] and 'Snow' in i[20] and 'Fog' in i[20]:
            tempLst.append(6)
        elif 'Rain' in i[20] and 'Thunderstorm' in i[20]:
            tempLst.append(5)
        elif 'Snow' in i[20]:
            tempLst.append(4)
        elif 'Thunderstorm' in i[20]:
            tempLst.append(3)
        elif 'Fog' in i[20]:
            tempLst.append(2)
        elif 'Rain' in i[20]:
            tempLst.append(1)
        else:
            tempLst.append(0)            
        tempMonth.append(tempLst)
    weatherDayLst_avg.append(tempMonth)
        

pickle_weather_in .close()


'''
Days
'''
dayTypeLst = [[0]*31, [0]*28, [0]*31, [0]*30, [0]*31, [0]*30]
for month in range(len(dayTypeLst)):
    for day in range(len(dayTypeLst[month])):
        tempDateTime = datetime(2017, month + 1, day + 1)
        if tempDateTime.weekday() >= 5:      #if weekend
            dayTypeLst[month][day] = 1
dayTypeLst[0][0] = 3    #New Year's
dayTypeLst[0][1] = 3    #New Year's
dayTypeLst[0][15] = 3   #MLK
dayTypeLst[0][19] = 3   #Inauguration
dayTypeLst[0][20] = 3   #Women's March
dayTypeLst[1][19] = 3   #George Washington's Birthday
dayTypeLst[4][28] = 3   #Memorial Day


'''
Station Clusters
'''
stationClusterString = '0 0 4 5 7 0 3 1 1 7 0 2 1 7 7 7 7 0 7 3 0 1 3 1 7 1 4 4 2 4 4 1 0 1 7 3 7 1 1 3 1 1 2 3 2 1 1 7 4 4 4 0 1 3 4 0 1 3 3 7 0 0 0 0 3 3 1 7 7 2 0 3 6 0 1 3 7 3 7 3 1 7 1 7 3 1 0 0 7 1 7'
stationClusterLst = stationClusterString.split(' ')
for i in range(len(stationClusterLst)):
    stationClusterLst[i] = int(stationClusterLst[i])

#%%
'''
Generating Training Set
'''
def convertTime(rawTime):
    rawTime = rawTime.replace('\n', '')
    timeFMT = '%I:%M:%S%p'
    groundTime = '04:00:00AM'
    timeDiff = datetime.strptime(rawTime, timeFMT) - datetime.strptime(groundTime, timeFMT)
    timeDiff = timeDiff.total_seconds()
    timeDiff = timeDiff // 60       #minutes since 4am
    return timeDiff
#%%
janData = open('JAN.csv', 'r')
febData = open('FEB.csv', 'r')
marData = open('MAR.csv', 'r')
aprData = open('APR.csv', 'r')
mayData = open('MAY.csv', 'r')
#junData = open('JUN.csv', 'r')
pickle_stations_in = open('StationList.pickle', 'rb')
stationLst = pickle.load(pickle_stations_in)
pickle_stations_in.close()
trainingLst = []


#monthNum = 0        #for January
#count = 0
#for line in janData:
#    if count % 300000 == 0:
#        print(count)
#        print(len(trainingLst))
#    if 'ENTRY' not in line:
##        print(line)
#        lineLst = []
#        arr = line.split(',')
#        for i in range(len(arr)):
#            arr[i] = arr[i].replace('"', '')
#        lineDate = int(arr[0][0:2])
#        lineDayType = dayTypeLst[monthNum][lineDate - 1]
#        lineStartStation = stationLst.index(arr[1])
#        lineStartTime = convertTime(arr[2])
#        travelTime = convertTime(arr[4]) - lineStartTime
#        if lineStartTime > 0:
#            lineWeatherLst = weatherDayLst_avg[monthNum][lineDate - 1] 
#            lineStationCluster = stationClusterLst[lineStartStation]
#            lineEndStation = stationLst.index(arr[3])
#            lineLst.append(lineDayType)
#            lineLst.append(lineStartStation)
#            lineLst.append(lineStartTime)
#            lineLst.extend(lineWeatherLst)
#            lineLst.append(lineStationCluster)
#            lineLst.append(lineEndStation)
#            lineLst.append(travelTime)
#            trainingLst.append(lineLst)
#    count += 1
#pickle_training_out = open('JanuaryWholeTrainingTime.pickle', 'wb')
#pickle.dump(trainingLst, pickle_training_out)
#pickle_training_out.close()
#    
#trainingLst = []
#monthNum = 1        #for february
#count = 0
#for line in febData:
#    if count % 300000 == 0:
#        print(count)
#        print(len(trainingLst))
#    if 'ENTRY' not in line:
##        print(line)
#        lineLst = []
#        arr = line.split(',')
#        for i in range(len(arr)):
#            arr[i] = arr[i].replace('"', '')
#        lineDate = int(arr[0][0:2])
#        lineDayType = dayTypeLst[monthNum][lineDate - 1]
#        lineStartStation = stationLst.index(arr[1])
#        lineStartTime = convertTime(arr[2])
#        travelTime = convertTime(arr[4]) - lineStartTime
#        if lineStartTime > 0:
#            lineWeatherLst = weatherDayLst_avg[monthNum][lineDate - 1] 
#            lineStationCluster = stationClusterLst[lineStartStation]
#            lineEndStation = stationLst.index(arr[3])
#            lineLst.append(lineDayType)
#            lineLst.append(lineStartStation)
#            lineLst.append(lineStartTime)
#            lineLst.extend(lineWeatherLst)
#            lineLst.append(lineStationCluster)
#            lineLst.append(lineEndStation)
#            lineLst.append(travelTime)
#            trainingLst.append(lineLst)
#    count += 1
#pickle_training_out = open('FebruaryWholeTrainingTime.pickle', 'wb')
#pickle.dump(trainingLst, pickle_training_out)
#pickle_training_out.close()

trainingLst = []
monthNum = 2
count = 0
for line in marData:
    if count % 300000 == 0:
        print(count)
        print(len(trainingLst))
    if 'ENTRY' not in line:
#        print(line)
        lineLst = []
        arr = line.split(',')
        for i in range(len(arr)):
            arr[i] = arr[i].replace('"', '')
        lineDate = int(arr[0][0:2])
        lineDayType = dayTypeLst[monthNum][lineDate - 1]
        lineStartStation = stationLst.index(arr[1])
        lineStartTime = convertTime(arr[2])
        travelTime = convertTime(arr[4]) - lineStartTime
        if lineStartTime > 0:
            lineWeatherLst = weatherDayLst_avg[monthNum][lineDate - 1] 
            lineStationCluster = stationClusterLst[lineStartStation]
            lineEndStation = stationLst.index(arr[3])
            lineLst.append(lineDayType)
            lineLst.append(lineStartStation)
            lineLst.append(lineStartTime)
            lineLst.extend(lineWeatherLst)
            lineLst.append(lineStationCluster)
            lineLst.append(lineEndStation)
            lineLst.append(travelTime)
            trainingLst.append(lineLst)
    count += 1    
pickle_training_out = open('MarchWholeTrainingTime.pickle', 'wb')
pickle.dump(trainingLst, pickle_training_out)
pickle_training_out.close()

trainingLst = []
monthNum = 3        #for april
count = 0
for line in aprData:
    if count % 300000 == 0:
        print(count)
        print(len(trainingLst))
    if 'ENTRY' not in line:
#        print(line)
        lineLst = []
        arr = line.split(',')
        for i in range(len(arr)):
            arr[i] = arr[i].replace('"', '')
        lineDate = int(arr[0][0:2])
        lineDayType = dayTypeLst[monthNum][lineDate - 1]
        lineStartStation = stationLst.index(arr[1])
        lineStartTime = convertTime(arr[2])
        travelTime = convertTime(arr[4]) - lineStartTime
        if lineStartTime > 0:
            lineWeatherLst = weatherDayLst_avg[monthNum][lineDate - 1] 
            lineStationCluster = stationClusterLst[lineStartStation]
            lineEndStation = stationLst.index(arr[3])
            lineLst.append(lineDayType)
            lineLst.append(lineStartStation)
            lineLst.append(lineStartTime)
            lineLst.extend(lineWeatherLst)
            lineLst.append(lineStationCluster)
            lineLst.append(lineEndStation)
            lineLst.append(travelTime)
            trainingLst.append(lineLst)
    count += 1
pickle_training_out = open('AprilWholeTrainingTime.pickle', 'wb')
pickle.dump(trainingLst, pickle_training_out)
pickle_training_out.close()
#
#trainingLst = []    
#monthNum = 4        #for may
#count = 0
#for line in mayData:
#    if count % 300000 == 0:
#        print(count)
#        print(len(trainingLst))
#    if 'ENTRY' not in line:
##        print(line)
#        lineLst = []
#        arr = line.split(',')
#        for i in range(len(arr)):
#            arr[i] = arr[i].replace('"', '')
#        lineDate = int(arr[0][0:2])
#        lineDayType = dayTypeLst[monthNum][lineDate - 1]
#        lineStartStation = stationLst.index(arr[1])
#        lineStartTime = convertTime(arr[2])
#        travelTime = convertTime(arr[4]) - lineStartTime
#        if lineStartTime > 0:
#            lineWeatherLst = weatherDayLst_avg[monthNum][lineDate - 1] 
#            lineStationCluster = stationClusterLst[lineStartStation]
#            lineEndStation = stationLst.index(arr[3])
#            lineLst.append(lineDayType)
#            lineLst.append(lineStartStation)
#            lineLst.append(lineStartTime)
#            lineLst.extend(lineWeatherLst)
#            lineLst.append(lineStationCluster)
#            lineLst.append(lineEndStation)
#            lineLst.append(travelTime)
#            trainingLst.append(lineLst)
#    count += 1
#pickle_training_out = open('MayWholeTrainingTime.pickle', 'wb')
#pickle.dump(trainingLst, pickle_training_out)
#pickle_training_out.close()

janData.close()
febData.close()
marData.close()
aprData.close()
mayData.close()
#junData.close()

#pickle_training_out = open('JanuaryWholeTraining.pickle', 'wb')
#pickle.dump(trainingLst, pickle_training_out)
#pickle_training_out.close()
#pickle_training_out = open('JanuaryFebruaryWholeTraining.pickle', 'wb')
#pickle.dump(trainingLst2, pickle_training_out)
#pickle_training_out.close()
#pickle_training_out = open('JanuaryFebruaryMarchWholeTraining.pickle', 'wb')
#pickle.dump(trainingLst3, pickle_training_out)
#pickle_training_out.close()
#pickle_training_out = open('MarchAprilWholeTraining.pickle', 'wb')
#pickle.dump(trainingLst4, pickle_training_out)
#pickle_training_out.close()

#%%
'''
Generating Testing Set
'''
aprData = open('APR.csv', 'r')
pickle_stations_in = open('StationList.pickle', 'rb')
stationLst = pickle.load(pickle_stations_in)
pickle_stations_in.close()
testLst = []
monthNum = 3        #for april
count = 0
for line in aprData:
    if count % 1000000 == 0:
        print(count)
        print(len(testLst))
    if '25-APR' in line:
#        print(line)
        lineLst = []
        arr = line.split(',')
        for i in range(len(arr)):
            arr[i] = arr[i].replace('"', '')
        lineDate = int(arr[0][0:2])
        lineDayType = dayTypeLst[monthNum][lineDate - 1]
        lineStartStation = stationLst.index(arr[1])
        lineStartTime = convertTime(arr[2])
        travelTime = convertTime(arr[4]) - lineStartTime
        if lineStartTime > 0:
            lineWeatherLst = weatherDayLst_avg[monthNum][lineDate - 1] 
            lineStationCluster = stationClusterLst[lineStartStation]
            lineEndStation = stationLst.index(arr[3])
            lineLst.append(lineDayType)
            lineLst.append(lineStartStation)
            lineLst.append(lineStartTime)
            lineLst.extend(lineWeatherLst)
            lineLst.append(lineStationCluster)
            lineLst.append(lineEndStation)
            lineLst.append(travelTime)
            testLst.append(np.array(lineLst))
    count += 1


aprData.close()

#%%
'''
Formatting training and testing data for predicting end station
'''
pickle_trainingLst_in = open('AprilWholeTrainingTime.pickle', 'rb')
trainingLst = pickle.load(pickle_trainingLst_in)
pickle_trainingLst_in.close()
pickle_trainingLst_in = open('MarchWholeTrainingTime.pickle', 'rb')
trainingLst.extend(pickle.load(pickle_trainingLst_in))
pickle_trainingLst_in.close()
trainingLst_input = []
trainingLst_target = []
#for i in trainingLst:
for i in range(len(trainingLst)):
    if i % 15 == 0:        #2,600,000 trips from february used for training
        trainingLst_input.append(np.array(trainingLst[i][:-1]))
        trainingLst_target.append(np.array(trainingLst[i][-1] // 5))

trainingLst_input = np.array(trainingLst_input)
trainingLst_input[np.where(trainingLst_input == 'T')] = 0    
trainingLst_input = trainingLst_input.astype(float)
#trainingLst_input[np.where(trainingLst_input < 0)] = 0
trainingLst_target = np.array(trainingLst_target)
trainingLst_target[np.where(trainingLst_target == 'T')] = 0
trainingLst_target = trainingLst_target.astype(float)
#%%
testLst_input = []
testLst_target = []
for i in range(len(trainingLst)):
    if i % 30001 == 0:  #433 trips for testing February
        testLst_input.append(np.array(trainingLst[i][:-1]))
        testLst_target.append(np.array(trainingLst[i][-1] // 5))

testLst_input2 = []
testLst_target2 = []
for i in range(len(testLst)):
    testLst_input2.append(testLst[i][:-1])
    testLst_target2.append(testLst[-i][-1] // 5)    

testLst_input = np.array(testLst_input)
testLst_input[np.where(testLst_input == 'T')] = 0
testLst_input = testLst_input.astype(float)
#testLst_input[np.where(testLst_input < 0)] = 0
testLst_target = np.array(testLst_target)
testLst_target[np.where(testLst_target == 'T')] = 0
testLst_target = testLst_target.astype(float)

testLst_input2 = np.array(testLst_input2)
testLst_input2[np.where(testLst_input2 == 'T')] = 0
testLst_input2 = testLst_input2.astype(float)
#testLst_input[np.where(testLst_input < 0)] = 0
testLst_target2 = np.array(testLst_target2)
testLst_target2[np.where(testLst_target2 == 'T')] = 0
testLst_target2 = testLst_target2.astype(float)

#%%
smallTrainingLst_input = []
smallTrainingLst_target = []
for i in range(len(trainingLst_input)):
    if i % 5000 == 0:
        smallTrainingLst_input.append(trainingLst_input[i])
        smallTrainingLst_target.append(trainingLst_target[i])
mediumTrainingLst_input = []
mediumTrainingLst_target = []
for i in range(len(trainingLst_input)):
    if i % 290 == 0:
        mediumTrainingLst_input.append(trainingLst_input[i])
        mediumTrainingLst_target.append(trainingLst_target[i])
        
largeTrainingLst_input = []
largeTrainingLst_target = []
for i in range(len(trainingLst_input)):
    if i % 100 == 0:
        largeTrainingLst_input.append(trainingLst_input[i])
        largeTrainingLst_target.append(trainingLst_target[i])
        
hugeTrainingLst_input = []
hugeTrainingLst_target = []
for i in range(len(trainingLst_input)):
    if i % 20 == 0:
        hugeTrainingLst_input.append(trainingLst_input[i])
        hugeTrainingLst_target.append(trainingLst_target[i])


#%%
'''
Linear Regression
'''
lr = LinearRegression()
y_pred_lr = lr.fit(trainingLst_input, trainingLst_target)

#%%
'''
Neighbors
'''
knr = KNeighborsRegressor()
y_pred_knr = knr.fit(trainingLst_input, trainingLst_target)

#%%
'''
Ensemble
'''
gbr = GradientBoostingRegressor()
y_pred_gbc = gbr.fit(trainingLst_input, trainingLst_target)

#%%
abr = AdaBoostRegressor()
y_pred_abr = abr.fit(trainingLst_input, trainingLst_target)

#%%
br = BaggingRegressor()
y_pred_br = br.fit(trainingLst_input, trainingLst_target)

#%%
etr = ExtraTreesRegressor()
y_pred_etr = etr.fit(trainingLst_input, trainingLst_target)

#%%
rfr = RandomForestRegressor()
y_pred_rfr = rfr.fit(trainingLst_input, trainingLst_target)

#%%
'''
Neural Network
'''
mlp = MLPRegressor()
y_pred_mlp = mlp.fit(hugeTrainingLst_input, hugeTrainingLst_target)
#%%
print('Testing with training data:')
print('Linear Regression: ', end = '')
print(lr.score(trainingLst_input[:100], trainingLst_target[:100]))
print('K Neighbors Regression: ', end = '')
print(knr.score(trainingLst_input[:100], trainingLst_target[:100]))
print('Gradient Boosting Regression: ', end = '')
print(gbr.score(trainingLst_input[:100], trainingLst_target[:100]))
print('AdaBoost Regression: ', end = '')
print(abr.score(trainingLst_input[:100], trainingLst_target[:100]))
print('Bagging Regression: ', end = '')
print(br.score(trainingLst_input[:100], trainingLst_target[:100]))
print('Extra Trees Regression: ', end = '')
print(etr.score(trainingLst_input[:100], trainingLst_target[:100]))
print('Random Forest Regression: ', end = '')
print(rfr.score(trainingLst_input[:100], trainingLst_target[:100]))
print('Multi-Layer Perceptron Regression: ', end = '')
print(mlp.score(trainingLst_input[:100], trainingLst_target[:100]))
print()

print('Testing with February data:')
print('Linear Regression: ', end = '')
print(lr.score(testLst_input, testLst_target))
print('K Neighbors Regression: ', end = '')
print(knr.score(testLst_input, testLst_target))
print('Gradient Boosting Regression: ', end = '')
print(gbr.score(testLst_input, testLst_target))
print('AdaBoost Regression: ', end = '')
print(abr.score(testLst_input, testLst_target))
print('Bagging Regression: ', end = '')
print(br.score(testLst_input, testLst_target))
print('Extra Trees Regression: ', end = '')
print(etr.score(testLst_input, testLst_target))
print('Random Forest Regression: ', end = '')
print(rfr.score(testLst_input, testLst_target))
print('Multi-Layer Perceptron Regression: ', end = '')
print(mlp.score(testLst_input, testLst_target))
print()

print('Testing with April data:')
print('Linear Regression: ', end = '')
print(lr.score(testLst_input2, testLst_target2))
print('K Neighbors Regression: ', end = '')
print(knr.score(testLst_input2, testLst_target2))
print('Gradient Regression: ', end = '')
print(gbr.score(testLst_input2, testLst_target2))
print('AdaBoost Regression: ', end = '')
print(abr.score(testLst_input2, testLst_target2))
print('Bagging Regression: ', end = '')
print(br.score(testLst_input2, testLst_target2))
print('Extra Trees Regression: ', end = '')
print(etr.score(testLst_input2, testLst_target2))
print('Random Forest Regression: ', end = '')
print(rfr.score(testLst_input2, testLst_target2))
print('Multi-Layer Perceptron Regression: ', end = '')
print(mlp.score(testLst_input2, testLst_target2))
print()

#%%
pickle_lr_out = open('MarchAprWholeLR5.pickle', 'wb')
pickle.dump(lr, pickle_lr_out)
pickle_lr_out.close()

pickle_knr_out = open('MarchAprWholeKNR5.pickle', 'wb')
pickle.dump(knr, pickle_knr_out)
pickle_knr_out.close()

pickle_gbr_out = open('MarchAprWholeGBR5.pickle', 'wb')
pickle.dump(gbr, pickle_gbr_out)
pickle_gbr_out.close()

pickle_abr_out = open('MarchAprWholeABR5.pickle', 'wb')
pickle.dump(abr, pickle_abr_out)
pickle_abr_out.close()

pickle_br_out = open('MarchAprWholeBR5.pickle', 'wb')
pickle.dump(br, pickle_br_out)
pickle_br_out.close()

pickle_etr_out = open('MarchAprWholeETR5.pickle', 'wb')
pickle.dump(etr, pickle_etr_out)
pickle_etr_out.close()

pickle_rfr_out = open('MarchAprWholeRFR5.pickle', 'wb')
pickle.dump(rfr, pickle_rfr_out)
pickle_rfr_out.close()

pickle_mlp_out = open('MarchAprWholeMLPR5.pickle', 'wb')
pickle.dump(mlp, pickle_mlp_out)
pickle_mlp_out.close()





