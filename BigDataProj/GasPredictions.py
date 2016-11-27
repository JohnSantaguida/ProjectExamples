# Author: Dillon Koval


import scipy
import numpy
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold


# year, month, day, AverageTemperature, AverageTermperatureUncertainty, State, Country
def load_temps():
    f = open('USTemps.txt')
    lines = f.readlines()
    f.close()

    US_Temps = []
    for line in lines:
	line = line.strip()
	line = line.split(',')
	line[0] = int(line[0][1:])
	line[1] = int(line[1])
	line[2] = int(line[2])
	US_Temps.append(line)

    startMonth = US_Temps[0][1]
    startTemp = US_Temps[0][3]
    USTemp_Change = []

    for entry in US_Temps:
	if entry == US_Temps[0]:
	    continue
	tempYear = entry[0]
	tempMonth = entry[1]
	temp = entry[3]
	tempChange = float(temp) - float(startTemp)
	newTempEntry = [tempYear, startMonth, tempMonth, tempChange]
	USTemp_Change.append(newTempEntry)
	startMonth = tempMonth
	startTemp = temp
	
    return USTemp_Change


#year, month, price
def load_gasprices():
    f = open('GasPrices.txt')
    lines = f.readlines()
    f.close()

    GasPrices = []
    
    lines = lines[1:-3] 
 
    for line in lines:
	line = line.split(',')
        
        line[0] = int(line[0][1:])
	line[1] = int(line[1])
	line[2] = float(line[2][:-2])
 	if line[0] != 1993:
	    GasPrices.append(line)   
        

    
    monthly_avg = []
    count = 0
    sum = 0
    thisYear = int(GasPrices[0][0])
    thisMonth = int(GasPrices[0][1])
    nextMonth = thisMonth + 1
    for entry in GasPrices:
	
	if entry[1] == thisMonth:
	    sum = sum + entry[2]
	    count = count + 1
	else:
	    if thisMonth == 12:
		thisYear = entry[0] - 1
	    else:
		thisYear = entry[0]
	    avgPrice = sum/count
	    monthly_avg.append([thisYear, thisMonth, avgPrice])
	    thisMonth = nextMonth
	    nextMonth = nextMonth + 1
	    sum = 0
	    count = 0
	    if nextMonth > 12:
		nextMonth = 1

	
    
    startMonth = monthly_avg[0][1]
    startPrice = monthly_avg[0][2]
    GasPriceChange = []
    for entry in monthly_avg:
	if entry == monthly_avg[0]:
	    continue
	year = entry[0]
	endMonth = entry[1]
	if year == 2013 and endMonth == 10:
	    break
	
	price = entry[2]
	priceChange = price - startPrice
	newEntry = [year, startMonth, endMonth, priceChange]
	GasPriceChange.append(newEntry)
	startMonth = endMonth
	startPrice = price
    
    return GasPriceChange

    
def createInput(USTemp_Change):
    WIDTH = len(USTemp_Change[0])
    X = scipy.zeros((len(USTemp_Change), WIDTH))
    for i in range(0, len(USTemp_Change)):
	for j in range(0,WIDTH):
	     X[i, j] = float(USTemp_Change[i][j])
    return X


def createOutput(TempAndGasChange):
    Y = scipy.zeros(len(TempAndGasChange))
    for i in range(0, len(TempAndGasChange)):
	if TempAndGasChange[i][4] > 0:
	    Y[i] = 1
    print 'Number of gas increases over a month time', sum(Y)
    return Y

def createBaseline(USTemp_Change):
    X = scipy.zeros((len(USTemp_Change), 1))
    for i in range(0, len(USTemp_Change)):
	X[i,0] = USTemp_Change[i][3]
    
    return X

def combineData(USTemp_Change, GasPrice_Change):
    assert len(USTemp_Change) == len(GasPrice_Change)
    TempAndGasChange = []
    for i in range(1, len(USTemp_Change)):
	try:
	    assert USTemp_Change[i][0] == GasPrice_Change[i][0]
	    assert USTemp_Change[i][1] == GasPrice_Change[i][1]
	    assert USTemp_Change[i][2] == GasPrice_Change[i][2]
	except AssertionError:
	    print USTemp_Change[i]
	    print GasPrice_Change[i]
	    print '\n'
	newEntry = [USTemp_Change[i][0], USTemp_Change[i][1], USTemp_Change[i][2], USTemp_Change[i][3], GasPrice_Change[i][3]]
	TempAndGasChange.append(newEntry)
    
    return TempAndGasChange

def test_classifier(clf, X, Y):
    folds = StratifiedKFold(Y,5)
    aucs = []
    for train, test in folds:
	# print X[train].shape, Y[train].shape
	# print X[test].shape
	clf.fit(X[train], Y[train])
	prediction = clf.predict_proba(X[test])
	aucs.append(roc_auc_score(Y[test], prediction[:,1]))
    print clf.__class__.__name__, aucs, numpy.mean(aucs)

def main():
    
    USTemp_Change = load_temps()
    GasPrice_Change = load_gasprices()
    TempAndGasChange = combineData(USTemp_Change, GasPrice_Change)
        
  #  X = createBaseline(USTemp_Change)
    X = createInput(USTemp_Change)    
    
    Y = createOutput(TempAndGasChange)
   
    clf = linear_model.SGDClassifier(loss='log')
    test_classifier(clf, X, Y)

    clf = GaussianNB()
    test_classifier(clf, X, Y)

    clf = RandomForestClassifier(n_estimators=10, max_depth=10)
    test_classifier(clf, X, Y)
    
    

if __name__ == '__main__':
    main()
