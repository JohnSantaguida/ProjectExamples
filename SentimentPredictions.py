import scipy
import numpy
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold


# year, month, day, AverageTemperature, AverageTermperatureUncertainty, State, Country
def load_news():
    f = open('hdfs:///user/brewereg/fpout/sentiment_data')
    lines = f.readlines()
    f.close()

    AvgSentValues = []
    for line in lines:
		line = line.split(',')
		line = line.strip()
		line[0] = int(line[0][1:])
		line[1] = int(line[1])
		line[2] = float(line[2])
		AvgSentValues.append(line)

    startMonth = AvgSentValues[0][1]
    startSent = AvgSentValues[0][2]
    AvgSentChange = []

    for entry in AvgSentValues:
	if entry == AvgSentValues[0]:
	    continue
	if entry[0] < 1994
		continue
	if entry[0] >= 2013 && entry[1] > 9
		continue
	sentYear = entry[0]
	sentMonth = entry[1]
	sent = entry[3]
	sentChange = float(sent) - float(startSent)
	newSentEntry = [sentYear, startMonth, sentMonth, sentChange]
	AvgSentChange.append(newSentEntry)
	startMonth = sentMonth
	startSent = sent
	
    return AvgSentChange


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
        

    print(GasPrices[1])
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

    
def createInput(AvgSentChange):
    WIDTH = len(AvgSentChange[0])
    X = scipy.zeros((len(AvgSentChange), WIDTH))
    for i in range(0, len(AvgSentChange)):
	for j in range(0,WIDTH):
	     X[i, j] = float(AvgSentChange[i][j])
    return X


def createOutput(SentAndGasChange):
    Y = scipy.zeros(len(SentAndGasChange))
    for i in range(0, len(SentAndGasChange)):
	if SentAndGasChange[i][4] > 0:
	    Y[i] = 1
    print 'Number of gas increases over a month time', sum(Y)
    return Y

def createBaseline(AvgSentChange):
    X = scipy.zeros((len(AvgSentChange), 1))
    for i in range(0, len(AvgSentChange)):
	X[i,0] = AvgSentChange[i][3]
    
    return X

def combineData(AvgSentChange, GasPrice_Change):
    assert len(AvgSentChange) == len(GasPrice_Change)
    SentAndGasChange = []
    for i in range(1, len(AvgSentChange)):
	try:
	    assert AvgSentChange[i][0] == GasPrice_Change[i][0]
	    assert AvgSentChange[i][1] == GasPrice_Change[i][1]
	    assert AvgSentChange[i][2] == GasPrice_Change[i][2]
	except AssertionError:
	    print AvgSentChange[i]
	    print GasPrice_Change[i]
	    print '\n'
	newEntry = [AvgSentChange[i][0], AvgSentChange[i][1], AvgSentChange[i][2], AvgSentChange[i][3], GasPrice_Change[i][3]]
	SentAndGasChange.append(newEntry)
    
    return SentAndGasChange

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
    print("Started main")
    AvgSentChange = load_temps()
    GasPrice_Change = load_gasprices()
    SentAndGasChange = combineData(AvgSentChange, GasPrice_Change)
    
    X = createInput(AvgSentChange)    
    
    Y = createOutput(SentAndGasChange)
   
    clf = linear_model.SGDClassifier(loss='log')
    test_classifier(clf, X, Y)

    clf = GaussianNB()
    test_classifier(clf, X, Y)

    clf = RandomForestClassifier(n_estimators=10, max_depth=10)
    test_classifier(clf, X, Y)
    
    

if __name__ == '__main__':
    main()
