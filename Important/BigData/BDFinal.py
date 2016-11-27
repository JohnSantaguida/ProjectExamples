# conda install scikit-learn
import scipy
import numpy
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from mrjob.job import MRJob 
# Loads data from the HOUSING AFFORDABILITY DATA SYSTEM (hads)
def load_orig():
    f = open('hads2001.txt', 'r')
    lines = f.readlines()
    f.close()
    hads01 = []
    for line in lines:
        line = line.strip()
        if line[0:7] == 'CONTROL':
            continue
        line = line.split(',')
	hads01.append(line)
    return hads01


# loads from same data year 2003
def load_new():
    f = open('hads2003.txt', 'r')
    lines = f.readlines()
    f.close()

    hads03 = []
    for line in lines:
        line = line.strip().split(',')
        if line[0] == 'CONTROL':
            continue
	hads03.append(line)
    return hads03	

#load input gas data that was formed using a mapreduce script over multiple csv files
#gas_dict[i] is the year by year gas price avg list for geography code i 
#gas_dict[0] contains city gas price data
def load_gas():    
    gas_dict = {}
    for i in range(0,6):
	fname = 'output' + str(i) + '.csv'
	f = open(fname)
	lines = f.readlines()
	f.close()
	gas_dict[i] = {} 
	for line in lines:
	    line = line.strip().split(',')
	    line = int(line[0][:4]), float(line[0][5:])
	    gas_dict[i][line[0]] = line[1]
    return gas_dict



def load():
    return load_orig(), load_new()


#Extracts important columns from messy hads dataset and sort by id first then environment type
def create_important1(hads2001):
    # USE contains indexes of useful data [ID, Deneloped Env Type, Region: of US, Age of oldest resident, Weight, Age of House, Num Units]
    USE = [0,10,16,1, 17,3, 6]
    WIDTH = len(USE)
    HEIGHT = len(hads2001)
    WEIGHT_COL = USE.index(17)
    AGE_COL = USE.index(1)
    HOUSE_BUILT_COL = USE.index(3)
    UNITS_COL = USE.index(6)
    X = scipy.zeros((HEIGHT, WIDTH))
    for i in range(0, HEIGHT):
	for j in range(0,WIDTH):
	    val = float(hads2001[i][USE[j]].strip('\''))
	    if not((AGE_COL-j)):
	        val = 0 if val>55 else val/abs(val) 
	    elif j == WEIGHT_COL:
	        val = val/1000
	    elif j == HOUSE_BUILT_COL:
		val = -1 if (2016-val)>60 else (0 if (2016-val)>20 else 1)
	    elif j == UNITS_COL:
		val = -1 if val == 1 else (0 if val < 50 else 1)
	    X[i, j] = val
    return sorted(X,key=lambda l:(l[0], -l[1]))

#Extracts important columns using a dictionary with key ID --- NOTE The columns are different for the 2 datasets
def create_important3(hads2003):
    # Same as create_input1 with adjusted USE and dict usage
    USE = [0,2,3,1, 17,5, 8]
    WIDTH = len(USE)
    HEIGHT = len(hads2003)
    WEIGHT_COL = USE.index(17)
    AGE_COL = USE.index(1)
    HOUSE_BUILT_COL = USE.index(5)
    UNITS_COL = USE.index(8)
    Y = {}
    for i in range(0, HEIGHT):
	index = int(hads2003[i][0].strip('\''))
        Y[str(index)] = []
        for j in range(1,WIDTH):
            rval = float(hads2003[i][USE[j]].strip('\''))
            val = rval
	    if not((AGE_COL-j)):
                val = 0 if rval>55 else rval/abs(rval)
            elif j == WEIGHT_COL:
                val = rval/1000
            elif j == HOUSE_BUILT_COL:
                val = -1 if (2016-rval)>60 else (0 if (2016-rval)>20 else 1)
            elif j == UNITS_COL:
                val = -1 if rval == 1 else (0 if rval <50 else 1)
       	    Y[str(index)].append(val)
    return Y
#Originally intended to represent changes in data but most houses do not change in 2 years
#now simply returns properly formatted (for classification) hads2003 info for all houses in both datasets
def create_joined_input(hads2001, hads2003):
    WIDTH = len(hads2001[0]-1)
    HEIGHT = len(hads2001)
    Z = scipy.zeros((HEIGHT, WIDTH))
    same = set(hads2003.keys())
    for i in range(0,HEIGHT):
	keyi = int(hads2001[i][0])
	if str(keyi) in same:
	    for j in range(0,WIDTH-1):
		Z[i][j] = hads2003[str(keyi)][j]#(hads2001[str(keyi)][j])
    return Z

#Create classifier array with a 1 if the house would be expected to increase 0 for decrease
#Prints num houses with > AVG gas price growth
def create_output(gas,hads):
    A = scipy.zeros(len(hads))
    cityavg = float(gas[0][2003])
    for x in range(0,len(hads)):
	region_change = (gas[int(hads[x][2])][2003] - gas[int(hads[x][2])][2001])
	A[x] = region_change
    avg = sum(A)/len(A) if hads[x][1] != 5 else cityavg
    B = scipy.zeros(len(hads))
    for a in range(0,len(A)):
	B[a] = int((A[a]-avg)>0)
    print str(sum(B)) + ' houses out of ' + str(len(B)) + ' had larger than avg gas price increase'
    return B

#tests increasing gas prices against hads data
def test_classifier(clf, X, Y):
    folds = StratifiedKFold(Y, 5)
    aucs = []
    for train, test in folds:
        # Sizes
 	clf.fit(X[train], Y[train])
        prediction = clf.predict_proba(X[test])
        aucs.append(roc_auc_score(Y[test], prediction[:, 1]))
#Debug Statements
#	print 
#	print prediction
#	print X[train].shape, Y[train].shape
#       print X[test].shape, len(prediction)

    print clf.__class__.__name__, aucs, numpy.mean(aucs)





def main():
    G = load_gas()
    hads2001, hads2003 = load()
    X = create_important1(hads2001)
    Y = create_important3(hads2003)
    Z = create_joined_input(X,Y) 
    A = create_output(G,X)
#Debuggin
#    for y in Y:
#	print y
#    print G
#    print A, X, Y, Z 
#    for a in A:
#	print a
#    print G
#    for z in Z:
#	print z
#    print Y
#    print Y
#    print X
    clf = linear_model.SGDClassifier(loss='log')
    test_classifier(clf, Z, A)

    clf = GaussianNB()
    test_classifier(clf, Z, A)

    clf = RandomForestClassifier(n_estimators=10, max_depth=10)
    test_classifier(clf, Z, A)


if __name__ == '__main__':
    main()
    
