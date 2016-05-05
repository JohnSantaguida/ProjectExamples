
from mrjob.job import MRJob
from mrjob.step import MRStep



class GasMRJobReg(MRJob):
    def mapper(self, _, line):
        try:
            c = line.strip().split(',')
	    #print c	    
	    yield int(c[0][-4:]),float(c[1])
	except:
	    pass
    def reducer(self, key, values):
	lst=list(values)
	#print lst
	yield key, sum(lst)/len(lst)

if __name__ == "__main__":
    GasMRJobReg.run()
