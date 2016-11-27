
from mrjob.job import MRJob

class GasMRJob(MRJob):
    def mapper(self, _, line):
        try:
            c = line.strip().split(',')
            key = int(str('20' + c[0][-2:]))
            value = (float(c[1]) + float(c[2]) + float(c[3]) + float(c[4]) + float(c[5]) + float(c[6]) + float(c[7]) + float(c[8]) + float(c[9]) + float(c[10]))/10.
            #print key, value
            yield key, value
        except:
            pass
    def reducer(self, key, values):
        lst = list(values)
        yield key, sum(lst)/len(lst)

if __name__ == '__main__':
    GasMRJob.run()
