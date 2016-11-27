from pyspark import SparkConf, SparkContext

import sys
import re

from variables import MACHINE, VUID, PAGE_TABLE, INDEX_TABLE, COLUMN_FAMILY, COLUMN
import unicodedata


links_file = 'hdfs:///user/brewereg/fpout/links_file'
buzz_words = ('gas','gasoline','oil','petroleum','fuel','the pump','diesel','crude','petrol')


def index(spark, news_links):

    news_data = spark.textFile(news_links) 
    #tab-delimited dataset
    news_data = news_data.map(lambda line: line.split("\t")) \
    .map(lambda x: (x[0],x[1],x[2],x[4],x[7])) \
    .filter(lambda x: gas_related(x[1]))  #are articles related to gas prices/parsing  
    
    news_data.saveAsTextFile(links_file)

#used in lambda function and filter call to determine relevancy of article
def gas_related(headline):
    headline = unicodedata.normalize('NFKD',headline).encode('ascii','ignore')
    headline =  headline.split(" ")
    for word in headline:
        if buzz_words.count(word) > 0:
            return True
    return False


if __name__ == '__main__':
    conf = SparkConf()
    if sys.argv[1] == 'local':
        conf.setMaster("local[3]")
        print 'Running locally'
    elif sys.argv[1] == 'cluster':
        conf.setMaster("spark://10.0.22.241:7077")
        print 'Running on cluster' 
    conf.set("spark.executor.memory", "10g")
    conf.set("spark.driver.memory", "10g")
    spark = SparkContext(conf = conf)
    index(spark, sys.argv[2])


