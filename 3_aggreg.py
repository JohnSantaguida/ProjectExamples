'''Thank you to Hitchiker's Guide to python:http://docs.python-guide.org/en/latest/scenarios/scrape/'''
from pyspark import SparkConf, SparkContext
import sys
import re
import time
import requests
import unicodedata

positive_words = ('happy','glad','good','slip','save','low','downturn','slid','decreas')
negative_words = ('sad','upset','bad','mad','spike','increase','rise','rose','risen','high','soar')


links_and_content = 'hdfs:///user/brewereg/fpout/links_and_content'
data_output = 'hdfs:///user/brewereg/fpout/sentiment_data'

def aggregator(spark, contents_data):
    contents_rdd = spark.textFile(contents_data)\
    .map(lambda x:(x[0],x[3],sentiments(x[4])))\
    .map(lambda x:(x[0],timeConvert(x[1]),x[2]))

    reduction = contents_rdd.map(lambda x:(timeConvert(x[1]),x[2]))\
    .reduceByKey(lambda z: (z,1) lambda x, z:(x[0]+z,x[1]+1),lambda x,y:(x[0]+y[0],x[1]+y[1])).map(lambda (x,y):(x[0],x[1],y))
    #^this computes the average with a reduceByKey call.

    reduction.saveAsTextFile(data_output)

#cleanup html data and run a basic sentiment analysis on it to retrieve score
def sentiments(raw_text):
    cleaned = raw_text.encode('ascii','ignore')
    text_str = str(cleaned).lower()
    pos = 0
    neg = 0
    for word1 in positive_words:
        pos += text_str.count(str(word1))
    for word2 in negative_words:
        neg += text_str.count(str(word2))
    return pos-neg 

#convert milliseconds to seconds and then to the date of the article
def timeConvert(raw_mills):
    mills = int(raw_mills.encode('ascii','ignore'))
    return time.strftime('%Y,%m',time.localtime(mills/1000)
    
if __name__ == '__main__':
    conf = SparkConf()
    if sys.argv[1] == 'local':
        conf.setMaster("local[3]")
        print 'Running locally'
    elif sys.argv[1] == 'cluster':
        conf.setMaster("spark://10.0.22.241:7077")
        print 'Running on cluster'
    conf.set("spark.executor.memory","10g")
    conf.set("spark.driver.memory","10g")
    spark = SparkContext(conf = conf)
    aggregator(spark, links_and_content)



