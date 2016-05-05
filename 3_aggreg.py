'''Thank you to Hitchiker's Guide to python:http://docs.python-guide.org/en/latest/scenarios/scrape/'''
from pyspark import SparkConf, SparkContext
import sys
import re
import time
import requests
import unicodedata

#page = requests.get('http://www.cnn.com/2014/01/18/health/fish-oil-recovery/', stream=True)
#content = page.content
#f = open('workfile', 'r+')
#f.write();
#f.close()

positive_words = ('happy','glad','good','slip','save','low','downturn','slid','decreas')
negative_words = ('sad','upset','bad','mad','spike','increase','rise','rose','risen','high','soar')

links_file = 'hdfs:///user/brewereg/fpout/links_file'
links_and_content = 'hdfs:///user/brewereg/fpout/links_and_content'
data_output = 'hdfs:///user/brewereg/fpout/sentiment_data'

def aggregator(spark, contents_data):
    contents_rdd = spark.textFile(contents_data)\
    .map(lambda x:(x[0],x[3],sentiments(x[4])))\
    .map(lambda x:(x[0],timeConvert(x[1]),x[2]))

    reduction = contents_rdd.map(lambda x:(timeConvert(x[1]),x[2]))\
    .reduceByKey(lambda z: (z,1) lambda x, z:(x[0]+z,x[1]+1),lambda x,y:(x[0]+y[0],x[1]+y[1])).map(lambda (x,y):(x[0],x[1],y))

    reduction.saveAsTextFile(data_output)


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


def timeConvert(raw_mills):
    mills = int(raw_mills.encode('ascii','ignore'))
    return time.strftime('%Y,%m',time.localtime(mills/1000)
    
'''def getPageContent(url_data):
    rawurl = unicode(url_data)
    #f.write(url_data);
    asciiurl = ''.join(i for i in rawurl if ord(i)<128)
    index = asciiurl.find("http")
    finalurl = asciiurl[index:]
    try:
        r = requests.get(url=finalurl, stream=True)
        print "trying:%s", finalurl 
    except requests.exceptions.RequestException as e:
        r = -1
    if r!=-1:
        print r
        return r
    else:
        return -1
'''
#tree = html.fromstring(page.text)
#f = open('workfile', 'r+')
#f.write(page.content);
#f.close()
#This will create a list of buyers:
#buyers = tree.xpath('//div[@title="buyer-name"]/text()')
#This will create a list of prices
#`prices = tree.xpath('//span[@class="item-price"]/text()')

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
    aggregator(spark, links_file)



