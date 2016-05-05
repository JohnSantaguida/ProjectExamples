from pyspark import SparkConf, SparkContext

import sys
import re

from variables import MACHINE, VUID, PAGE_TABLE, INDEX_TABLE, COLUMN_FAMILY, COLUMN
import unicodedata

#MIN_OCCURRENCES = 10
#MAX_WORDS = 5000

#index_file = 'hdfs:///user/brewereg/word_index'
links_file = 'hdfs:///user/brewereg/fpout/links_file'
buzz_words = ('gas','gasoline','oil','petroleum','fuel','the pump','diesel','crude','petrol')

'''
Complete the index function to write tuples: title,word, count
Where tuple and word are concatenated with title + ',' + word.

Write the output to <index_file> using saveAsTextFile(index_file)

A good example can be found at: http://www.mccarroll.net/blog/pyspark2/
'''

def index(spark, news_links):

    news_data = spark.textFile(news_links) 
    news_data = news_data.map(lambda line: line.split("\t")) \
    .map(lambda x: (x[0],x[1],x[2],x[4],x[7])) \
    .filter(lambda x: gas_related(x[1])) 
    #.map(lambda x: (x[0],x[1],x[2],x[3],x[4])) \    
    #.filter(x : x[4] == "b")
    #.flatMap(lambda [nID,nTitle,URL,pub,category,clusterID,hostname,timestamp]: [nID,nTitle,URL,category,timestamp])
    
          
        #.flatMap(lambda (title, text): [(title + ',' + word, 1) for word in text]) \
        #.reduceByKey(lambda c, d:c+d) #\
        #filter()   
    
    news_data.saveAsTextFile(links_file)

'''
# return true if a word is frequent.
def is_frequent(index_record):
    return index_record[1] > MIN_OCCURRENCES
'''

#def csv_parser(nID,title,URL,publisher,category,story,hostname,timestamp)
#    return (, title, URL, category, timestamp)

def gas_related(headline):
    headline = unicodedata.normalize('NFKD',headline).encode('ascii','ignore')
    headline =  headline.split(" ")
    for word in headline:
        if buzz_words.count(word) > 0:
            return True
    return False
    #set(hl).set(buzz_words))
    #return not (len(hl) == 0)
def non_ASCII(uni_text):
    return ''.join(i for i in uni_text if ord(i)<128) 



def get_title(text):
    title = '<title>'
    title_end = '</title>'
    start = text.index(title) + len(title)
    end = text.index(title_end)
    return text[start:end].lower()


def get_text(text):
    text_tag = '<text xml:space="preserve">'
    text_end = '</text>'
    start = text.index(text_tag) + len(text_tag)
    end = text.index(text_end)
    text_block = text[start:end].lower()
    return re.sub(r"\W+", ' ', text_block).strip().split(' ')[:MAX_WORDS]

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


