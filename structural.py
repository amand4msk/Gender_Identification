import argparse, errno, glob, re, random, sys, ntpath, nltk
import numpy as np
from classifier import SVMclassifier
from collections import Counter
from pickle import dump
from string import punctuation

path = 'PAN_2015/tweets/*.txt'  

regex_str = [
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
 
def parseArguments():
	# Create argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument("--extract", \
						help="extract counts of mentions, hashTags, Urls and punctuation signs features.", \
						action="store_true")
	args = parser.parse_args()
	return args

def tokenize(s):
  return tokens_re.findall(s)

def preprocess(s, lowercase=False):
  tokens = tokenize(s)
  if lowercase:
    tokens = [token for token in tokens]
  return tokens

def path_leaf(name):
  head, tail = ntpath.split(name)
  return tail or ntpath.basename(head)

def extract_hasTags_Mentions_Urls():
  cm = open('Counts.txt', 'w') 
  files = glob.glob(path)  
  for name in files: 
    n = path_leaf(name)
    x = n.split('.')
    hastags = []
    mentions = []
    urls = []
    try:
      with open(name) as f:
        for line in f.readlines():
          if line.rstrip(): # remove the empty lines
            l = line.split("\n") 
            if preprocess(l[0]):
              for i in preprocess(l[0]):
                if i.startswith('#'):
                  hastags.append([i])
                elif i.startswith('@'):
                  mentions.append([i])
                elif i.startswith('h'):
                  urls.append([i])
                else:
                  hastags.append([0])
                  mentions.append([0])
                  urls.append([0])
    except IOError as exc:
      # Do not fail if a directory is found, just ignore it.
      if exc.errno != errno.EISDIR: 
        raise # Propagate other kinds of IOError.

    counts = Counter(open(name).read())
    punctuation_counts = {k:v for k, v in counts.iteritems() if k in punctuation}
    cm.write(x[0]+','+str(len(mentions))+','+str(len(hastags))+','+str(len(urls))+','+str(sum(punctuation_counts.values()))+'\n')

def load(args):
  array_counts, array_labels = [], []
  dict_labels = {}

  f1 = open('Counts.txt') 
  f2 = open('PAN_2015/truth') 
  #f3 = open('labels.txt','w') 

  for line in f2.readlines():
    user = line.split(":::")
    dict_labels[user[0]] = user[1]
  
  for line in f1.readlines():
    counts = line.replace("\n","").split(",")
    for key,value in dict_labels.iteritems():
      if key == counts[0]:
        array_counts.append([float(counts[1]),float(counts[2]),float(counts[3]),float(counts[4])])
        array_labels.append(value)
        #f3.write(key+","+value+"\n")
  SVMclassifier(args, array_counts, array_labels)

def main(args):
	if args.extract:
		extract_hasTags_Mentions_Urls()
	else:
  		load(args=None)

if __name__ == '__main__':
	args = parseArguments()
	main(args)
