import argparse, errno, glob, matplotlib, ntpath, nltk, sys, scipy.sparse
import matplotlib.pyplot as plt	
import numpy as np
import pylab as pl 
from classifier import SVMclassifier
from gensim import corpora, models, similarities, matutils
from nltk import bigrams, trigrams
from pprint import pprint   # pretty-printer
from wordcloud import WordCloud
from scipy.sparse import csr_matrix
from sklearn import svm, grid_search, cross_validation
from sklearn.preprocessing import normalize, scale
from sklearn.cross_validation import StratifiedKFold, KFold, \
									 StratifiedShuffleSplit, ShuffleSplit, \
									 train_test_split

path = 'preprocessing/*.txt'
path1 = 'pos/*.txt'
path2 = 'svm/*.txt'

def parseArguments():
	#Create argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument("--pos", nargs='*', help="compute part of speech n-grams")
	parser.add_argument("--bag", nargs='*', help="compute bag of words n-grams" )
	parser.add_argument("--tfidf", nargs='*', help="compute tfidf of n-grams")
	parser.add_argument("--plot", help="features performance for different values of parameter C", action="store_true")
	parser.add_argument("--cloud", help="plot wordCloud", action="store_true")
	args = parser.parse_args()
	return args

#Load POS files which extractd from stanford code #
def load_pos(args):
	documents = [] 
	files = glob.glob(path1)  
	for name in files:
		try:
			with open(name) as f:
				documents.append(f.read().replace('\n'," "))
		except IOError as exc:
			if exc.errno != errno.EISDIR: 
				raise 
	content(documents, args)

#Load Preprocessed Tweets
def load_tweets(args):
	documents = []
	files = glob.glob(path)  
	for name in files:
		try:
			with open(name) as f:
				documents.append(f.read().replace('\n'," "))
		except IOError as exc:
			if exc.errno != errno.EISDIR: 
				raise 
	content(documents, args)

#Create the content based features 
def content(documents, args):
	#[['human', 'interface', 'computer'],['survey', 'user', 'computer', 'system', 'response', 'time']]
	terms = []
	texts = [[word for word in document.lower().split()]for document in documents]
	#pprint(texts)

	if args.pos:
		combination = args.pos[0]
	if args.bag:
		combination = args.bag[0]
	if args.tfidf:
		combination = args.tfidf[0]
	#Create unigrams, bigrams, trigrams 
	for text in texts:
		arr = []
		if combination == "tri" or combination == "all":
			for trigram in trigrams(text):
				arr.append(' '.join(trigram))
		if combination == "bi" or combination == "all":
			for bigram in bigrams(text):
				arr.append(' '.join(bigram))
		if combination == "uni" or combination == "all":
			for unigram in text:
				arr.append(unigram)
		terms.append(arr)
 

	#Create dictionary
	dictionary = corpora.Dictionary(terms)
	dictionary.filter_extremes(no_below=2)
	#print dictionary
	#dictionary.save('tweets_dict.dict') # store the dictionary, for future reference
	#print(dictionary.token2id)

	#Create the corpus
	corpus = [dictionary.doc2bow(text) for text in terms]
	#corpora.MmCorpus.serialize('tweets.mm', corpus) # store to disk, for later use
	#print(corpus)

	#Bag of words
  	#The corpus is a bow unigrams convert tuples to sparse matrix
	if args.bag:
		bow_sparse = sparsify(corpus, dictionary)
		SVMclassifier(args, bow_sparse, labels=labels())
	
	#POS
	if args.pos:
		pos_sparse = sparsify(corpus, dictionary)
		SVMclassifier(args, pos_sparse, labels=labels())

	#TF-IDF
	if args.tfidf:
		tfidf = models.TfidfModel(corpus)
		tfidf_corpus = tfidf[corpus]
		tfidf_sparse = sparsify(tfidf_corpus, dictionary)
		SVMclassifier(args, tfidf_sparse, labels=labels())
		if args.cloud:
			visualize_coef(tfidf_sparse, dictionary)

#Create sparse matrix for each feature
def sparsify(corpus, dictionary):
	N = len(corpus)
	d = len(dictionary.token2id)
	sparse_corpus = csr_matrix((N, d))
	for i, doc in enumerate(corpus):
		for j, feat in enumerate(doc):
			idx = feat[0] 
			val = feat[1]
			sparse_corpus[i, idx] = val
	return sparse_corpus

#Load labels
def labels():
	array_labels = []
	f = open('labels.txt') 
	for line in f.readlines():
		user = line.replace("\n","").split(",")
		array_labels.append(user[1])
	return array_labels 

#Create the word cloud for the best feature
def visualize_coef(features, dictionary):
	X = normalize(features)
	y = np.array(labels())
	y_new = []
	for i,yy in enumerate(y):
		if yy == 'M':
			y_new.append(-1)
		else:
			y_new.append(1)
	y_new = np.array(y_new)

	clf = svm.LinearSVC(C=3.0)
	clf.fit(X, y)

	words = np.array(dictionary.token2id.keys())
	coef = clf.coef_[0]
	coef_sorted = np.sort(coef)[::-1]
	coef_idxs = np.argsort(coef)[::-1]
	idxs = coef_idxs.tolist()
	wc = WordCloud(background_color="white")
	wc.generate_from_frequencies(zip(words[idxs][24187:],[abs(v) for v in coef_sorted[24187:]]))
	plt.imshow(wc)
	plt.show()

	wc.generate_from_frequencies(zip(words[idxs][:1000],[abs(v) for v in coef_sorted[:1000]]))
	plt.imshow(wc)
	plt.show()

#Extract the name of the file from the path
def path_leaf(name):
    head, tail = ntpath.split(name)
    return tail or ntpath.basename(head)

#Load the C parameters and the accuracy for each feature
def load_parameters():
	all_results = []
	filename = []
	files = glob.glob(path2)  
	for name in files:
		try:
			results = []
			with open(name) as f:
				filen = path_leaf(name)
				fname = filen.split(".")
				filename.append(fname[0])
				for line in f.readlines():
					l = line.strip()
					val = l.replace(" ","").split(":")
					cparam = val[0].split("=")
					results.append((float(val[1]),float(cparam[1])))
			all_results.append(results)
		except IOError as exc:
			if exc.errno != errno.EISDIR: 
				raise 

	bow_results = all_results[0:4]
	pos_results = all_results[4:8]
	str_results = [all_results[8]]
	tf_results = all_results[9:]

	plot_results(bow_results, filename[0:4])
	plot_results(pos_results, filename[4:8])
	plot_results(tf_results, filename[9:])
	plot_results(str_results, [filename[8]])
	print len(filename)

#Plot the C parameters and the accuracy for each feature
def plot_results(results, legends):
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.set_xscale('log')
		matplotlib.rcParams.update({'font.size': 17})
		for i,res in enumerate(results): 
			sorted_arr = sorted(res, key=lambda tup: tup[1])
			sorted_res = [val[0] for val in sorted_arr]
			sorted_c = [val[1] for val in sorted_arr]
			ax.plot(sorted_c,sorted_res, label=legends[i])
		plt.ylabel('Accuracy', fontsize=20)
		plt.xlabel('C',fontsize=20)
		plt.legend(loc=4)
		plt.show()

def main(args):
	if args.pos:
		load_pos(args)
	elif args.bag or args.tfidf:
		load_tweets(args)
	elif args.plot:
		load_parameters()

if __name__ == '__main__':
	args = parseArguments()
	print args
	main(args)

