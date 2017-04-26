import errno, glob, ntpath, os, sys, re

path = 'preprocessing/*.txt'
print path

def command():
	files = glob.glob(path)  
	for name in files:
		try:
			with open(name) as f:
				name_file = path_leaf(name)
				os.system('java -mx2000m -classpath stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model models/english-bidirectional-distsim.tagger  -textFile /Users/amanda/Desktop/NLP/preprocessing/'+str(name_file)+' > /Users/amanda/Desktop/NLP/pos/pos_'+str(name_file)+'')

		except IOError as exc:
			# Do not fail if a directory is found, just ignore it.
			if exc.errno != errno.EISDIR: 
				raise # Propagate other kinds of IOError.

# Extract the name of the file from the path
def path_leaf(name):
	head, tail = ntpath.split(name)
	return tail or ntpath.basename(head)

def main():
	command()
if __name__ == '__main__':
	main()	