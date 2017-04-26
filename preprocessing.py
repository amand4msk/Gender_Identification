import errno, glob, ntpath, sys, re

path = 'PAN_2015/tweets/*.txt'  
path1 = 'preprocessing/*.txt'
i=0
j=0

####################################################################################
# Preprocessing for removing hastags, mentions and urls
####################################################################################

def preprocessing(): 
	files = glob.glob(path)  
	for name in files:
		array_tweets = [] 
		try:
			with open(name) as f:
				for line in f.readlines():
					if line.rstrip(): # remove the empty lines
						l = line.split("\n") 
						#removing hastags, mentions, urls
						expr = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",l[0]).split())
						array_tweets.append(expr)
						name = path_leaf(name)
						save_to_file(array_tweets,name)
		except IOError as exc:
			# Do not fail if a directory is found, just ignore it.
			if exc.errno != errno.EISDIR: 
				raise # Propagate other kinds of IOError.

# Extract the name of the file from the path
def path_leaf(name):
    head, tail = ntpath.split(name)
    return tail or ntpath.basename(head)

# Save to the preprocessing tweets in new files
def save_to_file(array_tweets,name):
	f = open('preprocessing/'+str(name), 'w') 
	f.write('\n'.join(array_tweets))
	print f

def main():
	preprocessing()

if __name__ == '__main__':
	main()	