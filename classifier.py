import numpy as np
from sklearn.preprocessing import normalize
from sklearn import svm, cross_validation, grid_search
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit

def SVMclassifier(args, array_counts, labels):
  f = open("svm/structural.txt","w")
  if not args:
    f = open("svm/structural.txt","w")
  else:
    for k,v in vars(args).items(): 
      if vars(args)[k] is not None and vars(args)[k] is not False:
        if not args.cloud:
          f = open("svm/"+k+"_"+v[0]+".txt","w")
  X = normalize(array_counts)
  y = np.array(labels) 
  print len(y)
  print X.shape[0]

  val_range  = [0.0001*pow(10,i) for i in range(8)]
  val_range += [0.0003*pow(10,i) for i in range(8)]
  val_range += [0.0005*pow(10,i) for i in range(8)]
  val_range += [0.0007*pow(10,i) for i in range(8)]
  val_range += [0.0009*pow(10,i) for i in range(8)]
  
  parameters = { 'C': val_range}

  for cval in val_range:
    clf = svm.LinearSVC(C=cval)
    cv = StratifiedShuffleSplit(y, n_iter=50, test_size=0.15, random_state=123)
    scores = cross_validation.cross_val_score(clf, X, y, cv=cv)
    #print scores
    print "C=%.4f: %.4f" % (cval, np.array(scores).mean())
    f.write("C=%.4f: %.4f" % (cval, np.array(scores).mean())+"\n")