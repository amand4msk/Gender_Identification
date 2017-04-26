## Seminar Project

Identifying the gender of Twitter users using style-based features and the content-based features in combination with a Support Vector Machine (SVM) classifier.

## content.py

This file contains the extraction of the content-based features and the experiments using the SVM classifier. We examined four style-based features: counts of mentions, hashTags, Urls and punctuation signs features.

## structural.py

This file contains the extraction of the structural-based features and the experiments using the SVM classifier. We examined tf-idf of n-grams, bag of words n-grams, part of speech n-grams.

## preprocessing.py

This file contains the appropriate preprocessing steps that followed for the content-based features (removing twitter bias such as @mentions, hashtags, URLs, ‘@username’ and  stripped of the character ‘#’ from the hashtags). There was no preprocessing done for style-based features.

## classifier.py 

SVM classifier with a linear kernel.

## svm

The accuracies of different C parameters for each feature.

## stanford-postagger-2015-04-20

Create Part of speech files for each user using the Stanford code.

## pos

This folder contains the Part of speech files for each user.

## preprocessing

This folder contains the preprocessing files for each user.

## PAN_2015

This folder contains the tweets collected from the PAN-AP-2015 corpus from Twitter in English.

## Counts.txt, labels.txt

These files contain the counts for each style-based feature and the true labels for each user.

## Run experiments

Extract counts of mentions, hashTags, Urls and punctuation signs features:
```
python structural.py --extract
``` 

Run the experiments using the style-based features:
```
python structural.py

```


Run the experiments using the bag of unigrams, bigrams, trigrams:
```
python content.py --bag uni

python content.py --bag bi

python content.py --bag tri

python content.py --bag all
```
For running the experiments using the tfidf n-grams: `--tfidf uni`, `--tfidf bi`, `--tfidf tri`, `--tfidf all`
For running the experiments using the pos n-grams: `--pos uni`, `--pos bi`, `--pos tri`, `--pos all`
For plotting the C parameters and the accuracy for each feature: `python content.py --plot`


