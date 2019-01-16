# -*- coding: utf-8 -*-
"""
QUESTION 1
"""
print("----------- QUESTION 1 -----------")
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups


newsgroups_train = fetch_20newsgroups(subset='train')
dictt = {}
for i in newsgroups_train.target_names:
		training_data = fetch_20newsgroups(subset='train', categories=[i])
		dictt[i] = len(training_data.data)

fig,ax = plt.subplots()
plt.bar(list(newsgroups_train.target_names), list(dictt.values()))
labels = ax.get_xticklabels()
plt.setp(labels, rotation=20, fontsize=10)
plt.xlabel('Categories')
plt.ylabel('# Documents')
plt.title('Histogram of training documents')
plt.show()

"""
plt.hist(newsgroups_train.target)
plt.xticks(range(20), newsgroups_train.target_names)
locs, labels = plt.xticks()
plt.setp(labels, rotation=25)
plt.xlabel('Categories')
plt.ylabel('# Documents')
plt.title('Histogram of training documents')
plt.grid(True)
plt.show()
"""

"""
QUESTION 2
"""
print("----------- QUESTION 2 -----------")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

categories = ['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'rec.autos', 'rec.motorcycles',
'rec.sport.baseball', 'rec.sport.hockey']
train_dataset = fetch_20newsgroups(subset = 'train', categories = categories, 
                                   shuffle = True, random_state = None)
test_dataset = fetch_20newsgroups(subset = 'test', categories = categories, 
                                  shuffle = True, random_state = None)



vectorizer = CountVectorizer(min_df=3, stop_words='english')
X_train_counts = vectorizer.fit_transform(train_dataset.data)
X_train_counts.toarray()

X_test_counts = vectorizer.transform(test_dataset.data)
X_test_counts.toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print('Shape of TF-IDF train matrices subset: ')
print(X_train_tfidf.shape)
print()

X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
print('Shape of TF-IDF test matrices subset: ')
print(X_test_tfidf.shape)


# Add LEMMATIZATION: check doc on ccle (https://ccle.ucla.edu/mod/resource/view.php?id=2277904)
