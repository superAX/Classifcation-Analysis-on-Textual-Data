# -*- coding: utf-8 -*-
"""
QUESTION 1
"""
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import numpy as np
np.random.seed(42)
import random
random.seed(42)

newsgroups_train = fetch_20newsgroups(subset='train')

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
QUESTION 2
"""
categories = ['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'rec.autos', 'rec.motorcycles',
'rec.sport.baseball', 'rec.sport.hockey']
train_dataset = fetch_20newsgroups(subset = 'train', categories = categories, 
                                   shuffle = True, random_state = None)
test_dataset = fetch_20newsgroups(subset = 'test', categories = categories, 
                                  shuffle = True, random_state = None)

