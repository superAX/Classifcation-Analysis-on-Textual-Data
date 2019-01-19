# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
import nltk
nltk.download('wordnet')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from string import punctuation
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups

"""
QUESTION 1
"""
print("----------- QUESTION 1 -----------")
newsgroups_train = fetch_20newsgroups(subset='train',random_state=42)
dictt = {}
for i in newsgroups_train.target_names:
		training_data = fetch_20newsgroups(subset='train', categories=[i],random_state=42)
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
QUESTION 2
"""
print("----------- QUESTION 2 -----------")

categories = ['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'rec.autos', 'rec.motorcycles',
'rec.sport.baseball', 'rec.sport.hockey']
train_dataset = fetch_20newsgroups(subset = 'train', categories = categories, 
                                   shuffle = True, random_state = 42)
test_dataset = fetch_20newsgroups(subset = 'test', categories = categories, 
                                  shuffle = True, random_state = 42)

stop_words_skt = text.ENGLISH_STOP_WORDS
stop_words_en = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_en),set(punctuation),set(stop_words_skt))

#  Converts Penn Treebank tags to WordNet.
def penn2morphy(penntag):
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'

    
wnl = nltk.wordnet.WordNetLemmatizer()    

def lemmatize_sent(list_word):
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) for word, tag in pos_tag(list_word)]

def stem_rmv_punc(doc):
    return (word for word in lemmatize_sent(analyzer(doc)) if word not in combined_stopwords and not word.isdigit())

analyzer = text.CountVectorizer().build_analyzer()

# Remove stopword, number and set min_df = 3
vectorizer = text.CountVectorizer(min_df=3, stop_words='english', analyzer=stem_rmv_punc, token_pattern = r'(?u)\b[A-Za-z][A-Za-z]+\b')

# For the train data
X_train_counts = vectorizer.fit_transform(train_dataset.data)
X_train_counts.toarray()

# For the test data
X_test_counts = vectorizer.transform(test_dataset.data)
X_test_counts.toarray()

# The following is the tfidf part
# For the train data
X_train_tfidf = text.TfidfTransformer().fit_transform(X_train_counts)
print('Shape of TF-IDF train matrices subset: ')
print(X_train_tfidf.shape)

# For the test data
X_test_tfidf = text.TfidfTransformer().fit_transform(X_test_counts)
print('Shape of TF-IDF test matrices subset: ')
print(X_test_tfidf.shape)

"""
QUESTION 3
"""
print("----------- QUESTION 3 -----------")

# LSI dimensionality reduction

from sklearn.decomposition import TruncatedSVD

lsi = TruncatedSVD(n_components=50, random_state=42) 
X_train_LSI = lsi.fit_transform(X_train_tfidf)
X_test_LSI = lsi.transform(X_test_tfidf)
print(X_train_LSI.shape)
print(X_test_LSI.shape)


# NMF dimensionality reduction

from sklearn.decomposition import NMF

nmf = NMF(n_components=50, random_state=42)
X_train_NMF = nmf.fit_transform(X_train_tfidf)
X_test_NMF = nmf.transform(X_test_tfidf)
print(X_train_NMF.shape)
print(X_test_NMF.shape)


# compare LSI & NMF

# for NMF
H = nmf.components_
sum_train_NMF = np.sum(np.array(X_train_tfidf - X_train_NMF.dot(H))**2)
sum_test_NMF = np.sum(np.array(X_test_tfidf - X_test_NMF.dot(H))**2)
print(sum_train_NMF)
print(sum_test_NMF)

# for LSI

"""
QUESTION 6
"""
print("----------- QUESTION 6 -----------")

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
import itertools
from scipy import interp

# Plot ROC 
def plot_roc(fpr, tpr):
    fig, ax = plt.subplots()

    roc_auc = auc(fpr,tpr)

    ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate',fontsize=15)
    ax.set_ylabel('True Positive Rate',fontsize=15)

    ax.legend(loc="lower right")

    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(15)

# Plot the confusion_matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Classify by Gaussian Naive Bayes 
classifierNB = GaussianNB().fit(X_train_LSI, train_dataset.target).predict(X_test_LSI)

# plot ROC curve
classifierNB_score = GaussianNB().fit(X_train_LSI, train_dataset.target).predict_proba(X_test_LSI)
BinaryLabel = label_binarize(test_dataset.target, classes=np.unique(test_dataset.target))  # Binarize the output
n_classes = BinaryLabel.shape[1]
print("Micro-Average ROC Curve")
fpr, tpr, _ = roc_curve(BinaryLabel.ravel(), classifierNB_score.ravel())
plot_roc(fpr, tpr)
print("Marco-Average ROC Curve")
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(BinaryLabel[:, i], classifierNB_score[:, i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
plot_roc(all_fpr, mean_tpr)

# Compute confusion matrix
cnf_matrix_NB = confusion_matrix(test_dataset.target, classifierNB)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_NB, classes=test_dataset.target_names, normalize=True, title='Normalized confusion matrix')
plt.show()

# Calculate 4 scores 
accuracy_NB = accuracy_score(test_dataset.target, classifierNB)
print("The accuracy score is  ", accuracy_NB)
recall_NB = recall_score(test_dataset.target, classifierNB, average='weighted')
print("The recall score is ",recall_NB)
precision_NB = precision_score(test_dataset.target, classifierNB, average='weighted')
print("The precision score is: ",precision_NB)
f1_NB = f1_score(test_dataset.target, classifierNB, average='weighted')
print("The F1 score is: ",precision_NB)