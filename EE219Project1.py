# -*- coding: utf-8 -*-

import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from nltk import pos_tag
from nltk.corpus import stopwords
from string import punctuation
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import itertools
plt.rcParams['figure.figsize'] = [12, 8]  #set image size for display

"""
QUESTION 1
"""
print("----------- QUESTION 1 -----------")
from matplotlib import pyplot as plt

newsgroups_train = fetch_20newsgroups(subset='train')
cat_Ndocs = {}
for i in newsgroups_train.target_names:
		training_data = fetch_20newsgroups(subset='train', categories=[i])
		cat_Ndocs[i] = len(training_data.data)

fig,ax = plt.subplots()
plt.barh(list(newsgroups_train.target_names), list(cat_Ndocs.values()))
#labels = ax.get_yticklabels()
#plt.setp(labels, rotation=20, fontsize=10)
plt.xlabel('No. of Documents')
plt.ylabel('Categories')
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
lsi = TruncatedSVD(n_components=50, random_state=42) 
X_train_LSI = lsi.fit_transform(X_train_tfidf)
X_test_LSI = lsi.transform(X_test_tfidf)
print(X_train_LSI.shape)
print(X_test_LSI.shape)


# NMF dimensionality reduction
nmf = NMF(n_components=50, random_state=42)
X_train_NMF = nmf.fit_transform(X_train_tfidf)
X_test_NMF = nmf.transform(X_test_tfidf)
print(X_train_NMF.shape)
print(X_test_NMF.shape)

# compare LSI & NMF

# for LSI
VT = lsi.components_
sum_train_LSI = np.sum(np.square(X_train_tfidf - X_train_LSI.dot(VT)))
sum_test_LSI = np.sum(np.square(X_test_tfidf - X_test_LSI.dot(VT)))
print(sum_train_LSI)
print(sum_test_LSI)

# for NMF
H = nmf.components_
sum_train_NMF = np.sum(np.array(X_train_tfidf - X_train_NMF.dot(H))**2)
sum_test_NMF = np.sum(np.array(X_test_tfidf - X_test_NMF.dot(H))**2)
print(sum_train_NMF)
print(sum_test_NMF)


# Plot ROC, for q4, q5, q6 
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

# Plot the confusion_matrix, for q4, q5, q6
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

# Classify the dataset into 2 categories
cat_0 = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
cat_1 = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
classes = ['Computer Technology', 'Recreational Activity']    
def classify(dataset):
    cat = []
    for i in dataset.target:
        if(i < 4):
            cat.append(0)
        else: 
            cat.append(1)
    return(cat)


y_train = classify(train_dataset)    
y_test = classify(test_dataset) 
           
"""
QUESTION 4
"""
print("----------- QUESTION 4 -----------")           

svm_hard = LinearSVC(C = 1000, random_state = 42)
svm_soft = LinearSVC(C = 0.0001, random_state = 42)

def question4(svm):
    svm.fit(X_train_LSI, y_train)	
    
    # ----------------------
    # ROC Curves
    test_score = svm.decision_function(X_test_LSI)
    fpr, tpr, threshold = roc_curve(y_test, test_score)
    plot_roc(fpr, tpr)
    
    # ----------------------
    # Metrics
    y_test_predict = svm.predict(X_test_LSI)
    
    confusionMatrix = confusion_matrix(y_test, y_test_predict)
    accuracy = accuracy_score(y_test, y_test_predict)
    recall = recall_score(y_test, y_test_predict)
    precision = precision_score(y_test, y_test_predict)
    f1_score = 2/((1/recall) + (1/precision))
    
    print('Confusion Matrix: ')
    plt.figure()
    #plot_confusion_matrix(confusionMatrix, classes=classes, normalize=True, title='Normalized confusion matrix')
    plot_confusion_matrix(confusionMatrix, classes=classes,title='confusion matrix without normalization')
    plt.show()
    print('Accuracy:', accuracy)
    print('Recall:', recall)
    print('Precision:', precision)
    print('F1-Score:', f1_score)


print('\nHard Margin ----------------------------')
question4(svm_hard)
print('\nSoft Margin ----------------------------')
question4(svm_soft)

print("--- PART B: Cross-Validation ---")

C_best = 0;
max_score = 0;

for i in range(-3, 4): 
    C = 10**i;
    clf = LinearSVC(C = C,random_state=42);
    scores = cross_val_score(clf, X_train_LSI, train_dataset.target, cv=5).mean()
    #print('Mean score: ', scores)
    
    if(scores > max_score):
        max_score = scores
        C_best = C
    
print('Best value of \u03B3 = ', C_best, ' is obtained with cross validation score of ', max_score)
svm_best = LinearSVC(C = C_best,random_state=42)
print('\nBest SVM ----------------------------')
question4(svm_best)

"""
QUESTION 5
"""
print("----------- QUESTION 5 -----------")
def logistic_train_plot_score(clf):
    # train a model
    clf.fit(X_train_LSI,y_train)
    
    # plot ROC curve
    test_score = clf.decision_function(X_test_LSI)
    fpr, tpr, threshold = roc_curve(y_test, test_score)
    plot_roc(fpr, tpr)   
    
    # calculate scores
    y_test_predict = clf.predict(X_test_LSI)
    
    confusionMatrix = confusion_matrix(y_test, y_test_predict)
    accuracy = accuracy_score(y_test, y_test_predict)
    recall = recall_score(y_test, y_test_predict)
    precision = precision_score(y_test, y_test_predict)
    f1_score = 2/((1/recall) + (1/precision))   
    print('Confusion Matrix: ')
    plt.figure()
    plot_confusion_matrix(confusionMatrix, classes=classes, title='confusion matrix without normalization')
    #plot_confusion_matrix(confusionMatrix, classes=classes, normalize=True,title='normalized confusion matrix')
    plt.show()
    print('Accuracy:', accuracy)
    print('Recall:', recall)
    print('Precision:', precision)
    print('F1:', f1_score)

# ROC curve and scores for logistic regression without regularization
print("--------ROC curve and scores without regularization--------")
clf_logistic = LogisticRegression(C=10**8,random_state=42)
logistic_train_plot_score(clf_logistic)
print('\n\n\n\n')



# 5-fold cross validation to find the best C for L1 regularization and L2 regularization
def score_with_k(pen):
    score_list = []
    k_list = range(-3,4)
    for k in k_list:
        clf = LogisticRegression(penalty=pen, C=10**k, random_state=42)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        scores = cross_val_score(clf, X_train_LSI, y_train, cv=cv, scoring='f1_macro')
        avg_score = np.average(scores)
        score_list.append(avg_score)
    print("the score list is ",score_list)
    best_k_loc = np.argmax(score_list)
    best_k = k_list[best_k_loc]
    return best_k

# best k for L1-regularization 
print("--------find the best k for L1 regularization--------")
best_k_L1 = score_with_k('l1')
print("the best k for L1 regularization is ",best_k_L1)
print('\n\n\n\n')

# best k for L2-regularization
print("--------find the best k for L2 regularization--------")
best_k_L2 = score_with_k('l2')
print("the best k for L1 regularization is ",best_k_L2)
print('\n\n\n\n')
    
    
# ROC curve and scores for logistic regression with L1 regularization
print("--------ROC curve and scores for L1 regularization with k=1--------")
clf_logistic_L1 = LogisticRegression(penalty='l1',C=10**best_k_L1, random_state=42)
logistic_train_plot_score(clf_logistic_L1)
print('\n\n\n\n')

# ROC curve and scores for logistic regression with L2 regularization
print("--------ROC curve and scores for L2 regularization with k=1--------")
clf_logistic_L2 = LogisticRegression(penalty='l2',C=10**best_k_L2, random_state=42)
logistic_train_plot_score(clf_logistic_L2)

"""
QUESTION 6
"""
print("----------- QUESTION 6 -----------")

# plot ROC curve
classifierNB_score = GaussianNB().fit(X_train_LSI, y_train).predict_proba(X_test_LSI)
fpr, tpr, _ = roc_curve(y_test, classifierNB_score[:,1])
plot_roc(fpr, tpr)

# Classify by Gaussian Naive Bayes 
classifierNB = GaussianNB().fit(X_train_LSI, y_train).predict(X_test_LSI)

# Compute confusion matrix
cnf_matrix_NB = confusion_matrix(y_test, classifierNB)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_NB, classes=classes, normalize=True, title='Normalized confusion matrix')
plt.show()

# Calculate 4 scores 
accuracy_NB = accuracy_score(y_test, classifierNB)
print("Accuracy: ", accuracy_NB)
recall_NB = recall_score(y_test, classifierNB, average='weighted')
print("Recall: ",recall_NB)
precision_NB = precision_score(y_test, classifierNB, average='weighted')
print("Precision: ",precision_NB)
f1_NB = f1_score(y_test, classifierNB, average='weighted')
print("F1: ",precision_NB)
print('\n\n\n\n')

"""
QUESTION 7
"""
print("----------- QUESTION 7 -----------")
from sklearn.pipeline import Pipeline

"""
QUESTION 8
"""

print("----------- QUESTION 8 -----------")

# import data and generate LSI-reduced TF-IDF matrix
categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
train_dataset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test_dataset = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

stop_words_skt = text.ENGLISH_STOP_WORDS
stop_words_en = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_en),set(punctuation),set(stop_words_skt))

ps = nltk.stem.PorterStemmer()
wnl = nltk.wordnet.WordNetLemmatizer()     

def stem_rmv_punc(doc):
    return (word for word in lemmatize_sent(analyzer(doc)) if word not in combined_stopwords and not word.isdigit())

analyzer = text.CountVectorizer().build_analyzer()
vectorizer = text.CountVectorizer(min_df=3, stop_words='english', analyzer=stem_rmv_punc, token_pattern = r'(?u)\b[A-Za-z][A-Za-z]+\b')

X_train_counts = vectorizer.fit_transform(train_dataset.data)
X_train_counts.toarray()
X_test_counts = vectorizer.transform(test_dataset.data)
X_test_counts.toarray()
X_train_tfidf = text.TfidfTransformer().fit_transform(X_train_counts)
print('Shape of TF-IDF train matrices subset: ')
print(X_train_tfidf.shape)
X_test_tfidf = text.TfidfTransformer().fit_transform(X_test_counts)
print('Shape of TF-IDF test matrices subset: ')
print(X_test_tfidf.shape)

#LSI: 
from sklearn.decomposition import TruncatedSVD
lsi = TruncatedSVD(n_components=50, random_state=42) 
X_train_LSI = lsi.fit_transform(X_train_tfidf)
X_test_LSI = lsi.transform(X_test_tfidf)

#NMF:
from sklearn.decomposition import NMF
nmf = NMF(n_components=50, random_state=42)
X_train_NMF = nmf.fit_transform(X_train_tfidf)
X_test_NMF = nmf.transform(X_test_tfidf)

X_train_target = train_dataset.target
X_test_target = test_dataset.target
    
# Multiclass SVM (one vs. one)
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier

print("Multiclass SVM one vs one (with LSI): ")

clf_1v1 = OneVsOneClassifier(svm.SVC(C=1000)).fit(X_train_LSI, X_train_target)
X_predict_1v1 = clf_1v1.predict(X_test_LSI)

cf_mat = confusion_matrix(X_test_target, X_predict_1v1)
np.set_printoptions(precision=2)

#Plot confusion matrices -- unnormalized and normalized
plt.figure()
plot_confusion_matrix(cf_mat, classes=categories,
                      title='Confusion Matrix (1v1 SVM, LSI)')
plt.figure()
plot_confusion_matrix(cf_mat, classes=categories, normalize=True,
                      title='Confusion Matrix (Normalized, 1v1 SVM, LSI)')
plt.show()

# Compute Metrics: 
accuracy = accuracy_score(X_test_target, X_predict_1v1)
recall = recall_score(X_test_target, X_predict_1v1, average='weighted')
precision = precision_score(X_test_target, X_predict_1v1, average='weighted')
f1 = f1_score(X_test_target, X_predict_1v1, average='weighted')


print("Accuracy:  ", accuracy)
print("Recall:    ", recall)
print("Precision: ", precision)
print("F1: ", precision_NB)

print("Multiclass SVM one vs one (with NMF): ")

clf_1v1 = OneVsOneClassifier(svm.SVC(C=1000)).fit(X_train_NMF, X_train_target)
X_predict_1v1 = clf_1v1.predict(X_test_NMF)

cf_mat = confusion_matrix(X_test_target, X_predict_1v1)
np.set_printoptions(precision=2)

#Plot confusion matrices -- unnormalized and normalized
plt.figure()
plot_confusion_matrix(cf_mat, classes=categories,
                      title='Confusion Matrix (1v1 SVM, NMF)')
plt.figure()
plot_confusion_matrix(cf_mat, classes=categories, normalize=True,
                      title='Confusion Matrix (Normalized, 1v1 SVM, NMF)')
plt.show()

# Compute Metrics: 
accuracy = accuracy_score(X_test_target, X_predict_1v1)
recall = recall_score(X_test_target, X_predict_1v1, average='weighted')
precision = precision_score(X_test_target, X_predict_1v1, average='weighted')
f1 = f1_score(X_test_target, X_predict_1v1, average='weighted')

print("Accuracy:  ", accuracy)
print("Recall:    ",recall)
print("Precision: ",precision)
print("F1: ", f1)

# Multiclass SVM (one vs. the rest)

from sklearn.multiclass import OneVsRestClassifier

print("Multiclass SVM one vs the rest (with LSI): ")

clf_1vR = OneVsRestClassifier(svm.SVC(C=1000)).fit(X_train_LSI, X_train_target)
X_predict_1vR = clf_1vR.predict(X_test_LSI)

cf_mat = confusion_matrix(X_test_target, X_predict_1vR)
np.set_printoptions(precision=2)

#Plot confusion matrices -- unnormalized and normalized
plt.figure()
plot_confusion_matrix(cf_mat, classes=categories,
                      title='Confusion Matrix (1vR SVM, LSI)')
plt.figure()
plot_confusion_matrix(cf_mat, classes=categories, normalize=True,
                      title='Confusion Matrix (Normalized, 1vR SVM, LSI)')
plt.show()

# Compute Metrics: 
accuracy = accuracy_score(X_test_target, X_predict_1vR)
recall = recall_score(X_test_target, X_predict_1vR, average='weighted')
precision = precision_score(X_test_target, X_predict_1vR, average='weighted')
f1 = f1_score(X_test_target, X_predict_1v1, average='weighted')

print("Accuracy:  ", accuracy)
print("Recall:    ", recall)
print("Precision: ", precision)
print("F1: ", f1)

print("Multiclass SVM one vs the rest (with NMF): ")

clf_1vR = OneVsRestClassifier(svm.SVC(C=1000)).fit(X_train_NMF, X_train_target)
X_predict_1vR = clf_1vR.predict(X_test_NMF)

cf_mat = confusion_matrix(X_test_target, X_predict_1vR)
np.set_printoptions(precision=2)

#Plot confusion matrices -- unnormalized and normalized
plt.figure()
plot_confusion_matrix(cf_mat, classes=categories,
                      title='Confusion Matrix (1vR SVM, NMF)')
plt.figure()
plot_confusion_matrix(cf_mat, classes=categories, normalize=True,
                      title='Confusion Matrix (Normalized, 1vR SVM, NMF)')
plt.show()

# Compute Metrics: 
accuracy = accuracy_score(X_test_target, X_predict_1vR)
recall = recall_score(X_test_target, X_predict_1vR, average='weighted')
precision = precision_score(X_test_target, X_predict_1vR, average='weighted')
f1 = f1_score(X_test_target, X_predict_1v1, average='weighted')

print("Accuracy:  ", accuracy)
print("Recall:    ",recall)
print("Precision: ",precision)
print("F1: ", f1)

# Multiclass Naive Bayes with NMF

from sklearn.naive_bayes import MultinomialNB
print("Multiclass Naive Bayes (with NMF): ")

mnb = MultinomialNB()
mnb.fit(X_train_NMF, X_train_target)
X_predict_NB = mnb.predict(X_test_NMF)

cf_mat = confusion_matrix(X_test_target, X_predict_NB)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cf_mat, classes=categories,
                      title='Confusion Matrix (NB)')

plt.figure()
plot_confusion_matrix(cf_mat, classes=categories, normalize=True,
                      title='Confusion Matrix (Normalized, NB)')

plt.show()

# Compute Metrics: 
accuracy = accuracy_score(X_test_target, X_predict_NB)
recall = recall_score(X_test_target, X_predict_NB, average='weighted')
precision = precision_score(X_test_target, X_predict_NB, average='weighted')
f1 = f1_score(X_test_target, X_predict_1v1, average='weighted')
print("Accuracy:  ", accuracy)
print("Recall:    ", recall)
print("Precision: ", precision)
print("F1 :", f1)

