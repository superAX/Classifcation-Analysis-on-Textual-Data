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
import numpy as np
from sklearn.datasets import fetch_20newsgroups

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

#%%
"""
QUESTION 2
"""
print("----------- QUESTION 2 -----------")


categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 
              'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 
              'rec.sport.hockey']
train_dataset = fetch_20newsgroups(subset = 'train', categories = categories, 
                                   shuffle = True, random_state = None)
test_dataset = fetch_20newsgroups(subset = 'test', categories = categories, 
                                  shuffle = True, random_state = None)

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

ps = nltk.stem.PorterStemmer()
wnl = nltk.wordnet.WordNetLemmatizer()    

ps = nltk.stem.PorterStemmer()
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

#%%
"""
QUESTION 3
"""
print("----------- QUESTION 3 -----------")

# LSI dimensionality reduction

from sklearn.decomposition import TruncatedSVD

lsi = TruncatedSVD(n_components=50, random_state=42) 
X_train_LSI = lsi.fit_transform(X_train_tfidf)
X_test_LSI = lsi.transform(X_test_tfidf)
print('Shape of reduce TF-IDF matrix with LSI: ')
print('Train: ', X_train_LSI.shape)
print('Test: ', X_test_LSI.shape)


# NMF dimensionality reduction

from sklearn.decomposition import NMF

nmf = NMF(n_components=50, random_state=42)
X_train_NMF = nmf.fit_transform(X_train_tfidf)
X_test_NMF = nmf.transform(X_test_tfidf)
print('Shape of reduce TF-IDF matrix with NMF: ')
print('Train: ', X_train_NMF.shape)
print('Test: ', X_test_NMF.shape)


# Compare LSI & NMF

# for NMF
H = nmf.components_
sum_train_NMF = np.sum(np.array(X_train_tfidf - X_train_NMF.dot(H))**2)
sum_test_NMF = np.sum(np.array(X_test_tfidf - X_test_NMF.dot(H))**2)
print(sum_train_NMF)
print(sum_test_NMF)

# for LSI


#%%
"""
QUESTION 4
"""
print("----------- QUESTION 4 -----------")
print("--- PART A ---")
from sklearn.svm import LinearSVC
from sklearn.metrics import auc, confusion_matrix, recall_score, roc_curve, precision_score, accuracy_score

cat_0 = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
cat_1 = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

classes = ['Computer Technology', 'Recreational Activity']

# Classify the dataset into 2 categories
def classify(dataset):
    cat = []
    for i in dataset.target:
        if(i < 4):
            cat.append(0)
        else: 
            cat.append(1)
    return(cat)


X_train_cat = classify(train_dataset)    
X_test_cat = classify(test_dataset)              

svm_hard = LinearSVC(C = 1000)
svm_soft = LinearSVC(C = 0.0001)

def question4(svm_train):
    svm_train.fit(X_train_LSI, X_train_cat)	
    
    # ----------------------
    # ROC Curves
    test_score = svm_train.decision_function(X_test_LSI)
    FPR, TPR, threshold = roc_curve(X_test_cat, test_score)
    
    # Plot ROC
    fig,ax = plt.subplots()
    plt.plot(FPR, TPR)
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (LSI)')
    plt.grid()
    plt.show()
    
    # ----------------------
    # Metrics
    test_predict = svm_train.predict(X_test_LSI)
    
    confusionMatrix = confusion_matrix(X_test_cat, test_predict)
    accuracy = accuracy_score(X_test_cat, test_predict)
    recall = recall_score(X_test_cat, test_predict)
    precision = precision_score(X_test_cat, test_predict)
    f1_score = 2/((1/recall) + (1/precision))
    
    print('Confusion Matrix: ')
    print(confusionMatrix)
    print('Accuracy:', accuracy)
    print('Recall:', recall)
    print('Precision:', precision)
    print('F1-Score:', f1_score)


print('\nHard Margin ----------------------------')
question4(svm_hard)
print('\nSoft Margin ----------------------------')
question4(svm_soft)
print()

print("--- PART B: Cross-Validation ---")
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

C_best = 0;
max_score = 0;

for i in range(-3, 4): 
    C = 10**i;
    clf = LinearSVC(C = C);
    scores = cross_val_score(clf, X_train_LSI, train_dataset.target, cv=5).mean()
    #print('Mean score: ', scores)
    
    if(scores > max_score):
        max_score = scores
        C_best = C
    
print('Best value of \u03B3 = ', C_best, ' is obtained with cross validation score of ', max_score)
svm_best = LinearSVC(C = C_best)
print('\nBest SVM ----------------------------')
question4(svm_best)


# TO DO:
# - Change LinearSVC to SVC()
# - Fix bug question 2: there should be less words for test dataset
# - Fix bug question 4: weird results for Soft Margin