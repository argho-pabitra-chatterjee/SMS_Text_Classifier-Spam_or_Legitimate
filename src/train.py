import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split 
from sklearn import metrics

csv_dataset = pd.read_csv('../data/spam.csv', encoding='latin-1')

del csv_dataset['Unnamed: 2']
del csv_dataset['Unnamed: 3']
del csv_dataset['Unnamed: 4']

csv_dataset.columns = ['class', 'data']

def table(df, col):
    return df.groupby(col).count()


table(csv_dataset,"class")

print(csv_dataset.head())


# converting responses to int
csv_dataset.loc[csv_dataset['class']=='ham', 'class'] = 0
csv_dataset.loc[csv_dataset['class']=='spam', 'class'] = 1

print(csv_dataset.head())


## No. of examples belonging to class "SPAM" is far too less compared to number of examples for class "HAM"
## seperating dependent and independent variables
y_class = csv_dataset.pop('class')
X_data  = csv_dataset['data'].str.strip()

## splitting dataset
x_train, x_test, y_train, y_test = train_test_split(X_data, y_class, test_size = 0.3, stratify=y_class)


print("=======================")
print(y_train.groupby(y_train).count())
print(y_test.groupby(y_test).count())
print("=======================")
print(type(x_train).__name__)
print(x_train.shape)
print(type(x_train.to_frame()).__name__)
print((x_train.to_frame()).shape)

x_train_1 = x_train
x_test_1 = x_test
y_train_1 = y_train
y_test_1 = y_test

x_train = x_train.to_frame()
x_test = x_test.to_frame()
y_train = y_train.to_frame()
y_test = y_test.to_frame()


train = x_train.join(y_train)
test = x_test.join(y_test)

x_train_merged_doc = train.groupby('class')['data'].apply('.'.join).reset_index()


## computing NULL accuracyprint(y_test["class"].value_counts())
#calculate null accuracy for binary classifier
print("\nNull Accuracy : " + str(max((y_test["class"].mean()), (1-(y_test["class"].mean())))))

## creating TF-IDF feature vectors_are_parallel## creating TFIDF features
#vectorizer = TfidfVectorizer(ngram_range=(1, 2),  sublinear_tf = True, stop_words='english')
vectorizer = TfidfVectorizer(ngram_range=(1, 2),sublinear_tf = True, stop_words='english', max_df=0.5)
#vectorizer.fit(raw_documents = x_train_merged_doc['data'], y = x_train_merged_doc['class'])
#features_train_transformed   = vectorizer.transform(raw_documents=x_train, copy = False)
features_train_transformed    = vectorizer.fit_transform(raw_documents=x_train['data'])
features_test_transformed     = vectorizer.transform(raw_documents=x_test['data'], copy = False)

print('=======================')
print(features_train_transformed.shape)
print('=======================')
print(features_test_transformed.shape)


## Using the classic Naive Bayes classifier which is a proven classification
## technique for SPAM / HAM problem
mnb = MultinomialNB(alpha=1e-10, fit_prior=True)
mnb.fit(X=features_train_transformed, y=np.char.mod('%d',y_train['class'].values))
pred_y = mnb.predict(features_test_transformed)
actual_y = np.char.mod('%d',y_test['class'].values)
cf_matrix = metrics.confusion_matrix(actual_y, pred_y)

TP = cf_matrix[1,1]
FN = cf_matrix[1,0]
FP = cf_matrix[0,1]
TN = cf_matrix[0,0]

# Classification Accuracy
#metrics.accuracy_score(actual_y, pred_y)
print("Classification accuracy : " + str(((TP+TN)/float(TP+TN+FP+FN))*100) + "%")

print("Classification Error : " + str(((FP+FN)/float(TP+TN+FP+FN))*100) + "%")

print("Sensitivity : " + str(((TP)/float(TP+FN))*100) + "%")

print("Specificity : " + str(TN/float(TN+FP)))

print("Precision : " + str(TP/float(TP+FP)))

print("Recall : " + str(TP/(TP+FN)))

## F1 score
precision = TP/float(TP+FP)
recall = TP/(TP+FN)
F1 = 2*precision*recall/(precision + recall)
print("F1 score : " + str(F1))