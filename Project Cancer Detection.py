#!/usr/bin/env python
# coding: utf-8

# # Project Cancer Detection
# 
# # Breast Cancer Wisconsin (Disgnostic) Data Set
# 
# [Source: UCI](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
# 
# [Data Set info](http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names)
7. Attribute Information: (class attribute has been moved to last column)
   #  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
# In[2]:


import numpy as np
import pandas as pd


# In[4]:


col = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 
       'Uniformity of Cell Shape', 'Marginal Adhesion', 
       'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
       'Normal Nucleoli', 'Mitoses', 'Class']
df = pd.read_csv("breast-cancer-wisconsin.data.csv", names=col,
                 header=None)
df.head()


# # Data Pre-processing

# In[6]:


np.where(df.isnull())


# In[7]:


df.info()


# In[8]:


df['Bare Nuclei'].describe()


# In[9]:


df['Bare Nuclei'].value_counts()


# How do we drop the `?`

# In[10]:


df[df['Bare Nuclei'] == "?"]


# In[11]:


df['Class'].value_counts()


# In[12]:


df['Bare Nuclei'].replace("?", np.NAN, inplace=True)
df = df.dropna()


# Note that for class: 2 is benign, 4 is for malignant
# 
# $$\frac{\text{df["Class"]}}{2} - 1$$

# In[13]:


df['Bare Nuclei'].value_counts()


# In[14]:


df['Class'] = df['Class'] / 2 - 1


# In[15]:


df['Class'].value_counts()


# In[16]:


df.columns


# In[17]:


df.info()


# In[18]:


X = df.drop(['id', 'Class'], axis=1)
X_col = X.columns


# In[19]:


y = df['Class']


# In[20]:


from sklearn.preprocessing import StandardScaler


# In[21]:


X = StandardScaler().fit_transform(X.values)


# Training

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


df1 = pd.DataFrame(X, columns=X_col)


# In[24]:


df1.head()


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(df1, y,
                                                    train_size=0.8,
                                                    random_state=42)


# In[26]:


from sklearn.preprocessing import MinMaxScaler
pd.DataFrame(MinMaxScaler().fit_transform(df.drop(['id', 'Class'], axis=1).values), columns=X_col).head()


# In[27]:


from sklearn.neighbors import KNeighborsClassifier


# In[28]:


knn = KNeighborsClassifier(n_neighbors=5,
                           p=2, metric='minkowski')


# In[29]:


.knn.fit(X_train, y_train)


# In[ ]:


from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[ ]:


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))        


# In[30]:


print_score(knn, X_train, y_train, X_test, y_test, train=True)


# In[31]:


print_score(knn, X_train, y_train, X_test, y_test, train=False)


# # Grid Search

# In[32]:


from sklearn.model_selection import GridSearchCV


# In[33]:


knn.get_params()


# In[33]:


params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}


# In[34]:


.


grid_search_cv = GridSearchCV(KNeighborsClassifier(),
                              params, 
                              n_jobs=-1,
                              verbose=1)


# In[1]:


grid_search_cv.fit(X_train, y_train)


# In[36]:


grid_search_cv.best_estimator_


# In[37]:


print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=True)


# In[38]:


print_score(grid_search_cv, X_train, y_train, X_test, y_test, train=False)


# In[39]:


grid_search_cv.best_params_


# In[40]:


grid_search_cv.cv_results_['mean_train_score']


# In[41]:


grid_search_cv.cv_results_


# SVM, Random Forest, XGBoost

# In[42]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
print_score(clf, X_train, y_train, X_test, y_test, train=True)
print_score(clf, X_train, y_train, X_test, y_test, train=False)


# In[43]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
print_score(clf, X_train, y_train, X_test, y_test, train=True)
print_score(clf, X_train, y_train, X_test, y_test, train=False)


# In[44]:


import xgboost as xgb
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)
print_score(clf, X_train, y_train, X_test, y_test, train=True)
print_score(clf, X_train, y_train, X_test, y_test, train=False)


# ***
