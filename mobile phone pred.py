#%%
#Import libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore') 
# %%
#import Models
from sklearn.model_selection import train_test_split
from sklearn import tree
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# %%
#More Models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# %%
#More Models
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.feature_selection import RFE
# %%
#Load Data into DataFrame
df_train = pd.read_csv('phone_train.csv')
df_test = pd.read_csv('phone_test.csv')
# %%
df_train.head()

# %%
df_train.columns
# %%
#Data Types
df_train.dtypes
# %%
#Missing data check
df_train.isnull().sum()
# %%
print("Rows:", len(df_train))
print("Columns:", df_train.shape[1])
# %%
df_train.describe()
# %%
#Distribution of variables
def plot_hist(variable):
    sns.kdeplot(df_train[variable], shade = True)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution".format(variable))
    plt.show()
# %%
variables = ['battery_power', 'int_memory', 'ram']
for v in variables:
    plot_hist(v)
# %%
labels = df_train['dual_sim'].value_counts().index
sizes = df_train['dual_sim'].value_counts().values

plt.figure(figsize = (8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of phones by sim cards: Dual = 1, Single = 0',color = 'black',fontsize = 15)
# %%
labels = ["3G-supported",'Not supported']
values=df_train['three_g'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()
# %%
labels4g = ["4G-supported",'Not supported']
values4g = df_train['four_g'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values4g, labels=labels4g, autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()
# %%
#Relationship Between variables
sns.jointplot(x='ram',y='price_range',data=df_train,color='red',kind='kde')
# %%
sns.pointplot(y="int_memory", x="price_range", data=df_train)
# %%
sns.boxplot(x="price_range", y="battery_power", data=df_train)
#%%
sns.swarmplot(x=df_train['price_range'],
              y=df_train['battery_power'])
# %%
plt.figure(figsize=(10,6))
df_train['fc'].hist(alpha=0.5,color='blue',label='Front camera')
df_train['pc'].hist(alpha=0.5,color='red',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')
# %%
#Model Building
y = df_train['price_range']
x = df_train.drop('price_range', axis = 1)
# %%
x.head()
# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state = 101)

# %%
#Logistics Regression
model_lgr = 'Logistic Regression'
lr = LogisticRegression()
model = lr.fit(X_train, y_train)
# %%
lr_acc_score = lr.score(X_test,y_test)
# %%
#Knn Model
model_knn = 'K-NeighborsClassifier'
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
# %%
knn_acc_score = knn.score(X_test,y_test)
# %%
#DecisionTree
model_dtc = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)
dt.fit(X_train, y_train)
# %%
dt_acc_score = dt.score(X_test, y_test)
# %%
#RandomForest
model_rfc = 'Random Forest Classfier'
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train,y_train)
# %%
rf_acc_score = rf.score(X_test, y_test)
# %%
#Naive Bayes
model_nb = 'Naive Bayes'
nb = GaussianNB()
nb.fit(X_train,y_train)
# %%
nb_acc_score = nb.score(X_test, y_test)
# %%
#SVC
model_svc = 'Support Vector Classifier'
svc =  SVC(kernel='rbf', C=2)
svc.fit(X_train, y_train)
# %%
svc_acc_score = svc.score(X_test, y_test)
# %%
model_ev = pd.DataFrame({'Model': ['Logistic Regression','Naive Bayes','Random Forest','K-Nearest Neighbour','Decision Tree','Support Vector Classifier'], 'Accuracy': [lr_acc_score*100,
                    nb_acc_score*100,rf_acc_score*100,knn_acc_score*100,dt_acc_score*100,svc_acc_score*100]})
# %%
model_ev
# %%
#Classification report knn
knn_predicted = knn.predict(X_test)
knn_conf_matrix = confusion_matrix(y_test, knn_predicted)

# %%
print(classification_report(y_test,knn_predicted))
print(knn_conf_matrix)
# %%
#Classification report svc
svc_predicted = svc.predict(X_test)
svc_conf_matrix = confusion_matrix(y_test, svc_predicted)
# %%
print(classification_report(y_test, svc_predicted))
print(svc_conf_matrix)
# %%
#Predicting Phone price using test data_set
df_test = df_test.drop('id', axis = 1)
# %%
df_test.head()
# %%
#Using Knn to predict
knn_pred_price = knn.predict(df_test)
# %%
knn_pred_price
# %%
#Using svc to predict
svc_pred_price = svc.predict(df_test)
# %%
svc_pred_price
# %%
#Return predicted price to data_set
df_test['price_range'] = svc_pred_price
# %%
df_test.sample(10)
# %%
