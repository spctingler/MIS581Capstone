#Python Results and Code
#Read and load dataset

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

filename='KDDDATA.csv'
data=pd.read_csv(filename, header=None)
data.dropna(inplace=True, axis=1)

data.columns= [
    'duration', 
    'protocol', 
    'service', 
    'flag', 
    'src_bytes',
    'dst_bytes',
    'land', 
    'wrong_fragment', 
    'urgent', 
    'hot',
    'num_failed_logins', 
    'logged_in', 
    'num_compromised', 
    'root_shell', 
    'su_attempted', 
    'num_root', 
    'num_file_creations', 
    'num_shells', 
    'num_access_files', 
    'num_outbound_cmds', 
    'is_host_login', 
    'is_guest_login', 
    'count',
    'srv_count', 
    'serror_rate', 
    'srv_serror_rate', 
    'rerror_rate', 
    'srv_rerror_rate', 
    'same_srv_rate', 
    'diff_srv_rate', 
    'srv_diff_host_rate', 
    'dst_host_count', 
    'dst_host_srv_count', 
    'dst_host_same_srv_rate', 
    'dst_host_diff_srv_rate', 
    'st_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate', 
    'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 
    'dst_host_srv_rerror_rate',
    'outcome'
]

data

 


#Uploads first 5 rows of the dataset
data.head()

 

#Uploads specific information about variable outcome
data['outcome'].unique

 

#Data exploration
data['outcome'].value_counts()

 

#Change 'outcome' categorical values to binary
cleanup = {"outcome":{"normal":1, "anomaly":0}}
data = data.replace(cleanup)
data

#Data Cleaning to show necessary variables 
data1= data[['src_bytes', 'dst_bytes', 'outcome']]
data1

#Descriptive statistics for data1
data1.describe()

#Boxplot- Outcome Value Counts
sns.countplot(x='outcome', data=data, palette='hls')
plt.show()
plt.savefig('count_plot')

#Boxplot to show Source bytes by type of outcome (normal vs anomaly)

data1.boxplot(column ='src_bytes', by = 'outcome')
plt.ylim(0, 7500);
 
#Boxplot to show Destination bytes by type of outcome (normal vs anomaly)
data1.boxplot(column ='dst_bytes', by = 'outcome')
plt.ylim(0, 7500);

#Logistic Regression Analysis
import statsmodels.api as sm
logit_model=sm.Logit(outcome,dataX)
result=logit_model.fit()
print(result.summary())


#Logistic Regression Model Fitting
from sklearn.model_selection import train_test_split
dataX_train, dataX_test, outcome_train, outcome_test=train_test_split (dataX, outcome, test_size=0.20, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
logreg= LogisticRegression()
logreg.fit(dataX_train, outcome_train)

#Prediction for test set results and calculating accuracy
outcome_pred= logreg.predict(dataX_test)
print('Accuracy of logistic regression on test set: {:.2f}'.format(logreg.score(dataX_test, outcome_test)))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(outcome_test, outcome_pred)
print(confusion_matrix)

#Precision, recall, f-score
from sklearn.metrics import classification_report
print(classification_report(outcome_test, outcome_pred))

 

