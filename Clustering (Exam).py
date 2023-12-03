#!/usr/bin/env python
# coding: utf-8

# ## Importing Relevant Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve


# ## Data Preprocessing

# In[2]:


df=pd.read_csv(r'C:\Users\ASUS\Downloads\cluster.csv')
df


# In[3]:


data=df.copy()


# In[4]:


data.describe(include='all')


# In[5]:


data.drop('CUST_ID', axis=1, inplace=True)


# In[6]:


data.isnull().sum()


# In[7]:


data['CREDIT_LIMIT']=data['CREDIT_LIMIT'].fillna(value=data['CREDIT_LIMIT'].mean())
data['MINIMUM_PAYMENTS']=data['MINIMUM_PAYMENTS'].fillna(value=data['MINIMUM_PAYMENTS'].mean())


# In[8]:


data.isnull().sum()


# ## Modeling with K means

# In[9]:


wcss = []

for i in range(1,7):
    kmeans=KMeans(i)
    kmeans.fit(data)
    wcss_iter= kmeans.inertia_
    wcss.append(wcss_iter)


# In[10]:


wcss


# In[11]:


number_clusters = range(1,7)

plt.plot(number_clusters, wcss)

plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within Cluster Sum of Squares')


# In[12]:


kmeans=KMeans(5)
kmeans.fit(data)


# In[13]:


identified_clusters = kmeans.fit_predict(data)


# In[14]:


identified_clusters


# In[15]:


data['CLUSTER']=identified_clusters


# In[16]:


data


# In[17]:


plt.scatter(data['BALANCE'], data['CREDIT_LIMIT'], c=data['CLUSTER'], cmap='rainbow')
plt.show()


# ## Modeling with AgglomerativeClustering

# In[18]:


df.head()


# In[19]:


data2=df.copy()


# In[20]:


data2.drop('CUST_ID', axis=1, inplace=True)


# In[21]:


data2['CREDIT_LIMIT']=data2['CREDIT_LIMIT'].fillna(value=data2['CREDIT_LIMIT'].mean())
data2['MINIMUM_PAYMENTS']=data2['MINIMUM_PAYMENTS'].fillna(value=data2['MINIMUM_PAYMENTS'].mean())


# In[22]:


data2.isnull().sum()


# In[23]:


from sklearn.preprocessing import StandardScaler


# In[24]:


sc=StandardScaler()


# In[25]:


data_scaled=sc.fit_transform(data2)


# In[26]:


data_scaled=pd.DataFrame(data_scaled, columns=data2.columns)
data_scaled


# In[27]:


import scipy.cluster.hierarchy as sch
plt.figure(figsize=(10, 5))  
plt.title("Dendrograms")  
dend = sch.dendrogram(sch.linkage(data_scaled, method='ward'))


# In[28]:


cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')


# In[29]:


y= cluster.fit_predict(data_scaled)


# In[30]:


y


# In[31]:


plt.figure(figsize=(10, 5))  
plt.scatter(data_scaled['BALANCE_FREQUENCY'], data_scaled['CASH_ADVANCE_FREQUENCY'], c=cluster.labels_, cmap='rainbow')
plt.show()


# In[32]:


data2['CLUSTER'] = y


# In[33]:


data2


# In[34]:


data2.columns


# In[36]:


data2.corr()['CLUSTER']


# In[37]:


avarege_corr = data2.corr()['CLUSTER'].mean()


# In[39]:


avarege_corr


# In[41]:


dropped_columns = []

for i in data2[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
       'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
       'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
       'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
       'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',
       'TENURE', 'CLUSTER']]:
    
    
    if abs(data2.corr()['CLUSTER'][i])<avarege_corr:
        dropped_columns.append(i)
    
data2.drop(dropped_columns, axis=1, inplace=True) 


# In[42]:


data2.head()


# In[43]:


data2.columns


# In[46]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data2[[
    
    'BALANCE_FREQUENCY', 
#     'PURCHASES', 
    'ONEOFF_PURCHASES',
    'INSTALLMENTS_PURCHASES', 
#     'PURCHASES_FREQUENCY',
    'ONEOFF_PURCHASES_FREQUENCY', 
    'PURCHASES_INSTALLMENTS_FREQUENCY',
    'PURCHASES_TRX'
                 
                  
]]

vif=pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
vif


# In[47]:


data2.drop(['PURCHASES', 'PURCHASES_FREQUENCY'], axis=1, inplace=True)


# In[48]:


data2


# In[49]:


data2.columns


# In[50]:


for i in data2[['BALANCE_FREQUENCY', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
       'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
       'PURCHASES_TRX']]:
    
    sns.boxplot(x=data2[i], data=data2)
    plt.show()


# In[51]:


q1=data.quantile(0.25)
q3=data.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
Upper=q3+1.5*IQR


# In[52]:


for i in data2[['BALANCE_FREQUENCY', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
       'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
       'PURCHASES_TRX']]:
    
    data2[i] = np.where(data2[i] > Upper[i], Upper[i],data2[i])
    data2[i] = np.where(data2[i] < Lower[i], Lower[i],data2[i])


# In[53]:


for i in data2[['BALANCE_FREQUENCY', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
       'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
       'PURCHASES_TRX']]:
    
    sns.boxplot(x=data2[i], data=data2)
    plt.show()


# In[54]:


data2=data2.reset_index(drop=True)


# ## Applying Supervised ML using Agglomerative Clustering

# In[55]:


X=data2.drop('CLUSTER',axis=1)
y=data2['CLUSTER']


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[57]:


def evaluate(model, X_test, y_test):
    
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:,1]
    
    roc_score_test = roc_auc_score(y_test, y_prob_test)
    gini_score_test= roc_score_test*2-1
    
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:,1]
    
    roc_score_train = roc_auc_score(y_train, y_prob_train)
    gini_score_train = roc_score_train*2-1
    
    confusion_matrix=metrics.confusion_matrix(y_test, y_pred_test)
    report=classification_report(y_test, y_pred_test)
    
    print('Model Performance')

    print('Gini Score for Test:', gini_score_test*100)
    
    print('Gini Score for Train:', gini_score_train*100)
    
    print('Confusion Matrix', confusion_matrix)
    
    print('Classification report', report)


# In[58]:


from xgboost import XGBClassifier


# In[59]:


xgboost=XGBClassifier()


# In[60]:


xgboost.fit(X_train, y_train)


# In[61]:


result=evaluate(xgboost, X_test, y_test)


# In[62]:


y_prob = xgboost.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# ## Univariate Analysis

# In[64]:


variables= []
train_Gini=[]
test_Gini=[]

for i in X_train.columns:
    X_train_single=X_train[[i]]
    X_test_single=X_test[[i]]
    
    xgboost.fit(X_train_single, y_train)
    y_prob_train_single=xgboost.predict_proba(X_train_single)[:, 1]
    
    
    roc_prob_train=roc_auc_score(y_train, y_prob_train_single)
    gini_prob_train=2*roc_prob_train-1
    
    
    xgboost.fit(X_test_single, y_test)
    y_prob_test_single=xgboost.predict_proba(X_test_single)[:, 1]
    
    
    roc_prob_test=roc_auc_score(y_test, y_prob_test_single)
    gini_prob_test=2*roc_prob_test-1
    
    
    variables.append(i)
    train_Gini.append(gini_prob_train)
    test_Gini.append(gini_prob_test)
    

df = pd.DataFrame({'Variable': variables, 'Train Gini': train_Gini, 'Test Gini': test_Gini})

df= df.sort_values(by='Test Gini', ascending=False)

df   

