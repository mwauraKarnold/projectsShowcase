#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Data Analysis

# In order to know which services the bank should improve on to improve customer satisfaction, this project deals with analysis of credit card customer data trying to understand what features would affect customer attrition and building a model using Random Forest, Support Vector Machine and Gradient Boosting to predit customer attrition.
# 
# Main steps that will be taken:
# 
# >Exploratory Data Analysis 
# 
# >Feature Engineering and Selection
# 
# >Model Training 

# In[29]:


# importing libraries  
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import plotly
import plotly.figure_factory as ff
from plotly.offline import plot,iplot,download_plotlyjs

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go
import cufflinks as cf

cf.set_config_file(sharing='public',theme='white',offline=True)

import plotly.io as pio
pio.renderers.default = 'colab'

from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import  RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report,recall_score,f1_score,precision_score,confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.metrics import plot_confusion_matrix


# ### Reading of the data 

# In[2]:


df = pd.read_csv('BankChurners.csv')
df.head()


# In[3]:


# removing two categories from documentation

df1 = df.iloc[:, :-2]


# ## Exploratory Data Analysis

# ### Checking the numerical and categorical features in the data 

# In[4]:


n_features= df1.select_dtypes(include=['float64','int64'])
n_features.sample()


# In[5]:


c_features = df1.select_dtypes(exclude=['float64','int64'])
c_features.sample()


# ### Categorical features visualisation and analysis

# In[6]:


# percentage of existing and attried customers

fig = go.Figure(data=[go.Pie(labels=df1['Attrition_Flag'],title='Percentage of Existing and Attrited Customers', pull=[0.2, 0])])

fig.show()


# This is not a great ratio. It's unbalanced.
# This could cause false negative/positive resilts because the attried data is undersampled.
# It has to be dealt with before building the model.

# In[7]:


# education level of existing and attried customers

# for existing
plot_edu_e = df1[df1['Attrition_Flag']=='Existing Customer']
plot_edu_e = plot_edu_e['Education_Level'].value_counts().reset_index()
plot_edu_e.columns =['Education_level','Count']
plot_edu_e = plot_edu_e.sort_values('Count')

fig0 = px.bar(
    data_frame=plot_edu_e,
    x="Count",
    y="Education_level",
    color="Education_level",               
    opacity=0.9,                  
    orientation="h",              
    barmode='relative',
    width=700,                   
    height=360,
    title='Education level of existing customers'
    )

# for attried
plot_edu_a = df1[df1['Attrition_Flag']=='Attrited Customer']
plot_edu_a = plot_edu_a['Education_Level'].value_counts().reset_index()
plot_edu_a.columns =['Education_level','Count']
plot_edu_a = plot_edu_a.sort_values('Count')

fig1 = px.bar(
    data_frame=plot_edu_a,
    x="Count",
    y="Education_level",
    color="Education_level",              
    opacity=0.9,                  
    orientation="h",              
    barmode='relative',
    width=700,                 
    height=360,
    title='Education level of attried customers'
    )

fig0.show()
fig1.show()


# In[8]:


# gender and marital status of existing and attried customers

#for existing
plot_gen_e = df1[df1['Attrition_Flag']=='Existing Customer']
plot_gen_e = plot_gen_e[['Marital_Status','Gender','Attrition_Flag']].groupby(['Marital_Status','Gender']).count().reset_index()
plot_gen_e.columns =['Marital_Status','Gender','Count']
plot_gen_e = plot_gen_e.sort_values('Count')

fig0 = px.bar(
    data_frame=plot_gen_e,
    x="Marital_Status",
    y="Count",
    color="Gender",               # differentiate color of marks
    opacity=0.9,                  # set opacity of markers (from 0 to 1)
    orientation="v",              # 'v','h': orientation of the marks
    barmode='group',
    width=700,                   # figure width in pixels
    height=360,
    title='Marital Status and Gender of existing customers'
    )

# for attried 
plot_gen_a = df1[df1['Attrition_Flag']=='Attrited Customer']
plot_gen_a = plot_gen_a[['Marital_Status','Gender','Attrition_Flag']].groupby(['Marital_Status','Gender']).count().reset_index()
plot_gen_a.columns =['Marital_Status','Gender','Count']
plot_gen_a = plot_gen_a.sort_values('Count')

fig1 = px.bar(
    data_frame=plot_gen_a,
    x="Marital_Status",
    y="Count",
    color="Gender",              
    opacity=0.9,                  
    orientation="v",              
    barmode='group',
    width=700,                   
    height=360,
    title='Marital Status and Gender of attried customers'
    )

fig0.show()
fig1.show()


# In[9]:


# income level and card category of existing and attried customers

inc_a = df1.loc[df1['Attrition_Flag']=='Attrited Customer','Income_Category']
inc_e = df1.loc[df1['Attrition_Flag']=='Existing Customer','Income_Category']

fig = make_subplots(
    rows=1, cols=2,subplot_titles=('Existing Customers','Attrited Customers','Residuals'),
    vertical_spacing=0.09,
    specs=[[{"type": "pie","rowspan": 1},{"type": "pie"}]]
)


fig.add_trace(
    go.Pie(values=inc_e.value_counts().values,
        labels=['Less than $40k ','$40k - $60k','$80k - $120k','$60k - $80k','Unknown','$120k +']
        ),
    row=1, col=1
)

fig.add_trace(
    go.Pie(values=inc_a.value_counts().values,
        labels=['Less than $40k ','$40k - $60k','$80k - $120k','$60k - $80k','Unknown','$120k +']),
    row=1, col=2
)

fig.update_layout(
    height=400,
    showlegend=True,
    title_text="<b>Income_Category<b>",
)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# In[10]:


card_a = df1.loc[df1['Attrition_Flag']=='Attrited Customer','Card_Category']
card_e = df1.loc[df1['Attrition_Flag']=='Existing Customer','Card_Category']

fig = make_subplots(
    rows=1, cols=2,subplot_titles=('Existing Customers','Attrited Customers','Residuals'),
    vertical_spacing=0.09,
    specs=[[{"type": "pie","rowspan": 1},{"type": "pie"}]]
)


fig.add_trace(
    go.Pie(values=card_e.value_counts().values,
        labels=['Blue','Silver','Gold','Platinum']
        ),
    row=1, col=1
)

fig.add_trace(
    go.Pie(values=card_a.value_counts().values,
        labels=['Blue','Silver','Gold','Platinum']),
    row=1, col=2
)

fig.update_layout(
    height=600,
    showlegend=True,
    title_text="<b>Card_Category<b>",
)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# ### Numerical features visualisation and analysis

# In[11]:


# age

plot_age = df1['Customer_Age'].value_counts().reset_index()
plot_age.columns =['Customer_Age','Count']
plot_age = plot_age.sort_values('Count')

fig = px.bar(
    data_frame=plot_age,
    x="Customer_Age",
    y="Count",
    color="Customer_Age",               
    opacity=0.9,                  
    orientation="v",             
    barmode='relative',
    title='Age of customers'
)

fig.show()


# In[12]:


# number of products held by customer

fig = px.histogram(df1, x="Total_Relationship_Count", color="Attrition_Flag",title='Number of products held by customer')
fig.show()


# In[13]:


# dependents 
plot_dep = df1['Dependent_count'].value_counts().reset_index()
plot_dep.columns =['Dependent_count','Count']
plot_dep = plot_dep.sort_values('Count')

fig = px.bar(
    data_frame=plot_dep,
    x="Dependent_count",
    y="Count",
    color="Dependent_count",               
    opacity=0.9,                  
    orientation="v",             
    barmode='relative',
    title='Number of dependents'
)

fig.show()


# In[14]:


# number of months with no transactions

fig = px.histogram(df1, x="Months_Inactive_12_mon", color="Attrition_Flag",title='Number of months with no transactions in the last year')
fig.show()


# In[15]:


# change in transaction

fig = px.histogram(df1, x="Total_Ct_Chng_Q4_Q1", color="Attrition_Flag",title='Change in transaction number over the last year (Q4 over Q1)')
fig.show()


# In[16]:


# number of transactions 

fig = px.histogram(df1, x="Total_Trans_Ct", color="Attrition_Flag",title='Number of transactions made in the last year')
fig.update()
fig.show()


# ## Feature Engineering and Selection

# ### Correlation

# Because we are working with a dataset with mixed features(categorical and numerical), we just can't use standard correlation function. We have to split the features and measure categorical data correlation with Pearson and numerical data correlation with Cramer's V function.
# 
# 

# In[17]:


# correlation of categorical data 

label = preprocessing.LabelEncoder()
df_c_encoded = pd.DataFrame() 

for i in c_features.columns :
  df_c_encoded[i]=label.fit_transform(c_features[i])


def cramers_V(var1,var2) :
  crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None))
  stat = chi2_contingency(crosstab)[0] 
  obs = np.sum(crosstab) 
  mini = min(crosstab.shape)-1 
  return (stat/(obs*mini))

rows= []

for var1 in df_c_encoded:
  col = []
  for var2 in df_c_encoded :
    cramers =cramers_V(df_c_encoded[var1], df_c_encoded[var2]) 
    col.append(round(cramers,2))
  rows.append(col)
  
cramers_results = np.array(rows)
cramerv_matrix = pd.DataFrame(cramers_results, columns = df_c_encoded.columns, index =df_c_encoded.columns)
mask = np.triu(np.ones_like(cramerv_matrix, dtype=bool))
cat_heatmap = sns.heatmap(cramerv_matrix, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
cat_heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# From the heatmap we can tell that categorical columns are not correlated with customer churn by themselves.
# Therefore building an accurate model without considering the numerical values is impossible.

# In[18]:


# correlation oh numerical data

n_features['Attrition_Flag']=df1.loc[:,'Attrition_Flag']
oh=pd.get_dummies(n_features['Attrition_Flag'])
n_features=n_features.drop(['Attrition_Flag'],axis=1)
n_features=n_features.drop(['CLIENTNUM'],axis=1)
n_features=n_features.join(oh)
n_features.head()

num_corr=n_features.corr()
plt.figure(figsize=(16, 6))
mask = np.triu(np.ones_like(num_corr, dtype=bool))
num_heatmap = sns.heatmap(num_corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
num_heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# Better correlation measurements can be seen.
# The correlation coefficient of attrited and existing customer to all feature columns are identical in numbers, with mirroring signs.
# Lets take a closer look.

# In[19]:


fig, ax=plt.subplots(ncols=2,figsize=(15, 10))

heatmap = sns.heatmap(num_corr[['Existing Customer']].sort_values(by='Existing Customer', ascending=False), ax=ax[0],vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Existing Customers', fontdict={'fontsize':18}, pad=25);
heatmap = sns.heatmap(num_corr[['Attrited Customer']].sort_values(by='Attrited Customer', ascending=False), ax=ax[1],vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Attrited Customers', fontdict={'fontsize':18}, pad=25);

fig.tight_layout(pad=5)


# We can therefore conclude that,
# Credit Limit, Average open to buy, Age and Dependent count
# are not correlated to customer churn
# 

# In[20]:


# discard the features mentioned above to build an accurate model.

df_model=df1
df_model=df_model.drop(['CLIENTNUM','Credit_Limit','Customer_Age','Avg_Open_To_Buy','Months_on_book','Dependent_count'],axis=1)
df_model.head()


# In[21]:


# convert the categorical features into binary with one hot encoding

df_model['Attrition_Flag'] = df_model['Attrition_Flag'].map({'Existing Customer': 1, 'Attrited Customer': 0})
df_oh=pd.get_dummies(df_model)
df_oh['Attrition_Flag'] = df_oh['Attrition_Flag'].map({1: 'Existing Customer', 0: 'Attrited Customer'})
list(df_oh.columns)


# ## Model Training 

# ### Balancing the training dataset

# To balance the dataset we will utilize SMOTE.
# We will split the training and test data.
# SMOTE will onlt be applied to the training dataset.

# In[22]:


# balancing the training dataset

X = df_oh.loc[:, df_oh.columns != 'Attrition_Flag']
y = df_oh['Attrition_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[23]:


sm = SMOTE(sampling_strategy='minority', k_neighbors=20, random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


# ### Random forest classifier 

# In[24]:


rf_clf = RandomForestClassifier()
rf_clf.fit(X_train_res, y_train_res)


# ### Support Vector Machine

# In[25]:


svm_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm_clf.fit(X_train_res, y_train_res)


# ### Gradient boosting 

# In[26]:


gb_clf=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=42)
gb_clf.fit(X_train_res, y_train_res)


# ### First prediction and evaluation

# In[27]:


# first prediction

y_rf=rf_clf.predict(X_test)
y_svm=svm_clf.predict(X_test)
y_gb=gb_clf.predict(X_test)


# In[30]:


# first evaluation
fig,ax=plt.subplots(ncols=3, figsize=(20,6))
plot_confusion_matrix(rf_clf, X_test, y_test, ax=ax[0])
ax[0].title.set_text('Random Forest')
plot_confusion_matrix(svm_clf, X_test, y_test, ax=ax[1])
ax[1].title.set_text('Support Vector Machine')
plot_confusion_matrix(gb_clf, X_test, y_test, ax=ax[2])
ax[2].title.set_text('Gradient Boosting')
fig.tight_layout(pad=5)


# In[31]:


print('Random Forest Classifier')
print(classification_report(y_test, y_rf))
print('------------------------')
print('Support Vector Machine')
print(classification_report(y_test, y_svm))
print('------------------------')
print('Gradient Boosting')
print(classification_report(y_test, y_gb))


# From both the confusion matrix and classification report that random forest and gradient boosting are the best with recall scores above 85% 

# ### Tuning hyperparameters

# Let's try and increase the model accuracy of the random forest and gradient boosting

# In[32]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[ ]:


#rf_random = RandomizedSearchCV(estimator = rf_clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
#rf_random.fit(X_train_res, y_train_res)
#print(rf_random.best_params_)


# In[33]:


rf_clf_opt= RandomForestClassifier(n_estimators=750, min_samples_split=2, min_samples_leaf=1, 
                            max_features='auto', max_depth=50, bootstrap=False)
rf_clf_opt.fit(X_train_res,y_train_res)
y_rf_opt=rf_clf_opt.predict(X_test)
print('Random Forest Classifier (Optimized)')
print(classification_report(y_test, y_rf_opt))
_rf_opt=plot_confusion_matrix(rf_clf_opt, X_test, y_test)


# In[ ]:


#param_test1 = {'n_estimators':range(20,81,10)}
#gsearch1 = GridSearchCV(
#estimator = GradientBoostingClassifier(learning_rate=1.0, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
#param_grid = param_test1, scoring='roc_auc',n_jobs=4, cv=5)
#gsearch1.fit(X_train_res,y_train_res)
#print(gsearch1.best_params_)


# In[34]:


gb_clf_opt=GradientBoostingClassifier(n_estimators=80,learning_rate=1.0, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10)
gb_clf_opt.fit(X_train_res,y_train_res)
y_gb_opt=gb_clf_opt.predict(X_test)
print('Gradient Boosting (Optimized)')
print(classification_report(y_test, y_gb_opt))
print(recall_score(y_test,y_gb_opt,pos_label="Attrited Customer"))
_gbopt=plot_confusion_matrix(gb_clf_opt, X_test, y_test)


# ## Conclusion

# The model should be considered adequate as there was no significant change to the accuracy and has 95% accuracy and a 84% recall score.
# 
# However the performance is unkown in the real world since we used SMOTE to adjust the data set.
