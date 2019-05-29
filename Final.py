# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:09:25 2019

@author: zehra
"""

###############################################################################
#Importing Libraries
###############################################################################

import os
os.chdir('D:\Machine Learning\Individual Assignment')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeRegressor # Regression trees
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects
from sklearn.tree import DecisionTreeClassifier # Classification trees
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import itertools

###############################################################################
#Importing Dataset
###############################################################################

file = 'got.xlsx'
got = pd.read_excel(file)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

###############################################################################
# Fundamental Dataset Exploration
###############################################################################

# Column names

got.columns

# Dimensions of the DataFrame

got.shape

# Information about each variable

got.info()

# Descriptive statistics

desc = got.describe().round(2)

print(desc)

# Correcting typo in dataset

got['age'][got['age'] < 0] = 0

got['dateOfBirth'][got['dateOfBirth'] == 298299] = 298
got['dateOfBirth'][got['dateOfBirth'] == 278279] = 278

# Creating subset for exploration

alive = got[got.isAlive == 1]

alive.shape[0]

# 1,451 characters alive #

dead  = got[got.isAlive == 0]

dead.shape[0]

# 495 characters dead #

# Pie chart for Alive and Dead characters #

labels = 'Alive', 'Dead'
sizes = [1451, 495]
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.savefig('Pie_Dead_Alive.png')
plt.show()

# 76% characters are alive and 25% are dead #

# Pie chart for Male and Female characters #

labels = 'Male', 'Female'
sizes = [1176, 734]
colors = ['lightcoral', 'lightskyblue']
explode = (0.1, 0)  # explode 1st slice
 
# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.savefig('Pie_Male_Female.png')
plt.show()

# 62% characters are male and 38% are female #

# Barplot for number of characters per book #

objects = ('Book1', 'Book2', 'Book3', 'Book4', 'Book5')
y_pos = np.arange(len(objects))
performance = [378,717,914,772,760]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)
plt.ylabel('Characters')
plt.title('Number of characters per book')
plt.savefig('bar_no_of_books.png')
plt.show()

# Book 3 had the most characters #

got['isMarried'].value_counts()

# Majority characters are married #

got['isNoble'].value_counts()

# 53% characters are noble #

###############################################################################
#Missing Values
###############################################################################

# To see count of missing values per column #
print(
      got
      .isnull()
      .sum().sort_values(ascending = False)
      )

# To see % of missing values per column #

print(
      ((got[:].isnull().sum())
      /
      got
      .shape[0])
      .round(2).sort_values(ascending = False)
      )

#####################
#Drop columns
#####################
      
col =['isAliveSpouse','isAliveFather','isAliveMother','isAliveHeir']
got.drop(col, inplace=True, axis=1)

# 99% missing values in dropped columns #

########################
#Flagging missing values
########################

for col in got:
    
    """Create columns that are 0s if a value was not missing and 1 if
    a value is missing."""
    
    if got[col].isnull().astype(int).sum() > 0:
        
        got['m_'+col] = got[col].isnull().astype(int)

#####################
#Drop columns
#####################    
        
column = ['m_mother', 'm_father', 'm_heir', 'm_spouse', 'm_culture', 
          'm_title']

got.drop(column, inplace=True, axis=1)

''' As their binary columns with unknowns as 1 and rest as 0 
are created below '''

########################
#Imputing Missing Values
########################

# Age #

age_median = got['age'].median()

got['age'] = got['age'].fillna(age_median).round(2)

# Date Of Birth #

dob_median = got['dateOfBirth'].median()

got['dateOfBirth'] = got['dateOfBirth'].fillna(dob_median).round(2)

# House #

fill = 'Unknown'

got['house'] = got['house'].fillna(fill)

# Title #

fill = 'not noble'

got['title'] = got['title'].fillna(fill)

# Missing titles only for not noble characters #

# Culture #

fill = 'Unknown'

got['culture'] = got['culture'].fillna(fill)

# Spouse #

fill = 'Unknown'

got['spouse'] = got['spouse'].fillna(fill)

# Heir #

fill = 'Unknown'

got['heir'] = got['heir'].fillna(fill)

# Mother #

fill = 'Unknown'

got['mother'] = got['mother'].fillna(fill)

# Father #

fill = 'Mother Unknown too'

got['father'] = got['father'].fillna(fill)

# Missing fathers have missing mothers for all characters #

###############################################################################
#Creating binary columns for title , culture , spouse , heir , mother , father
###############################################################################

def func(x):
    
    if x == 'not noble' :
        return 1
    else:
        return 0
    
got['new_title'] = got['title'].map(func)

# Missing titles grouped as 1 #

def func(x):
    
    if x == 'Unknown' :
        return 1
    else:
        return 0
    
got['new_culture'] = got['culture'].map(func)

# Missing cultures grouped as 1 #

""" Missing cultures for least important characters, 76% of them are alive 
   and 95% of them have zero dead realations and these characters have only 
   appeared in book 3 and 4. """

def func(x):
    
    if x == 'Unknown' :
        return 1
    else:
        return 0
    
got['new_spouse'] = got['spouse'].map(func)

# Missing Spouses grouped as 1 #

""" 78% characters with missing spouses have alive spouses , maybe there 
    spouses were not important to be known. """

def func(x):
    
    if x == 'Unknown' :
        return 1
    else:
        return 0
    
got['new_heir'] = got['heir'].map(func)

# Missing Heirs grouped as 1 #

def func(x):
    
    if x == 'Unknown' :
        return 1
    else:
        return 0
    
got['new_mother'] = got['mother'].map(func)

# Missing Mothers grouped as 1 #

def func(x):
    
    if x == 'Mother Unknown too' :
        return 1
    else:
        return 0
    
got['new_father'] = got['father'].map(func)

# Missing Fathers grouped as 1 #

###############################################################################
#Engineering Data Features
###############################################################################

###########################
#Creating New Data Features 
###########################

## num_books ##

columns = ['book1_A_Game_Of_Thrones','book2_A_Clash_Of_Kings',
           'book3_A_Storm_Of_Swords','book4_A_Feast_For_Crows',
           'book5_A_Dance_with_Dragons']

got['num_books']= got[columns].sum(axis=1)

# New column created to look , in how many books a character appeared #

## alive_status ##

counts = got[['house','isAlive']].groupby(['house']).agg(['sum']) 

counts.columns=counts.columns.droplevel() 
counts['alive_status'] = counts['sum']
counts=counts.drop(columns=['sum'])

got = pd.merge(got,counts,on='house')

# Alive status column created to see alive members per house #

## house_size ##

countss = got[['house','name']].groupby(['house']).agg(['count']) 

countss.columns=countss.columns.droplevel() 
countss['house_size'] = countss['count']
countss=countss.drop(columns=['count'])

got = pd.merge(got,countss,on='house')

# House size column created to see members per house #

## per_alive ##

got['per_alive'] = got['alive_status']/got['house_size']

# Per alive column created to see proportion of alive members per house #

## new_numDeadRelations ##

def func(x):
    
    if x == 0 :
        return 1
    else:
        return 0
    
got['new_numDeadRelations'] = got['numDeadRelations'].map(func)

# Characters with zero dead relations grouped as 1 #

## pop_bighouse ##

got['pop_bighouse'] = got['house_size'] * got['popularity']

# Column created to see popular characters from big houses #

## pop_male ##

got['pop_male'] = got['popularity'] * got['male']

# Column created to see popular male charaacters #

###############################################################################
# Correlation Analysis
###############################################################################

# Creating correlation matrix to observe correlations #

df_corr = got.corr().round(2)

# Looking at correlations w.r.t isAlive column #

df_corr.loc['isAlive'].sort_values(ascending = False)

#####################
# Correlation Heatmap
#####################

# Using palplot to view a color scheme #

sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize = (15, 15))

df_corr2 = df_corr.iloc[1:19, 1:19]

sns.heatmap(
            df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5
            )

plt.savefig('got Correlation Heatmap.png')
plt.show()

###############################################################################
#Data Visualization
###############################################################################
 
# Examining relationship between survival and having zero dead relatives #
 
data = got.groupby(["new_numDeadRelations", 
                    "isAlive"]).count()["S.No"].unstack().copy(deep = True)
    
p = data.div(data.sum(axis = 1), axis = 0).plot.barh(stacked = True, 
            rot = 0, width = .5)

_ = p.set_xlim([0, 1]), p.set(yticklabels = ["No", "Yes"],
              xticklabels = "", xlabel = "Proportion of Dead vs. Alive", 
              ylabel = "Zero Dead Relations"), p.legend(["Dead", "Alive"])

plt.savefig('Dead_Relatives_Stack.png')

# Characters with zero dead relatives are more likely to stay alive #

# Examining relationship between survival and gender #
 
data = got.groupby(["male", 
                    "isAlive"]).count()["S.No"].unstack().copy(deep = True)
    
p = data.div(data.sum(axis = 1), axis = 0).plot.barh(stacked = True, 
            rot = 0, width = .5)

_ = p.set_xlim([0, 1]), p.set(yticklabels = ["No", "Yes"],
              xticklabels = "", xlabel = "Proportion of Dead vs. Alive", 
              ylabel = "male"), p.legend(["Dead", "Alive"])

plt.savefig('Male_Stack.png')

# Male characters are more likely to die #

# Examine relationship between survival and chatacter appearing in more books #

data = got.groupby(["num_books",
                    "isAlive"]).count()["S.No"].unstack().copy(deep = True)
    
p = data.div(data.sum(axis = 1), axis = 0).plot.barh(stacked = True, 
            rot = 0, figsize = (15, 8), width = .5)

_ = p.set(xticklabels = "", xlim = [0, 1], ylabel = "No. of Books", 
          xlabel = "Proportion of Dead vs. Alive"), p.legend(["Dead", "Alive"],
                                                 loc = "upper right", ncol = 2, 
                                                 borderpad = -.15)
plt.savefig('num_books_Stack.png')

''' Characters appearing in zero books are more likely to die, and character
    appearing in two books are more likely to survive '''

# Examining relationship between survival and house size #

data = got.groupby(["house_size",
                    "isAlive"]).count()["S.No"].unstack().copy(deep = True)
    
p = data.div(data.sum(axis = 1), axis = 0).plot.barh(stacked = True, 
            rot = 0, figsize = (15, 8), width = .5)

_ = p.set(xticklabels = "", xlim = [0, 1], ylabel = "house size", 
          xlabel = "Proportion of Dead vs. Alive"), p.legend(["Dead", "Alive"],
                                                 loc = "upper right", ncol = 2, 
                                                 borderpad = -.15)
plt.savefig('House_Size_Stack.png')

""" Members from large houses are more likely to die as larger houses are 
    more likely involved in battles """

# Examine relationship between survival and appearance of character in book 1 #
    
data = got.groupby(["book1_A_Game_Of_Thrones",
                    "isAlive"]).count()["S.No"].unstack().copy(deep = True)
    
p = data.div(data.sum(axis = 1), axis = 0).plot.barh(stacked = True, 
            rot = 0, figsize = (15, 8), width = .5)

_ = p.set(xticklabels = "", xlim = [0, 1], ylabel = "house size", 
          xlabel = "Proportion of Dead vs. Alive"), p.legend(["Dead", "Alive"],
                                                 loc = "upper right", ncol = 2, 
                                                 borderpad = -.15)
plt.savefig('Book1_Stack.png')

''' Characters appearing in book1 are more likely to die as they are getting
    older '''
    
# Examine relationship between survival and appearance of character in book 4 #
    
data = got.groupby(["book4_A_Feast_For_Crows",
                    "isAlive"]).count()["S.No"].unstack().copy(deep = True)
    
p = data.div(data.sum(axis = 1), axis = 0).plot.barh(stacked = True, 
            rot = 0, figsize = (15, 8), width = .5)

_ = p.set(xticklabels = "", xlim = [0, 1], ylabel = "house size", 
          xlabel = "Proportion of Dead vs. Alive"), p.legend(["Dead", "Alive"],
                                                 loc = "upper right", ncol = 2, 
                                                 borderpad = -.15)
plt.savefig('Book4_Stack.png')

''' Characters appearing in book 4 are more likely to survive as they are
    younger and still growing old '''

###############################################################################
#Classification Models
###############################################################################   

""" Analysis done below are through Classification Modeles listed below:
    
    1) Logistic Regression with mean AUC value = 0.846
    2) KNN with mean AUC value = 0.802
    3) Regression Tree with mean AUC value = 0.796
    4) Classification Tree with mean AUC value = 0.793
    5) Random Forest with mean AUC value = 0.867
    6) Gradient Boosted Machines with mean AUC value = 0.864
    
    Random Forest remained our best model with highest mean AUC value """

###############################################################################
#Logistic Regression
###############################################################################

# Base model based upon significant correlations #

logistic = smf.logit(formula = """isAlive ~ male 
                                          + dateOfBirth
                                          + new_mother
                                          + new_father
                                          + new_heir 
                                          + per_alive 
                                          + book1_A_Game_Of_Thrones
                                          + book4_A_Feast_For_Crows
                                          + new_numDeadRelations
                                          + popularity
                                          + pop_bighouse 
                                          + pop_male """, data=got)

results_logistic_full = logistic.fit()

results_logistic_full.summary()

# Significant model #

logistic_full0 = smf.logit(formula = """isAlive ~ male 
                                                + per_alive
                                                + dateOfBirth
                                                + pop_bighouse
                                                + book1_A_Game_Of_Thrones
                                                + book4_A_Feast_For_Crows
                                               
                                               """, data=got) 

results_logistic_full = logistic_full0.fit()

results_logistic_full.summary()

''' From the above model, pop_big_house (60% correlated with popularity) 
    and book 1 (32% correlated with popularity) is intentionally taken out
    as compared to it book 4 characters have higher proportion of alive 
    characters.
    
    With removal of these two, adding back popularity and new_numDeadRelations
    was now significant which made sense, as having zero dead relatives
    increases survival of the character, which was seen through visualization
    above.
    
    Moreover, above features generated lower Cross Validation scores, across
    all classification model.The results for models are not displayed to avoid 
    complexity '''
    
###############################################################################    
""" Significant MODIFIED model is presented below """
############################################################################### 
                         
# Significant MODIFIED model #

logistic_full = smf.logit(formula = """isAlive ~ male                 
                                               + popularity  
                                               + dateOfBirth
                                               + per_alive             
                                               + new_numDeadRelations  
                                               + book4_A_Feast_For_Crows""",
                                               
                                               data=got)
results_logistic_full = logistic_full.fit()

results_logistic_full.summary()

""" All analysis ahead are carried out with significant MODIFIED model """

###############################################################################
# Train/Test Split with Significant MODIFIED model #
###############################################################################

# Preparing train/test split with the optimal logistic model #

got_data = got.loc[: , ['male',
                        'popularity',
                        'per_alive',
                        'dateOfBirth',
                        'book4_A_Feast_For_Crows',
                        'new_numDeadRelations']]

got_target =  got.loc[: , 'isAlive']

X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.10,
            random_state = 508,
            stratify = got_target)

################################################
# Hyperparameter Tuning with Logistic Regression
################################################

grid={"C":np.logspace(-3, 3, 7), "penalty":["l1","l2"]}# l1 lasso # l2 ridge
logreg = LogisticRegression()
logreg_cv = RandomizedSearchCV(logreg, grid, cv = 3, random_state = 508)
logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

# Building Logistic Regression with optimal Parameter #

logreg=LogisticRegression(C = 1, penalty = "l1", random_state = 508)
logreg_fit = logreg.fit(X_train,y_train)

# Predictions

logreg_pred = logreg_fit.predict(X_test)

#Train/test scores

print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))

train = logreg_fit.score(X_train, y_train).round(4)
test  = logreg_fit.score(X_test, y_test).round(4)

print (train - test)

""" Cross Validation Score """

cv_lr_3 = cross_val_score(logreg_fit, 
                          got_data, 
                          got_target, 
                          cv = 3, 
                          scoring = 'roc_auc')

print(pd.np.mean(cv_lr_3))

Logistic_regression_CV_Score = pd.np.mean(cv_lr_3).round(3)

###############################################################################
#KNN
###############################################################################

# Neighbor optimization code with a small adjustment for classification #

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    
    # build the model
    
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train.values.ravel())
    
    # record training set accuracy
    
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    
    test_accuracy.append(clf.score(X_test, y_test))


fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

plt.show()

# Looking for the highest test accuracy

print(test_accuracy)

# Printing highest test accuracy

print(test_accuracy.index(max(test_accuracy)) + 1)

# It looks like 5 neighbors is the most accurate

knn_clf = KNeighborsClassifier(n_neighbors = 5)

# Fitting the model based on the training data

knn_clf_fit = knn_clf.fit(X_train, y_train)

# Let's compare the testing score to the training score #

print('Training Score', knn_clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))

train = knn_clf_fit.score(X_train, y_train).round(4)
test  = knn_clf_fit.score(X_test, y_test).round(4)

print (train - test)

""" Cross Validation Score """

cv_knn_3 = cross_val_score(knn_clf_fit, 
                           got_data, 
                           got_target,
                           cv = 3, 
                           scoring = 'roc_auc')

print(pd.np.mean(cv_knn_3))

KNN_CV_Score = pd.np.mean(cv_knn_3).round(3)

###############################################################################
# Regression Trees
###############################################################################

###############################################
# Hyperparameter Tuning with RandomizedSearchCV
###############################################

# Creating a hyperparameter grid

depth_space = pd.np.arange(1, 10)
leaf_space = pd.np.arange(1, 50)
split_space = pd.np.arange(0.10,1.00)

param_grid = {'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space,
             'min_samples_split': split_space}

# Building the model object

c_tree_2_hp = DecisionTreeRegressor(random_state = 508)

# Creating a RandomizedSearchCV object

c_tree_2_hp_cv = RandomizedSearchCV(c_tree_2_hp, param_grid, cv = 3, 
                                random_state = 508)
                             
# Fit it to the training data

c_tree_2_hp_cv.fit(X_train, y_train)

# Print the optimal parameters and best score

print("Tuned Regression Tree Parameter:", c_tree_2_hp_cv.best_params_)
print("Tuned Regression Tree Accuracy:", c_tree_2_hp_cv.best_score_.round(4))

# Building a Regression tree model object with optimal hyperparameters #

tree_leaf = DecisionTreeRegressor(criterion = 'mse',
                                     min_samples_leaf = 36,
                                     max_depth = 7,
                                     random_state = 508)

tree_leaf_fit = tree_leaf.fit(X_train, y_train)

print('Training Score', tree_leaf.score(X_train, y_train).round(4))
print('Testing Score:', tree_leaf.score(X_test, y_test).round(4))

train = tree_leaf.score(X_train, y_train).round(4)
test  =  tree_leaf.score(X_test, y_test).round(4)

print(train - test)

# Visualizing the tree

dot_data = StringIO()

export_graphviz(decision_tree = tree_leaf,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = got_data.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png(),
      height = 500,
      width = 800)

""" Cross Validation Score """

cv_RegTree_scores = cross_val_score(tree_leaf_fit,
                                    got_data,
                                    got_target,
                                    cv = 3, 
                                    scoring = 'roc_auc')

print(pd.np.mean(cv_RegTree_scores))

Reg_tree_Cv_Score = pd.np.mean(cv_RegTree_scores).round(3)

###############################################################################
# Classification Trees
###############################################################################

#########################################
# Hyperparameter Tuning with GridSearchCV
#########################################

# Creating a hyperparameter grid

depth_space = pd.np.arange(1, 10)
leaf_space = pd.np.arange(1, 50)
split_space = pd.np.arange(0.10,1.00)

param_grid = {'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space,
             'min_samples_split': split_space}

# Building the model object

c_tree_2_hp = DecisionTreeClassifier(random_state = 508)

# Creating a GridSearchCV object
c_tree_2_hp_cv = GridSearchCV(c_tree_2_hp, param_grid, cv = 3)

# Fit it to the training data
c_tree_2_hp_cv.fit(X_train, y_train)

# Print the optimal parameters and best score

print("Tuned Classification Tree Parameter:", c_tree_2_hp_cv.best_params_)
print("Tuned Classification Tree Accuracy:", c_tree_2_hp_cv.best_score_.round(4))

# Building a Classification tree model object with optimal hyperparameters #

c_tree_optimal = DecisionTreeClassifier(criterion = 'gini',
                                        random_state = 508,
                                        max_depth = 5,
                                        min_samples_leaf = 34)

c_tree_optimal_fit = c_tree_optimal.fit(X_train, y_train)

dot_data = StringIO()

export_graphviz(decision_tree = c_tree_optimal_fit,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = X_train.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png(),
      height = 500,
      width = 800)

""" Cross Validation Score """

cv_ClassTree_scores = cross_val_score(c_tree_optimal_fit,
                                    got_data,
                                    got_target,
                                    cv = 3, 
                                    scoring = 'roc_auc')

print(pd.np.mean(cv_ClassTree_scores))

Class_tree_Cv_Score = pd.np.mean(cv_ClassTree_scores).round(3)

###############################################################################
# Random Forest
###############################################################################

# Repreparing train/test split

got_data = got.loc[: , ['male',
                        'popularity',
                        'per_alive',
                        'dateOfBirth',
                        'book4_A_Feast_For_Crows'
                        ]]

got_target =  got.loc[: , 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.10,
            random_state = 508,
            stratify = got_target)

####################################
# Parameter tuning with GridSearchCV
####################################

# Creating a hyperparameter grid

estimator_space = pd.np.arange(100, 1350, 250)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['gini', 'entropy']
bootstrap_space = [True, False]
warm_start_space = [True, False]

param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}

# Building the model object
full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 508)

# Creating a GridSearchCV object
full_forest_cv = GridSearchCV(full_forest_grid, param_grid, cv = 3)

# Fit it to the training data
full_forest_cv.fit(X_train, y_train)

# Print the optimal parameters and best score

print("Tuned Random Forest Parameter:", full_forest_cv.best_params_)
print("Tuned Random Forest Accuracy:", 
      full_forest_cv.best_score_.round(4))

##########################################################
# Building Random Forest Model Based on Optimal Parameters
##########################################################

rf_optimal = RandomForestClassifier(bootstrap = True,
                                    criterion = 'entropy',
                                    min_samples_leaf = 16,
                                    n_estimators = 600,
                                    warm_start = False,
                                    random_state = 508)

rf_optimal_fit = rf_optimal.fit(X_train, y_train)

rf_optimal_pred = rf_optimal.predict(X_test)

# Training and Testing Scores

print('Training Score', rf_optimal.score(X_train, y_train).round(4))
print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))

train = rf_optimal.score(X_train, y_train)
test  = rf_optimal.score(X_test, y_test)

print((train - test).round(4))

""" Cross Validation score """

cv_RandomForest_scores = cross_val_score(rf_optimal_fit,
                                    got_data,
                                    got_target,
                                    cv=3, 
                                    scoring = 'roc_auc')
                                    

print(pd.np.mean(cv_RandomForest_scores))

RandomForest_Cv_Score = pd.np.mean(cv_RandomForest_scores).round(3)

###############################################################################
# Gradient Boosted Machines
###############################################################################

####################################
# Parameter tuning with GridSearchCV
####################################

# Creating a hyperparameter grid

learn_space = pd.np.arange(0.1, 1.6, 0.1)
estimator_space = pd.np.arange(50, 250, 50)
depth_space = pd.np.arange(1, 10)
criterion_space = ['friedman_mse', 'mse', 'mae']

param_grid = {'learning_rate' : learn_space,
              'max_depth' : depth_space,
              'criterion' : criterion_space,
              'n_estimators' : estimator_space}

# Building the model object
gbm_grid = GradientBoostingClassifier(random_state = 508)

# Creating a GridSearchCV object
gbm_grid_cv = GridSearchCV(gbm_grid, param_grid, cv = 3)

# Fit it to the training data
gbm_grid_cv.fit(X_train, y_train)

# Print the optimal parameters and best score

print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))

################################################
# Building GBM Model Based on Optimal Parameters
################################################

gbm_optimal = GradientBoostingClassifier(criterion = 'friedman_mse',
                                      learning_rate = 0.1,
                                      max_depth = 4,
                                      n_estimators = 50,
                                      random_state = 508)

gbm_optimal_fit = gbm_optimal.fit(X_train, y_train)

gbm_optimal_score = gbm_optimal.score(X_test, y_test)

gbm_optimal_pred = gbm_optimal.predict(X_test)

# Training and Testing Scores

print('Training Score', gbm_optimal.score(X_train, y_train).round(4))
print('Testing Score:', gbm_optimal.score(X_test, y_test).round(4))

train = gbm_optimal.score(X_train, y_train)
test  = gbm_optimal.score(X_test, y_test)

print((train - test).round(4))

""" Cross Validation score """

cv_GBM_scores = cross_val_score(gbm_optimal_fit,
                                    got_data,
                                    got_target,
                                    cv = 3, 
                                    scoring = 'roc_auc')
                                   

print(pd.np.mean(cv_GBM_scores))

GBM_Cv_Score = pd.np.mean(cv_GBM_scores).round(3)


##############################################################################
''' From all the models above mean AUC score of 0.867, after Cross Validation 
    remained highest for Random Forest therefore it's our best model '''
##############################################################################
    
    
###############################################################################
# Saving Results for best model (Random Forest)
###############################################################################

# Saving best model (Random Forest) Cross Validation Score #

model_scores_df = pd.DataFrame({'RF_Score': cv_RandomForest_scores,
                                'RF_mean_score' : RandomForest_Cv_Score})

model_scores_df.to_excel("Random_Forest_CV_scores.xlsx", index = False)

# Saving model (Random Forest) predictions #

model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'RF_Predicted': rf_optimal_pred})

model_predictions_df.to_excel("Random_Forest_predictions.xlsx", index = False)

###############################################################################
# Insightful Illustrations for Best model (Random Forest)
###############################################################################

#################
#Confusion Matrix 
#################

print(confusion_matrix(y_true = y_test,
                       y_pred = rf_optimal_pred))

# Visualizing confusion matrix #

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.
               format(accuracy, misclass))
    
    plt.savefig('RandomForest_Confusion_Matrix.png')
    plt.show()

plot_confusion_matrix(cm           = np.array([[ 24 , 26], 
                                               [ 8  , 137]]),
                      normalize    = False,
                      target_names = ['Not Alive', 'Alive'],
                      title        = "Confusion Matrix")
    
''' Model correctly predicted, 137 alive characters and 
    24 dead characters. 
    
    8 characters were incorrectly identified dead.
    
    26 characters were incorrectly identified alive. '''

#################################
#Creating a classification report
#################################

labels = ['Not Alive', 'Alive']    

rf_optimal.predict(X_test)

print(classification_report(y_true = y_test,
                            y_pred = rf_optimal_pred,
                            target_names = labels))

""" Of the entire test set, 83% of predicted alive status (dead/alive) 
    were the actual alive status (dead/alive) for characters.

   Model precision of 84% states model's ability to not predict a character 
   alive if it's dead.

   Model recall states that the model correctly identified alive characters 
   as alive 94% of the times.

   f1-score states that it correctly identified more alive than dead 
   characters as it has a high score for Alive v/s Not Alive. It's best 
   interpreted as weighted harmonic mean of precision and recall, and it 
   reaches it's best value at 1 and worst score at 0.

   Support tells us, that with in our test set, actual dead characters were 
   50 and actual alive characters were 145."""

############################
#Building feature importance
############################

def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('RandomeForest_Importance.png')

plot_feature_importances(rf_optimal,
                         train = X_train,
                         export = True)

plt.savefig('RandomForest_Feature_Importance.png')

''' Features in terms of their importance in predicting 
    alive status(Dead / Alive) of a character are listed below:
    
    1) Proportion of alive members per house
    2) Popularity of the character
    3) Date of birth of the character
    4) Character appearance in book 4
    4) Zero dead realtives of the character
    5) Male character 
    
    Insight for these features is provided in write up '''

#####################
#Building a ROC curve
#####################

logit_roc_auc = roc_auc_score(y_test, rf_optimal.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, 
                                 rf_optimal.predict_proba(X_test)[:,1])
plt.figure()

plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.savefig('Log_ROC')
plt.show()

""" Random Forest is a good classifier as it stays far away from purely random
classifier ROC curve denoted as dotted line """

################################# The End #####################################

""" Appendix """

#################################
#Applying Stanadard Scalar to KNN
#################################

#Redefining train/test split

got_2_data = got.loc[ : ,['male',
                        'popularity',
                        'per_alive',
                        'dateOfBirth',
                        'book4_A_Feast_For_Crows',
                        'new_numDeadRelations']]

got2_target =  got.loc[: , 'isAlive']

XS_train, XS_test, yS_train, yS_test = train_test_split(
            got_2_data,
            got2_target,
            test_size = 0.10,
            random_state = 508,
            stratify = got_target)

# Removing the target variable. It is (generally) not necessary to scale that.

got_features = got_2_data

# Instantiating a StandardScaler() object
scaler = StandardScaler()

# Fitting the scaler with our data
scaler.fit(got_features)

# Transforming our data after fit
X_scaled = scaler.transform(got_features)

# Putting our scaled data into a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)

# Running KNN Once more #

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    
    # build the model
    
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(XS_train, yS_train.values.ravel())
    
    # record training set accuracy
    
    training_accuracy.append(clf.score(XS_train, yS_train))
    
    # record generalization accuracy
    
    test_accuracy.append(clf.score(XS_test, yS_test))


fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "trainingScaled accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "testScaled accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

plt.show()

# Looking for the highest test accuracy

print(test_accuracy)

# Printing highest test accuracy

print(test_accuracy.index(max(test_accuracy)) + 1)

# It looks like 5 neighbors is the most accurate

knn_clf = KNeighborsClassifier(n_neighbors = 5)

# Fitting the model based on the training data

knn_clf_fit_Scaled = knn_clf.fit(XS_train, yS_train)

# Let's compare the testing score to the training score #

print('Training Score', knn_clf_fit.score(XS_train, yS_train).round(4))
print('Testing Score:', knn_clf_fit.score(XS_test, yS_test).round(4))

train = knn_clf_fit.score(XS_train, yS_train).round(4)
test  = knn_clf_fit.score(XS_test, yS_test).round(4)

print (train - test)

""" Cross Validation Score """

cv_knn_3_Scaled = cross_val_score(knn_clf_fit_Scaled, 
                           got_data, 
                           got_target,
                           cv = 3, 
                           scoring = 'roc_auc')

print(pd.np.mean(cv_knn_3))

KNN_CV_Score_Scaled = pd.np.mean(cv_knn_3_Scaled).round(3)

""" No difference observed, in previous KNN and Scaled KNN """