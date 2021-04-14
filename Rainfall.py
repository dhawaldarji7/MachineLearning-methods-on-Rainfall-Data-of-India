#!/usr/bin/python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import seaborn as sb
import xgboost as xg
import scikitplot as skp
from sklearn import model_selection, preprocessing, neighbors, tree, svm
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, confusion_matrix, \
    accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.plotting.decision_regions import plot_decision_regions
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from itertools import cycle

pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Preprocessing & generating data

raindf = pd.read_csv('./dataset/rainfall_india.csv')

# Generating the REGION label for original dataset
region = []
for row in raindf.iterrows():

    if row[1].values[14] < 500.0:
        region.append("SCANTY")

    elif (row[1].values[14] >= 500.0) & (row[1].values[14] < 1000.0):
        region.append("LOW")

    elif (row[1].values[14] >= 1000.0) & (row[1].values[14] < 2000.0):
        region.append("MODERATE")

    else:
        region.append("HEAVY")

raindf['REGION'] = pd.DataFrame(region)
region.clear()

# Avg rainfall over the years 1901-2015 for each division
division_wise_overall_avg = pd.DataFrame(raindf.groupby('SUBDIVISION').mean())
division_wise_overall_avg = division_wise_overall_avg.drop('YEAR', 1)

# Generating the REGION label for subdivision wise data
sdregion = []
for row in division_wise_overall_avg.iterrows():
    if row[1].values[-5] < 500.0:
        sdregion.append("SCANTY")

    elif (row[1].values[-5] >= 500.0) & (row[1].values[-5] < 1000.0):
        sdregion.append("LOW")

    elif (row[1].values[-5] >= 1000.0) & (row[1].values[-5] < 2000.0):
        sdregion.append("MODERATE")

    else:
        sdregion.append("HEAVY")

division_wise_overall_avg['REGION'] = sdregion
division_wise_overall_avg.to_csv('./dataset/dist_wise_avg.csv')

# Avg rainfall over the years 1901-2017 in India
year_wise_overall_avg = pd.DataFrame(raindf.groupby('YEAR').sum())
year_wise_overall_avg.to_csv('./dataset/year_wise_avg.csv')

# Finding out number of entries for each subdivision
entries_for_div = raindf.groupby('SUBDIVISION').size()

# Finding the total number of subdivisions
subdiv = raindf['SUBDIVISION'].unique()

# Filling in the NA values with the mean values of the column for that particular subdivision
for sd in subdiv:
    raindf[raindf['SUBDIVISION'] == sd] = raindf[raindf['SUBDIVISION'] == sd]\
        .fillna(raindf[raindf['SUBDIVISION'] == sd].mean())

# Converting SUBDIVISION from Categorical to Numerical feature
sd_trans = {}
i = 0
for sd in subdiv:
    sd_trans[sd] = i
    i += 1

# Creating dataset with records of previous three years for any given year

year = []
ann = []
subd = []
p1 = []
jjas1 = []
p2 = []
jjas2 = []
p3 = []
jjas3 = []

for sd in subdiv:

    df = raindf[raindf['SUBDIVISION'] == sd]
    df = df.fillna(df.mean())

    start_index = df.index[0]
    for index, row in df.iterrows():
        if index >= start_index + 3:
            i = df.index.get_loc(index)
            subd.append(sd)
            year.append(row[1])
            ann.append(row[-6])

            p1.append(df.iloc[i - 1, -6])
            jjas1.append(df.iloc[i - 1, -3])

            p2.append(df.iloc[i - 2, -6])
            jjas2.append(df.iloc[i - 2, -3])

            p3.append(df.iloc[i - 3, -6])
            jjas3.append(df.iloc[i - 3, -3])

rain_past = pd.DataFrame({'SUBDIVISION': subd,
                          'YEAR': year,
                          'JJAS1': jjas1,
                          'P1': p1,
                          'JJAS2': jjas2,
                          'P2': p2,
                          'JJAS3': jjas3,
                          'P3': p3,
                          'ANNUAL': ann}, columns=['SUBDIVISION', 'YEAR', 'JJAS1', 'P1', 'JJAS2', 'P2',
                                                   'JJAS3', 'P3', 'ANNUAL'])

# Writing out the newly generated, processed & cleaned data to csv files
rain_past.to_csv('./dataset/rainfall_past.csv')
raindf.to_csv('./dataset/rainfall_india_processed.csv')

# print(rain_past.head())
def getkey(num):
    for sd, number in sd_trans.items():
        if number == num:
            return sd


# Regression Methods

# Data Splitting
Xrdf = np.array(rain_past[['P1', 'P2', 'P3', 'JJAS1', 'JJAS2', 'JJAS3']])
yrdf = np.array(rain_past['ANNUAL'])

Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(Xrdf, yrdf, test_size=0.2, shuffle=True)
Xtrainv, Xtestv, ytrainv, ytestv = model_selection.train_test_split(Xtrain, ytrain, test_size=0.1, shuffle=True)


# Format for printing of results
# the below format is followed while printing the results of each method
# "RMSE using method_name: trainRMSE validationRMSE testRMSE"

# Linear Regression
# linear = LinearRegression()
# linear.fit(Xtrainv, ytrainv)

# linear_train_pred = linear.predict(Xtrainv)
# linear_val_pred = linear.predict(Xtestv)
# linear_test_pred = linear.predict(Xtest)

# linear_train_mse = mean_squared_error(ytrainv, linear_train_pred)
# linear_val_mse = mean_squared_error(ytestv, linear_val_pred)
# linear_test_mse = mean_squared_error(ytest, linear_test_pred)

# print("RMSE using LinearRegression:\t{}\t{}\t{}".format(np.sqrt(linear_train_mse), np.sqrt(linear_val_mse),
#                                                     np.sqrt(linear_test_mse)))

# # Ridge Regression

# ridge = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
#                 cv=15)
# ridge.fit(Xtrainv, ytrainv)

# ridge_train_pred = ridge.predict(Xtrainv)
# ridge_val_pred = ridge.predict(Xtestv)
# ridge_test_pred = ridge.predict(Xtest)

# ridge_train_mse = mean_squared_error(ytrainv, ridge_train_pred)
# ridge_val_mse = mean_squared_error(ytestv, ridge_val_pred)
# ridge_test_mse = mean_squared_error(ytest, ridge_test_pred)

# print("RMSE using RidgeRegression:\t{}\t{}\t{}".format(np.sqrt(ridge_train_mse), np.sqrt(ridge_val_mse),
#                                                     np.sqrt(ridge_test_mse)))

# # Lasso Regression

# lasso = LassoCV(alphas=[0.01, 0.1, 1, 10, 100],
#                 max_iter=100,
#                 cv=10,
#                 tol=0.1)
# lasso.fit(Xtrainv, ytrainv)

# lasso_train_pred = lasso.predict(Xtrainv)
# lasso_val_pred = lasso.predict(Xtestv)
# lasso_test_pred = lasso.predict(Xtest)

# lasso_train_mse = mean_squared_error(ytrainv, lasso_train_pred)
# lasso_val_mse = mean_squared_error(ytestv, lasso_val_pred)
# lasso_test_mse = mean_squared_error(ytest, lasso_test_pred)

# print("RMSE using LassoRegression:\t{}\t{}\t{}".format(np.sqrt(lasso_train_mse), np.sqrt(lasso_val_mse),
#                                                     np.sqrt(lasso_test_mse)))

# # ElasticNet Regression

# elastic = ElasticNetCV(l1_ratio=0.008,
#                        n_alphas=100,
#                        max_iter=20,
#                        cv=10,
#                        tol=0.1)
# elastic.fit(Xtrainv, ytrainv)

# elastic_train_pred = elastic.predict(Xtrainv)
# elastic_val_pred = elastic.predict(Xtestv)
# elastic_test_pred = elastic.predict(Xtest)

# elastic_train_mse = mean_squared_error(ytrainv, elastic_train_pred)
# elastic_val_mse = mean_squared_error(ytestv, elastic_val_pred)
# elastic_test_mse = mean_squared_error(ytest, elastic_test_pred)

# print("RMSE using ElasticNetRegression:\t{}\t{}\t{}".format(np.sqrt(elastic_train_mse), np.sqrt(elastic_val_mse),
#                                                     np.sqrt(elastic_test_mse)))

# df_rf = pd.DataFrame({'Actual': ytest, 'Predicted': elastic_test_pred})
# fig1 = pp.figure(figsize=(8, 6))
# df_rf.head(n=300).plot()
# pp.legend()
# pp.title("ElasticNet Actual v/s Predicted Annual Rainfall")
# pp.xticks(())
# pp.yticks(())
# pp.show()

# # RandomForests Regressor

# rfreg = RandomForestRegressor(n_estimators=1232, min_samples_split=10, min_samples_leaf=5, max_features='sqrt',
#                               max_depth=29)
# rfreg = RandomForestRegressor(n_estimators=5000)
# rfreg.fit(Xtrainv, ytrainv)

# rf_train_pred = rfreg.predict(Xtrainv)
# rf_val_pred = rfreg.predict(Xtestv)
# rf_test_pred = rfreg.predict(Xtest)

# rf_train_mse = mean_squared_error(ytrainv, rf_train_pred)
# rf_val_mse = mean_squared_error(ytestv, rf_val_pred)
# rf_test_mse = mean_squared_error(ytest, rf_test_pred)

# print("RMSE using RandomForestRegressor:\t{}\t{}\t{}".format(np.sqrt(rf_train_mse), np.sqrt(rf_val_mse),
#                                                     np.sqrt(rf_test_mse)))

# df_rf = pd.DataFrame({'Actual': ytest, 'Predicted': rf_test_pred})
# fig1 = pp.figure(figsize=(8, 6))
# df_rf.head(n=300).plot()
# pp.legend()
# pp.title("RandomForests Actual v/s Predicted Annual Rainfall")
# pp.xticks(())
# pp.yticks(())
# pp.show()

# # Gradient Boosting Regressor

# gbreg = GradientBoostingRegressor()
# gbreg.fit(Xtrainv, ytrainv)

# gb_train_pred = gbreg.predict(Xtrainv)
# gb_val_pred = gbreg.predict(Xtestv)
# gb_test_pred = gbreg.predict(Xtest)

# gb_train_mse = mean_squared_error(ytrainv, gb_train_pred)
# gb_val_mse = mean_squared_error(ytestv, gb_val_pred)
# gb_test_mse = mean_squared_error(ytest, gb_test_pred)

# print("RMSE using GradientBoostingRegressor:\t{}\t{}\t{}".format(np.sqrt(gb_train_mse), np.sqrt(gb_val_mse),
#                                                     np.sqrt(gb_test_mse)))

# # MultiLayerPerceptron Regressor

# mlpr = MLPRegressor(hidden_layer_sizes=100,
#                     alpha=100,
#                     tol=0.01,
#                     learning_rate_init=0.099,
#                     early_stopping=True,
#                     shuffle=True)

# mlpr.fit(Xtrainv, ytrainv)

# mlpr_train_pred = mlpr.predict(Xtrainv)
# mlpr_val_pred = mlpr.predict(Xtestv)
# mlpr_test_pred = mlpr.predict(Xtest)

# mlpr_train_mse = mean_squared_error(ytrainv, mlpr_train_pred)
# mlpr_val_mse = mean_squared_error(ytestv, mlpr_val_pred)
# mlpr_test_mse = mean_squared_error(ytest, mlpr_test_pred)

# print("RMSE using MLPRegressor:\t{}\t{}\t{}".format(np.sqrt(mlpr_train_mse), np.sqrt(mlpr_val_mse),
#                                                     np.sqrt(mlpr_test_mse)))

# estimators = [
#     ('el', ElasticNetCV(l1_ratio=0.008,
#                         n_alphas=100,
#                         max_iter=20,
#                         cv=10,
#                         tol=0.1)),
#     ('rf', RandomForestRegressor(n_estimators=1400,
#                                  min_samples_split=2,
#                                  min_samples_leaf=2,
#                                  max_features='sqrt',
#                                  max_depth=5,
#                                  oob_score=True)),
# ]

# stack = StackingRegressor(
#     estimators=estimators,
#     final_estimator=RandomForestRegressor(n_estimators=1400,
#                                           min_samples_split=2,
#                                           min_samples_leaf=2,
#                                           max_features='sqrt',
#                                           max_depth=5,
#                                           oob_score=True)
# )

# stack.fit(Xtrainv, ytrainv)
# stack_train_pred = stack.predict(Xtrainv)
# stack_val_pred = stack.predict(Xtestv)
# stack_test_pred = stack.predict(Xtest)

# stack_train_mse = mean_squared_error(ytrainv, stack_train_pred)
# stack_val_mse = mean_squared_error(ytestv, stack_val_pred)
# stack_test_mse = mean_squared_error(ytest, stack_test_pred)

# print("RMSE using StackRegressor:\t{}\t{}\t{}\n".format(np.sqrt(stack_train_mse), np.sqrt(stack_val_mse),
#                                                     np.sqrt(stack_test_mse)))

# df_rf = pd.DataFrame({'Actual': ytest, 'Predicted': stack_test_pred})
# fig1 = pp.figure(figsize=(8, 6))
# df_rf.head(n=300).plot()
# pp.legend()
# pp.title("StackRegressor Actual v/s Predicted Annual Rainfall")
# pp.xticks(())
# pp.yticks(())
# pp.show()

# print(rfreg.score(Xtest, ytest), elastic.score(Xtest, ytest), stack.score(Xtest, ytest))

# CLASSIFICATION & CLUSTERING METHODS

# Data splitting
X = np.array(raindf[['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC']])
y = np.array(raindf['REGION'])

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.fit_transform(y)

Xreduced = PCA(n_components=2).fit_transform(X)
results = pd.DataFrame(Xreduced,columns=['pca1', 'pca2'])

Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(Xreduced, y, test_size=0.3)

# Data with region to be predicted
Xt = np.array(division_wise_overall_avg[['JAN-FEB', 'MAR-MAY', 'JUN-SEP', 'OCT-DEC']])
yt = np.array(division_wise_overall_avg['REGION'])

Xtreduced = PCA(n_components=2).fit_transform(Xt)
resultst = pd.DataFrame(Xtreduced, columns=['pca1', 'pca2'])

# CLASSIFICATION

# SUPPORT VECTOR MACHINE
svm_linear = svm.SVC(kernel='linear')
svm_rad = svm.SVC(kernel='poly')

svm_linear.fit(Xtrain, ytrain)
svm_rad.fit(Xtrain, ytrain)

svm_linear_pred = svm_linear.predict(Xtest)
svm_poly_pred = svm_rad.predict(Xtest)

print("Classification accuracy using SVM linear classifier is: {}".format(accuracy_score(ytest, svm_linear_pred) * 100))
print("Classification accuracy using SVM radial classifier is: {}".format(accuracy_score(ytest, svm_poly_pred) * 100))

# K-NEAREST NEIGHBOURS
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtrain, ytrain)
knn_pred = knn.predict(Xtest)
print("Classification accuracy using KNN classification is: {}".format(accuracy_score(ytest, knn_pred) * 100))

# DECISION TREES
dt = tree.DecisionTreeClassifier()
dt.fit(Xtrain, ytrain)
dt_pred = dt.predict(Xtest)
print("Classification accuracy using Decision tress is: {}".format(accuracy_score(ytest, dt_pred) * 100))

# XGBOOST CLASSIFIER
xgc = xg.XGBClassifier()
xgc.fit(Xtrain, ytrain)
xg_pred = xgc.predict(Xtest)
print("Classification accuracy using XGBoost classifier is: {}".format(accuracy_score(ytest, xg_pred) * 100))

# MLP CLASSIFIER
mlpc = MLPClassifier(alpha=200, max_iter=1000, tol=0.0009)
mlpc.fit(Xtrain, ytrain)
mlpc_pred = mlpc.predict(Xtest)
print("Classification accuracy using MLP classifier is: {}".format(accuracy_score(ytest, mlpc_pred) * 100))

# CLUSTERING METHODS

# K-MEANS
km = KMeans(n_clusters=4)
km.fit(Xtrain)
labels = km.labels_
km_pred = km.predict(Xtreduced)

region_dict = {'SCANTY': 0,
               'LOW': 1,
               'MODERATE': 2,
               'HEAVY': 3}

# Converting Ytest from Categorical to Numerical
ytest_new = []
for label in yt:
    ytest_new.append(region_dict.get(label))
yt = np.array(ytest_new)

classplot = pp.figure(figsize=(8, 8))
# Change the third parameter to plot for other classifiers (svm_linear, svm_rad, knn, dt, xgc, mlpc)
plot_decision_regions(Xtest, ytest, svm_linear)
pp.title('Classification into Regions using K-Means', size=18)
legend = pp.legend()
legend.get_texts()[0].set_text('SCANTY')
legend.get_texts()[1].set_text('LOW')
legend.get_texts()[2].set_text('MODERATE')
legend.get_texts()[3].set_text('HEAVY')
pp.xlabel("PC1", size=14)
pp.ylabel("PC2", size=14)
pp.show()
