# Unsupervised_Learning_with_Clustering

#### Importing Relevant Libraries
The script starts by importing the necessary libraries for data analysis, visualization, clustering, and supervised machine learning. These libraries include pandas, numpy, matplotlib, seaborn, and various modules from scikit-learn. 

#### Data Preprocessing
The script reads a CSV file named 'cluster.csv' using the pandas `read_csv()` function and assigns it to the variable `df`. The data is then copied to another variable `data`. The script performs some basic data preprocessing steps, such as dropping the 'CUST_ID' column, handling missing values in the 'CREDIT_LIMIT' and 'MINIMUM_PAYMENTS' columns, and checking for any remaining missing values.

#### Modeling with K-means
The script uses the K-means clustering algorithm to cluster the data. It calculates the within-cluster sum of squares (WCSS) for different numbers of clusters (ranging from 1 to 6) and plots the WCSS values against the number of clusters to determine the optimal number of clusters. It then fits the K-means model with 5 clusters and assigns the cluster labels to the 'CLUSTER' column in the `data` DataFrame. Finally, it visualizes the clusters by creating a scatter plot of 'BALANCE' against 'CREDIT_LIMIT' with different colors representing different clusters.

#### Modeling with Agglomerative Clustering
The script performs agglomerative clustering on the data using the `AgglomerativeClustering` class from scikit-learn. It scales the data using the `StandardScaler` class, creates a dendrogram to visualize the hierarchical clustering, and assigns cluster labels to the 'CLUSTER' column in the `data2` DataFrame. It then visualizes the clusters by creating a scatter plot of 'BALANCE_FREQUENCY' against 'CASH_ADVANCE_FREQUENCY' with different colors representing different clusters.

#### Feature Selection
The script performs feature selection by calculating the correlation between each feature and the 'CLUSTER' column in the `data2` DataFrame. It drops the columns with absolute correlation values below the average correlation value. It also drops the 'PURCHASES' and 'PURCHASES_FREQUENCY' columns from the DataFrame.

#### Outlier Treatment
The script performs outlier treatment by replacing values above the upper bound and below the lower bound (calculated using the interquartile range) with the respective bounds for the features 'BALANCE_FREQUENCY', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', and 'PURCHASES_TRX'.

#### Applying Supervised ML using Agglomerative Clustering
The script applies supervised machine learning using the XGBoost classifier. It splits the data into training and testing sets, fits the XGBoost model on the training data, and evaluates the model's performance on the testing data. It calculates the Gini score, confusion matrix, and classification report for both the training and testing data. It also plots the ROC curve for the model's predictions on the testing data.

#### Univariate Analysis
The script performs univariate analysis by training separate XGBoost models on each individual feature and calculating the Gini scores for both the training and testing data. It creates a DataFrame that ranks the features based on their test Gini scores.
