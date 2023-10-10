# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 19:27:07 2023

@author: Ariana Negreiro
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, roc_auc_score

#############################################################################################################
#### Part 1 Q5 ####

# Given data
confidences = [0.95, 0.85, 0.8, 0.7, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1]
correct_classes = ['+', '+', '-', '+', '+', '-', '+', '+', '-', '-']

# Sort the predictions by descending confidence
sorted_data = sorted(zip(confidences, correct_classes), key=lambda x: -x[0])

# Initialize variables for ROC curve
true_positive_rate = []  # Sensitivity
false_positive_rate = []  # 1 - Specificity
num_positives = correct_classes.count('+')
num_negatives = correct_classes.count('-')
true_positives = 0
false_positives = 0

# Calculate TPR and FPR for different thresholds
for _, label in sorted_data:
    if label == '+':
        true_positives += 1
    else:
        false_positives += 1

    true_positive_rate.append(true_positives / num_positives)
    false_positive_rate.append(false_positives / num_negatives)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(false_positive_rate, true_positive_rate, marker='o', linestyle='-', color='b')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.show()


####################################################################################################################

#### Part 2 Q1 #####

training_data = np.loadtxt(r'C:\Users\Ariana Negreiro\Dropbox\Classes\CS760\hw\hw3\hw3Data\D2z.txt')

X_train = training_data[:, :-1]
y_train = training_data[:, -1]

# make a grid of test points
x1_range = np.arange(-2, 2.1, 0.1)
x2_range = np.arange(-2, 2.1, 0.1)
test_points = np.array([(x1, x2) for x1 in x1_range for x2 in x2_range])

####################### KNN from scratch #################################
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, y_train, X_test, k):
    y_pred = []
    y_prob = []
    
    if type(X_test) == np.ndarray:
        
        for x in X_test:
            # Compute distances between x and all examples in the training set
            distances = [euclidean_distance(x, x_train) for x_train in X_train]
            
            # Sort by distance and return indices of the first k neighbors
            k_indices = np.argsort(distances)[:k]
            
            # Extract the labels of the k nearest neighbor training samples
            k_nearest_labels = [y_train[i] for i in k_indices]

            # Return the most common class label
            most_common = np.bincount(k_nearest_labels).argmax()
            y_pred.append(most_common)
            
            # Calculate class probabilities
            class_probs = [k_nearest_labels.count(1) / k] #binary classification

            y_prob.append(class_probs)
    
    else:
        for x in X_test.values:
            # Ensure all values are of numeric data type
            x = x.astype(float)
            
            # Compute distances between x and all examples in the training set
            distances = [euclidean_distance(x, x_train) for x_train in X_train.values]
            
            # Sort by distance and return indices of the first k neighbors
            k_indices = np.argsort(distances)[:k]
            
            # Extract the labels of the k nearest neighbor training samples
            k_nearest_labels = [y_train[i] for i in k_indices]
            
            # Return the most common class label
            most_common = np.bincount(k_nearest_labels).argmax()
            y_pred.append(most_common)
            
            # Calculate class probabilities
            class_probs = [k_nearest_labels.count(1) / k] #binary classification
            y_prob.append(class_probs)
    
    return np.array(y_pred), np.array(y_prob)

############################################################################

# calculate 1NN predictions for the test points
y_pred, y_prob = knn_predict(X_train, y_train, test_points, 1)


# Step 4: Visualize the results
plt.figure(figsize=(8, 8))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', cmap=plt.cm.RdYlBu, label='Training Set', s=100)
plt.scatter(test_points[:, 0], test_points[:, 1], c=y_pred, marker='x', cmap=plt.cm.RdYlBu, label='Test Grid', s=30)
plt.title('Visualized predictions of 1NN on a 2D grid')
plt.legend()
plt.grid(True)
plt.show()


####################################################################################################################

#### Part 2 Q2 ####

df = pd.read_csv(r'C:\Users\Ariana Negreiro\Dropbox\Classes\CS760\hw\hw3\hw3Data\emails.csv')

df['idx'] = df['Email No.'].str.split(' ').str[1]
df.set_index('idx', inplace=True) 
df = df.drop(columns=['Email No.'])


X = df.drop('Prediction', axis=1)
y = df['Prediction']

# for 5-fold cross validation, split the dataset to specifications
folds = [
    (1, 1000),
    (1000, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 5000)
]

total = 5000

# Create lists to store the metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []

for i, (test_start, test_end) in enumerate(folds):
    
    # separate training and testing sets
    train_indices = np.concatenate((np.arange(0, test_start), np.arange(test_end, total)))
    test_indices = np.arange(test_start, test_end)
    
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    y_pred, y_prob = knn_predict(X_train, y_train, X_test, 1)

    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    
    print(f"Fold {i +1}, test set: Email {test_start}-{test_end}, training set: the rest")
    print(f"Accuracy = {accuracy:.2f}, Precision = {precision:.2f}, Recall = {recall:.2f}")


###################################################################################################
#### Part 2 Q3 #####

df = pd.read_csv(r'C:\Users\Ariana Negreiro\Dropbox\Classes\CS760\hw\hw3\hw3Data\emails.csv')

df['idx'] = df['Email No.'].str.split(' ').str[1]
df.set_index('idx', inplace=True) 
df = df.drop(columns=['Email No.'])

X = df.drop('Prediction', axis=1)
y = df['Prediction']

# for 5-fold cross validation, split the dataset to specifications
folds = [
    (1, 1000),
    (1000, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 5000)
]

total = 5000

# Create lists to store the metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []

############################# logistic regression from scratch ##############################

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def model(X, y, learning_rate, iterations):
    m = X.shape[1]
    n = X.shape[0]
    W = np.zeros((n,1))
    B = 0
    cost_list = []
    
    for i in range(iterations):
        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)
        
        # cost function
        cost = -(1/m)*np.sum( y*np.log(A) + (1-y)*np.log(1-A))
        
        # Gradient Descent
        dW = (1/m)*np.dot(A-y, X.T)
        dB = (1/m)*np.sum(A - y)
        
        W = W - learning_rate*dW.T
        B = B - learning_rate*dB
        
        # Keeping track of our cost function value
        cost_list.append(cost)
        
        if(i%(iterations/10) == 0):
            print("cost after ", i, "iteration is : ", cost)
            print(W)
            print(B)
        
    return W, B, cost_list
        
def accuracy_precision_recall(X, y, W, B):
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)
    A = A > 0.5
    A = np.array(A, dtype='int64')

    # Accuracy calculation
    acc = (1 - np.sum(np.absolute(A - y)) / y.shape[1]) * 100

    # Precision calculation
    true_positives = np.sum((y == 1) & (A == 1))
    false_positives = np.sum((y == 0) & (A == 1))
    precision = true_positives / (true_positives + false_positives)

    # Recall calculation
    false_negatives = np.sum((y == 1) & (A == 0))
    recall = true_positives / (true_positives + false_negatives)

    return round(acc, 2), round(precision, 2), round(recall, 2)

#################### end logistic regression from scratch ######################

# Set hyperparameters
learning_rate = 0.0001
iterations = 1000

# Create lists to store the metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []

for i, (test_start, test_end) in enumerate(folds):
    
    # separate training and testing sets
    train_indices = np.concatenate((np.arange(0, test_start), np.arange(test_end, total)))
    test_indices = np.arange(test_start, test_end)
    
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    y_train = y_train.values
    y_test = y_test.values
    
    X_train = X_train.T
    y_train = y_train.reshape(1, X_train.shape[1])
    X_test = X_test.T
    y_test = y_test.reshape(1, X_test.shape[1])
    
    W, B, cost_list = model(X_train, y_train, learning_rate = learning_rate, iterations = iterations)
    
    # Calculate accuracy, precision, and recall
    accuracy, precision, recall = accuracy_precision_recall(X_test, y_test, W, B)
    
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    
    print(f"Fold {i +1}, test set: Email {test_start}-{test_end}, training set: the rest")
    print(f"Accuracy = {accuracy:.2f}, Precision = {precision:.2f}, Recall = {recall:.2f}")

################################################################################################################
#### Part 2 Q4 #####

df = pd.read_csv(r'C:\Users\Ariana Negreiro\Dropbox\Classes\CS760\hw\hw3\hw3Data\emails.csv')

df['idx'] = df['Email No.'].str.split(' ').str[1]
df.set_index('idx', inplace=True) 
df = df.drop(columns=['Email No.'])

X = df.drop('Prediction', axis=1)
y = df['Prediction']

# for 5-fold cross validation, split the dataset to specifications
folds = [
    (1, 1000),
    (1000, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 5000)
]

total = 5000

# Create lists to store the metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []

k_values = [1, 3, 5, 7, 10]

for k in k_values:
    
    # Create lists to store the metrics for each fold
    fold_accuracy_scores = []
    fold_precision_scores = []
    fold_recall_scores = []

    for i, (test_start, test_end) in enumerate(folds):
        
        # separate training and testing sets
        train_indices = np.concatenate((np.arange(0, test_start), np.arange(test_end, total)))
        test_indices = np.arange(test_start, test_end)
        
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        
        y_pred, y_prob = knn_predict(X_train, y_train, X_test, k)
    
        # Calculate accuracy, precision, and recall
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        fold_accuracy_scores.append(accuracy)
        fold_precision_scores.append(precision)
        fold_recall_scores.append(recall)

        print(f"K = {k}, Fold {i +1}, test set: Email {test_start}-{test_end}, training set: the rest")
        print(f"Accuracy = {accuracy:.2f}, Precision = {precision:.2f}, Recall = {recall:.2f}")
        
    # Compute average metrics for this k
    avg_accuracy = sum(fold_accuracy_scores) / len(fold_accuracy_scores)
    avg_precision = sum(fold_precision_scores) / len(fold_precision_scores)
    avg_recall = sum(fold_recall_scores) / len(fold_recall_scores)

    accuracy_scores.append(avg_accuracy)
    precision_scores.append(avg_precision)
    recall_scores.append(avg_recall)

    print(f"Average Metrics for k={k}:")
    print(f"Average Accuracy = {avg_accuracy:.2f}, Average Precision = {avg_precision:.2f}, Average Recall = {avg_recall:.2f}")

# Plot average accuracy versus k
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='-')
plt.title('KNN')
plt.xlabel('k')
plt.ylabel('Average Accuracy')
plt.grid(True)
plt.xticks(k_values)
plt.show()

# List the average accuracy of each case
for i, k in enumerate(k_values):
    print(f"Average Accuracy for k={k}: {accuracy_scores[i]:.2f}")


###########################################################################################################
#### Part 2 Q5 #####
    
df = pd.read_csv(r'C:\Users\Ariana Negreiro\Dropbox\Classes\CS760\hw\hw3\hw3Data\emails.csv')

df['idx'] = df['Email No.'].str.split(' ').str[1]
df.set_index('idx', inplace=True) 
df = df.drop(columns=['Email No.'])

X = df.drop('Prediction', axis=1)
y = df['Prediction']
    
# separate training and testing sets (for a single split)
train_indices = np.concatenate((np.arange(0, 4001), np.arange(5000, 5000)))
test_indices = np.arange(4001, 5000)
        
X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

############################ KNN ###########################

#fit knn
y_pred, y_prob = knn_predict(X_train, y_train, X_test, 5)
    
# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Calculate ROC curve and AUC
knn_fpr, knn_tpr, _ = roc_curve(y_test, y_prob)
knn_roc_auc = auc(knn_fpr, knn_tpr)

##################### logistic regression #######################

# Set hyperparameters
learning_rate = 0.0001
iterations = 1000

#reshape
y_train = y_train.values
y_test = y_test.values
 
X_train = X_train.T
y_train = y_train.reshape(1, X_train.shape[1])
X_test = X_test.T
y_test = y_test.reshape(1, X_test.shape[1])
    
# call previously defined model
W, B, cost_list = model(X_train, y_train, learning_rate = learning_rate, iterations = iterations)

# Calculate predicted probabilities for the positive class
Z = np.dot(W.T, X_test) + B
A = sigmoid(Z)
y_scores = A

y_scores = y_scores.T
y_test = y_test.T

# Compute ROC curve and ROC AUC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)

#### combined plot ####

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve LR (AUC = {roc_auc:.2f})')
plt.plot(knn_fpr, knn_tpr, color='darkorange', lw=2, label=f'ROC curve KNN (AUC = {knn_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Logistic Regression')
plt.legend(loc='lower right')
plt.show()









