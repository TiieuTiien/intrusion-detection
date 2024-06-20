import csv
import pandas as pd
import sys
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from PIL import Image

def main():
    
    training("MLPClassifier", "kdd_data.csv", "kdd_test.csv")


def training(selected_model , train_file_path, test_file_path):
    if(selected_model is None):
        raise Exception("Data usage model")

    if(train_file_path is None):
        raise Exception("Data usage train file")

    if(test_file_path is None):
        raise Exception("Data usage test file")

    # Load data from spreadsheet and split into train and test sets
    X_train, labels_train = load_data(train_file_path)
    #scaler = preprocessing.StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    X_test, labels_test = load_data(test_file_path)
    #X_test = scaler.fit_transform(X_test)
    
    # Train model and make predictions
    model = train_model(selected_model)
    
    model.fit(X_train, labels_train)

    predictions = model.predict(X_test)
    
    sensitivity, specificity = evaluate(labels_test, predictions)

    # Print results
    print(f"Correct: {(labels_test == predictions).sum()}")
    print(f"Incorrect: {(labels_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"False Negative Rate: {(100 * (1-specificity)):.2f}%")
    
    return sensitivity , specificity, labels_test, predictions


def load_data(filename):

    print("This is the read data step")
    
    # [TODO] Chuẩn hóa dữ liệu (Reprocessing)

    # Read the CSV file
    csv_file = pd.read_csv(filename)
    
    # Convert labels to binary (1 for 'normal.' and 0 for others)
    csv_file['label'] = csv_file['label'].apply(lambda x: 1 if x == 'normal.' else 0)
    
    # Separate labels and features
    labels_df = csv_file['label']
    features_df = csv_file.drop(columns=['label'])
    
    # Apply label encoding to selected features
    label_encoder = LabelEncoder()
    features_df[['protocol_type', 'service', 'flag', 'is_host_login']] = features_df[['protocol_type', 'service', 'flag', 'is_host_login']].apply(label_encoder.fit_transform)
    
    # Convert DataFrames to lists
    evidence_list = features_df.values.tolist()
    labels_list = labels_df.values.tolist()
    
    return evidence_list, labels_list

def train_model(selected_model):

    print("this is train model step")

    # [TODO] Chỉnh các parameter

    # print(selected_model)
    if selected_model == "LogisticRegression":
        model = LogisticRegression(solver="saga")
    elif selected_model == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif selected_model == "MLPClassifier":
        model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                              hidden_layer_sizes=(100, 50), random_state=1)

    elif selected_model == "GaussianNB":
        model = GaussianNB()
    else:
        model = DecisionTreeClassifier()


    return model


def evaluate(labels, predictions):
    print("This is the evaluate step")

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    # True Positive Rate (Recall)
    TPR = tp / (tp + fn)

    # True Negative Rate
    TNR = tn / (tn + fp)

    # Precision
    precision = precision_score(labels, predictions)

    # # Recall
    # recall = recall_score(labels, predictions)

    # # F1-score
    # f1 = f1_score(labels, predictions)

    # Displaying metrics
    # print(f'True Positive Rate (Recall): {100*TPR:.2f}%')
    # print(f'True Negative Rate: {100*TNR:.2f}%')
    # print(f'Precision: {100*precision:.2f}%')
    # print(f'Recall: {100*recall:.2f}%')
    # print(f'F1-score: {100*f1:.2f}%')

    # Bar chart
    labels = ['Correct', 'Incorrect']
    counts = [tp, tn]
    colors = ['green', 'red']

    plt.bar(labels, counts, color=colors)
    plt.xlabel('Classification')
    plt.ylabel('Count')
    plt.title('Correct and Incorrect Classifications')

    # Displaying true positive and true negative rates as text on the plot
    plt.text(0, tp + 1000, f'True Positive Rate: {100*TPR:.2f}%', ha='center', va='center', color='blue')
    plt.text(1, tn + 1000, f'True Negative Rate: {100*TNR:.2f}%', ha='center', va='center', color='blue')
    plt.savefig('correct_img.png')

    plt.show()

    return TPR, TNR

if __name__ == "__main__":
    main()
