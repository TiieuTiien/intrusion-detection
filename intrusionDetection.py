import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def load_data(filename):
    csv_file = pd.read_csv(filename)

    # Encoding features (Features that is not numeric)
    label_encoder = LabelEncoder()
    clm=['protocol_type', 'service', 'flag']
    for x in clm:
        csv_file[x]=label_encoder.fit_transform(csv_file[x])
    csv_file['label'] = csv_file['label'].apply(lambda x: 1 if x == 'normal' else 0)

    features_df = csv_file.drop(['label'], axis=1)
    labels_df = csv_file['label']

    return features_df.values.tolist(), labels_df.values.tolist()

def train_model(selected_model):
    if selected_model == "LogisticRegression":
        model = LogisticRegression(max_iter=150 ,random_state=32)
    elif selected_model == "RandomForestClassifier":
        model = RandomForestClassifier(random_state=42)
    elif selected_model == "MLPClassifier":
        model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(300,), random_state=1)
    else:
        model = DecisionTreeClassifier()
    return model

def evaluate(labels, predictions, dataset, model_name):
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

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
    file_name = os.path.basename(dataset)  # Extract file name

    plt.savefig('result/'+file_name[:-10]+'_'+model_name+'.png')

    plt.show()

    return TPR, TNR
    # return TPR, TNR, precision, recall, f1

def training(selected_model, train_file_path, test_file_path):
    X_train, labels_train = load_data(train_file_path)
    X_test, labels_test = load_data(test_file_path)
    model = train_model(selected_model)
    model.fit(X_train, labels_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(labels_test, predictions, train_file_path, model.__class__.__name__)
    return sensitivity, specificity, labels_test, predictions