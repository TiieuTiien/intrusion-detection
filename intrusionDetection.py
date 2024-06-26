import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(dataset):
    df = pd.read_csv(dataset)

    label_n = []
    for i in df.label :
        if i == 'normal':
            label_n.append("normal")
        else:
            label_n.append("attack")

    df['label'] = label_n
    
    # Encoding features (Features that is not numeric)
    label_encoder = LabelEncoder()
    clm=['protocol_type', 'service', 'flag', 'label']
    for x in clm:
        df[x]=label_encoder.fit_transform(df[x])

    X = df.drop(["label"], axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,random_state=43)

    if os.path.basename(dataset) == 'kddcup1999.csv':
        columns=['service', 'flag', 'src_bytes', 'dst_bytes', 'logged_in', 'count',
        'serror_rate', 'srv_serror_rate', 'same_srv_rate', 'diff_srv_rate',
        'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate']
    elif os.path.basename(dataset) == 'nslkdd.csv':
        columns=['protocol_type', 'service', 'src_bytes', 'dst_bytes', 'logged_in',
        'count', 'srv_count', 'same_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate']
    else:
        print(f'{dataset} did not exist')
    
    #Continue model with top 15 features, because dataset is big enough
    X_train=X_train[columns]
    X_test=X_test[columns]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    # Use transform in order to prevent data leakage
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

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
    print(confusion_matrix(labels, predictions))
    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    # Bar chart
    bar_labels = ['Correct', 'Incorrect']
    width = 0.6
    colors = ['tab:green', 'tab:orange']
    counts = [tp, tn]

    plt.figure(num=model_name)
    p = plt.bar(bar_labels, counts, width, color=colors, label=None)
    plt.xlabel('Classification')
    plt.ylabel('Counts')
    plt.title('Correct and Incorrect Classifications')

    # Displaying true positive and true negative rates as text on the plot
    plt.bar_label(p, labels = [f'{100*TPR:.2f}%', f'{100*TNR:.2f}%'], label_type='center', color='white')
    plt.title('Correct and Incorrect Classifications')
    if not os.path.exists('result/'):
        os.mkdir('result/')
    file_name = os.path.basename(dataset)  # Extract file name
    plt.savefig('result/'+file_name[:-10]+'_'+model_name+'.png')
    
    plt.show()

    return TPR, TNR
    # return TPR, TNR, precision, recall, f1
    
# Helper function to evaluate if it's overfit or underfit.
def eval_metric(model, dataset):
    X_train, X_test, y_train, y_test = load_data(dataset)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    y_test_score = classification_report(y_test,y_pred,output_dict=True)
    y_train_score = classification_report(y_train,y_train_pred,output_dict=True)
    
    print("========================" + model.__class__.__name__ + "========================")
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test,y_pred,digits=4))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train,y_train_pred,digits=4))
    
    write_folder = "result/" + os.path.basename(dataset)[:-4] + "/"
    write_path = write_folder + os.path.basename(dataset)[:-4] + ".txt"
    if not os.path.exists(write_folder):
        os.mkdir(write_folder)
    with open(write_path, "a") as f:
        f.write("========================" + model.__class__.__name__ + "========================\n")
        f.write("Test_Set\n")
        f.write(str(confusion_matrix(y_test, y_pred)) + "\n")
        f.write(classification_report(y_test, y_pred, digits=4) + "\n\n")
        f.write("Train_Set\n")
        f.write(str(confusion_matrix(y_train, y_train_pred)) + "\n")
        f.write(classification_report(y_train, y_train_pred, digits=4) + "\n")
    
    return y_test_score['macro avg']

def evaluate(selected_model, dataset):
    model = selected_model
    model_name = model.__class__.__name__
    X_train, X_test, y_train, y_test = load_data(dataset)

    y_pred = model.predict(X_test)
    
    y_test_score = classification_report(y_test,y_pred,output_dict=True)
    
    print("========================" + model_name + "========================")
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test,y_pred,digits=4))

    fig, ax = plt.subplots()
    
    data = y_test_score['macro avg']
    data.pop('support', None)

    metrics_keys = data.keys()
    metrics_values = data.values()
    bar_colors = ['tab:red', 'tab:blue', 'tab:orange']

    rects = ax.bar(metrics_keys, metrics_values, label=metrics_keys, color=bar_colors)
    ax.bar_label(rects, labels=[f"{v:.4f}" for v in metrics_values], padding=3, rotation=90, label_type='center', color='white')

    ax.set_ylabel('Score')
    ax.legend(title='Metrics')

    ax.legend(ncols=1)
    
    plt.title(f'{model_name} on {os.path.basename(dataset)[:-4]}')
    
    # Show the chart
    plt.ylim(0, 1.4)
    plt.savefig("result/" + model_name + "_" + os.path.basename(dataset)[:-4] + ".png")
    plt.show()

def evaluate_all(models, dataset):
    # Open file in write mode
    # write_path = "/kaggle/working/" + os.path.basename(dataset)[:-4] + ".txt"
    # fname="/kaggle/working/" + os.path.basename(dataset)[:-4] + ".png"

    # if os.path.exists(write_path):
    #     os.remove(write_path)
    # if os.path.exists(fname):
    #     os.remove(fname)

    scores = {}
    for model in models:
        data = eval_metric(model, dataset)
        # Using list comprehension
        numeric_values = [value for value in data.values() if value != 'support']

        scores.update({model.__class__.__name__: numeric_values})

    # Extract specific test metrics
    metrics = ('Precision','Recall','F1-Score')
    model_name = ('DecisionTree\nClassifier','Logistic\nRegression','RandomForest\nClassifier','MLP\nClassifier')
    data_by_metrics = {'Precision': [], 'Recall': [], 'F1-Score': []}

    # Iterate through each metric (precision, recall, F1-score)
    i = 0
    for metric_name in data_by_metrics.keys():
        # Extract corresponding values from each model's list
        metric_values = [model_data[i] for model_name, model_data in scores.items()]
        # Append these values to the corresponding list in transformed_data
        data_by_metrics[metric_name].extend(metric_values)
        i+=1
        
    # Create the bar chart
    x = np.arange(len(model_name))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots()

    for attribute, measurement in data_by_metrics.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, labels=[f"{v:.4f}" for v in measurement], padding=3, rotation=90, label_type='center', color='white')
        multiplier += 1
        
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_xticks(x + width, model_name)
    ax.legend(ncols=1)
    
    plt.title(f'Overall performance on {os.path.basename(dataset)[:-4]}')
    
    # Show the chart
    plt.ylim(0, 1.4)
    plt.savefig("result/" + "eval_all_" + os.path.basename(dataset)[:-4] + ".png")
    plt.show()