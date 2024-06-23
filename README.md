# Intrusion detection on KDD Cup 1999 and NSL-KDD dataset

Intrusion detection school project for network security subject using scikit-learn LogisticRegression, RandomForestClassifier, MLPClassifier, DecisionTreeClassifier models on KDD Cup 1999 and NSL-KDD dataset.<br>

## Set up
Download or clone this repository. You can download KDD Cup 1999 and NSL-KDD dataset in the [dataset](https://github.com/TiieuTiien/intrusion-detection/tree/dataset) branch or use your own dataset.<br>

### Create a virtual environment
Create a virtual environment
```Python
   python -m venv venv
```
Activate virtual environment
```python
   source venv/Scripts/activate
```
Install requirements to acquired the same environment and needed library
```Python
   pip install -r requirements.txt
```
### Execute ```main.py``` to start the app
```Python
   python main.py
```

## Data Format and Training Data
<i>The dataset was used has been modified from the original. In particular, it was added 1 line of header in each csv file so that pandas can read it. If you want to use the origin dataset you can create a column with code instead, [here](https://www.kaggle.com/code/timgoodfellow/nsl-kdd-explorations) is an example.</i>
One dataset will have two file (train and test).<br>

The KDD Cup 1999 has header attached to it and the 'normal.' from feature 'label' was changed to 'normal' to encoding feature for both KDD Cup 1999 and NSL-KDD dataset. 
The column 'attack' in the original NSL-KDD was change to label so that I don't have to change the feature since they refer to the same feature.

Click ```Data train``` to import train file, ```Data test``` to import text file, choose model and click ```Train``` to train model.

The result will be save in the ```result/``` folder.

## Load data
In ```load_data``` function
Encoding features that are not numeric
```python
    label_encoder = LabelEncoder()
    clm=['protocol_type', 'service', 'flag']
    for x in clm:
        csv_file[x]=label_encoder.fit_transform(csv_file[x])
    csv_file['label'] = csv_file['label'].apply(lambda x: 1 if x == 'normal' else 0)
```
Split the data frame into two data frames
```python
    features_df = csv_file.drop(['label'], axis=1)
    labels_df = csv_file['label']
```
## Evaluation
### Training
This 'detector' use four models from scikit-learn include: LogisticRegression, RandomForestClassifier, MLPClassifier, DecisionTreeClassifier. Train model based on the selected model and model fitting
```python
    model = train_model(selected_model)
    model.fit(X_train, labels_train)
```
### Evaluate  
Create predictions to evalute model
```python
    predictions = model.predict(X_test)
```
Evaluate mode base on TP (True Positive), TN (True Negative), FP (False Positive), FN (False Negative) generated. On this project we choose to evaluate TPR and TNR
```python
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    print(confusion_matrix(labels, predictions))
    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
```
Result with KDD Cup 1999 dataset<br>
<img src="https://github.com/TiieuTiien/intrusion-detection/assets/106142689/52bec40f-d4b1-45b4-802d-a29c70248786" alt="kddcup_LogisticRegression" width="200px">
<img src="https://github.com/TiieuTiien/intrusion-detection/assets/106142689/90db2bba-28b3-4fe6-adb1-37ee11bd820d" alt="kddcup_DecisionTreeClassifier" width="200px">
<img src="https://github.com/TiieuTiien/intrusion-detection/assets/106142689/ab9478ed-931d-4509-9720-5fca8e175ff4" alt="kddcup_RandomForestClassifier" width="200px">
<img src="https://github.com/TiieuTiien/intrusion-detection/assets/106142689/21e0cb74-99d6-4ffc-8bef-5bc07921d14b" alt="kddcup_MLPClassifier" width="200px"><br>

Result with NSL-KDD dataset<br>
<img src="https://github.com/TiieuTiien/intrusion-detection/assets/106142689/90b4b9a5-f05c-4cca-b883-3363ca27ef56" alt="nslkdd_LogisticRegression" width="200px">
<img src="https://github.com/TiieuTiien/intrusion-detection/assets/106142689/941b7551-ffcf-4565-9967-ce61c712dd37" alt="nslkdd_DecisionTreeClassifier" width="200px">
<img src="https://github.com/TiieuTiien/intrusion-detection/assets/106142689/34381f42-aeb1-4a22-a5e7-34cf0d3bf0be" alt="nslkdd_RandomForestClassifier" width="200px">
<img src="https://github.com/TiieuTiien/intrusion-detection/assets/106142689/712e82ee-28cb-4a74-8dd0-ab484478667e" alt="nslkdd_MLPClassifier" width="200px">

## Meta
P. Tien: phamquoctien1903@gmail.com

Distributed under the MIT license. See ```LICENSE``` for more information.

https://github.com/TiieuTiien/

## Reference
[NgocDung211/KDD_CUP1999](https://github.com/NgocDung211/KDD_CUP1999)<br>[Network Intrusion Detection using Python](https://www.kaggle.com/code/nidhirastogi/network-intrusion-detection-using-python)<br>[Intrusion Detection System [NSL-KDD]](https://www.kaggle.com/code/eneskosar19/intrusion-detection-system-nsl-kdd).<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
