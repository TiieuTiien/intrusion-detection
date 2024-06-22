import pandas as pd

# Load the KDD Cup dataset (replace 'kddcup.data_10_percent' with the actual file name)
data = pd.read_csv("KDDDataset.txt")

# Split into training and testing sets (80/20 split)
train_data = data.sample(frac=0.8, random_state=42)  # Randomly select 80%
test_data = data.drop(train_data.index)  # The remaining 20%

# Save as CSV files
train_data.to_csv("spliteddataset/nslkdd_train.csv", index=False)
test_data.to_csv("spliteddataset/nslkdd_test.csv", index=False)