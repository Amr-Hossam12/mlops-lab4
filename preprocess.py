import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Step 1: Load raw data
url = "https://raw.githubusercontent.com/insaid2018/Term-2/master/Data/HR_comma_sep.csv"
print("HR Analytics dataset")
df = pd.read_csv(url)
print(f"Loaded {len(df)} rows")

# Step 2: Basic cleanup
df = df.dropna()
df = df.drop_duplicates()

# Target variable: 'left'
target_col = 'left'

# Split features/target
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Step 3: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: OneHotEncode categorical features 
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[cat_cols]))
X_test_encoded = pd.DataFrame(encoder.transform(X_test[cat_cols]))

# Reattach numeric features
X_train_final = pd.concat([X_train[num_cols].reset_index(drop=True), X_train_encoded], axis=1)
X_test_final = pd.concat([X_test[num_cols].reset_index(drop=True), X_test_encoded], axis=1)

# Reattach target
train_df = pd.concat([X_train_final, y_train.reset_index(drop=True)], axis=1)
test_df = pd.concat([X_test_final, y_test.reset_index(drop=True)], axis=1)

#Step 5: Save encoded CSVs
os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
df.to_csv("data/raw_data.csv", index=False)

print("Saved data/train_encoded.csv and data/test_encoded.csv")
