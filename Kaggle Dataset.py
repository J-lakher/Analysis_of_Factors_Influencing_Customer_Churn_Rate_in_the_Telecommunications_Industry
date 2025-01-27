import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
file_path = "C:/Users/Jagriti/OneDrive/Desktop/Dataset Analysis and Visualization Using Big Data Programs/Telco-Customer-Churn.csv"
data = pd.read_csv(file_path)

# Step 1: Display column names and datatypes
print("Dataset Information:")
print(data.info())

# Step 2: View the first few rows of the data
print("\nDataset Preview:")
print(data.head())

# Step 3: Check for missing values
print("\nMissing Values in Dataset:")
print(data.isnull().sum())

# Fill missing values (if any) with 0 (can be adjusted based on column relevance)
data.fillna(0, inplace=True)

# Step 4: Visualize the Churn Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Churn', data=data)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

# Step 5: Visualize Monthly Charges vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=data)
plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.show()

# Step 6: Encode Categorical Variables
print("\nEncoding Categorical Variables...")
data_encoded = pd.get_dummies(data, drop_first=True)

# Display encoded data columns
print("\nEncoded Dataset Preview:")
print(data_encoded.head())

# Step 7: Preprocess and Split the Dataset
X = data_encoded.drop(columns=['Churn_Yes'], errors='ignore')  # Independent variables
y = data_encoded['Churn_Yes']  # Target variable (binary: 1 for churn, 0 otherwise)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train a Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Step 9: Evaluate the Model
y_pred = log_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Step 10: Save Preprocessed Data for Tableau
# Ensure that the directory is valid and writable
try:
    # Save the preprocessed data to a simple directory
    preprocessed_file_path = "C:/Users/Jagriti/OneDrive/Desktop/Dataset Analysis and Visualization Using Big Data Programs/Preprocessed_Telco_Customer_Churn.csv"
    data_encoded.to_csv(preprocessed_file_path, index=False)
    print(f"\nPreprocessed dataset saved for Tableau at: {preprocessed_file_path}")
except Exception as e:
    print(f"\nError while saving the file: {e}")
