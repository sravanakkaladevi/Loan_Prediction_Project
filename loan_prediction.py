import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("loan_data.csv")

# Fill missing values for numerical columns using the median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Fill missing values for categorical columns using the mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features and Target
X = df.drop("Loan_Status", axis=1)  # Assuming 'Loan_Status' is the target
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# Prediction
predictions = model.predict(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Visualization - Adjust sizes
fig, axs = plt.subplots(2, 2, figsize=(18, 10))  # Wider figure

# 1. Loan Status Distribution
sns.countplot(x="Loan_Status", data=df, ax=axs[0,0])
axs[0,0].set_title("Loan Status Distribution")

# 2. Applicant Income Distribution
sns.histplot(df["ApplicantIncome"], bins=30, kde=True, ax=axs[0,1])
axs[0,1].set_title("Applicant Income Distribution")

# 3. Correlation Heatmap (give more space, small font)
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axs[1,0],
            cbar=True, square=False, annot_kws={"size":6})
axs[1,0].set_title("Feature Correlation Heatmap", fontsize=12)
axs[1,0].set_xticklabels(axs[1,0].get_xticklabels(), rotation=45, ha='right', fontsize=7)
axs[1,0].set_yticklabels(axs[1,0].get_yticklabels(), rotation=0, fontsize=7)

# 4. Actual vs Predicted
axs[1,1].scatter(y_test, predictions, alpha=0.6)
axs[1,1].set_xlabel("Actual Loan Status")
axs[1,1].set_ylabel("Predicted Loan Status")
axs[1,1].set_title("Actual vs Predicted")

plt.tight_layout()
plt.show()
# Visualization - Adjust sizes