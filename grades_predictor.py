import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load data
try:
    data = pd.read_csv("data/student-por.csv", sep=";")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: file not found. Please check the file path.")
    exit()

# 2. Encoding
data_encoded = data.copy()

for column in data_encoded.columns:
    if data_encoded[column].dtype == "object" or data_encoded[column].dtype == "string":
        le = LabelEncoder()
        data_encoded[column] = le.fit_transform(data_encoded[column])

# 3. Split features and target
# G3 is the final grade (target)
y = data_encoded["G3"]
X = data_encoded.drop("G3", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prediction & Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("-" * 30)
print(f"Mean Squared Error: {mse:.4f}")
print("-" * 30)
print("First 5 predictions vs Actual:")
comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred.round(2)})
print(comparison.head())
