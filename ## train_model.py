# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Sample dataset
data = pd.DataFrame({
    "tenure": [1, 5, 12, 24, 36],
    "monthly_charges": [20, 50, 70, 80, 100],
    "total_charges": [20, 250, 840, 1920, 3600],
    "churn": [1, 0, 0, 0, 1]
})

X = data[["tenure", "monthly_charges", "total_charges"]]
y = data["churn"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
pickle.dump(model, open("churn_model.pkl", "wb"))