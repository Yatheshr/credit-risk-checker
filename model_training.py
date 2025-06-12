import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Dummy dataset
np.random.seed(0)
df = pd.DataFrame({
    'age': np.random.randint(20, 70, 500),
    'income': np.random.randint(20000, 150000, 500),
    'credit_score': np.random.randint(300, 850, 500),
    'loan_amount': np.random.randint(1000, 50000, 500),
    'loan_term': np.random.choice([12, 24, 36, 60], 500),
    'num_dependents': np.random.randint(0, 4, 500),
    'default': np.random.choice([0, 1], 500, p=[0.7, 0.3])  # 0 = no default, 1 = default
})

X = df.drop('default', axis=1)
y = df['default']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression model with more iterations
model = LogisticRegression(max_iter=500)
model.fit(X_scaled, y)

# Save the trained model
with open(r'C:\Users\yrangeg\Downloads\credit_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the scaler (important for use in the Streamlit app)
with open(r'C:\Users\yrangeg\Downloads\scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")
