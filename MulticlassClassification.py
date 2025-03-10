import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = {
    'Age': [25, 32, 40, 22, 29, 35, 45, 50, 27, 31],
    'Income': [30000, 45000, 60000, 25000, 40000, 52000, 70000, 80000, 32000, 48000],
    'Education_Level': ['Bachelor', 'Master', 'PhD', 'High School', 'Bachelor', 'Master', 'PhD', 'PhD', 'High School', 'Bachelor'],
    'Job_Type': ['Engineer', 'Doctor', 'Professor', 'Technician', 'Manager', 'Lawyer', 'Scientist', 'Entrepreneur', 'Technician', 'Engineer'],
    'Credit_Score': [700, 750, 800, 650, 720, 770, 810, 850, 690, 730],
    'Target': ['Low', 'Medium', 'High', 'Low', 'Medium', 'Medium', 'High', 'High', 'Low', 'Medium']  # Classification target
}


df = pd.DataFrame(data)

# Encode 
label_enc = LabelEncoder()
df['Education_Level'] = label_enc.fit_transform(df['Education_Level'])
df['Job_Type'] = label_enc.fit_transform(df['Job_Type'])
df['Target'] = label_enc.fit_transform(df['Target'])

# Define features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='lbfgs', max_iter=200)  
model.fit(X_train, y_train)
#Test accuracy
model.score(X_test, y_test)
#predict model
model.predict([[50,100000,3,3,1000]])