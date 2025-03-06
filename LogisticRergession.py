import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Creating dataFrame

data = {
    "Age": [22, 25, 47, 52, 46, 56, 27, 48, 36, 29, 
            50, 40, 23, 60, 33, 49, 38, 55, 41, 30],
    "EstimatedSalary": [25000, 32000, 70000, 85000, 60000, 95000, 45000, 72000, 58000, 41000, 
                        90000, 62000, 27000, 99000, 50000, 73000, 57000, 91000, 63000, 44000],
    "Purchased": [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 
                  1, 0, 0, 1, 0, 1, 0, 1, 0, 0]  # 0 = Not Purchased, 1 = Purchased
}

# Convert to DataFrame
dataset = pd.DataFrame(data)

# Drop EstimatedSalary column
dataset.drop(columns=['EstimatedSalary'], inplace=True)

#scatterplot
sns.scatterplot(x='Age', y='Purchased', data=dataset)
plt.show()

#Saperate dependent and independent variable 
x = dataset[["Age"]]
y = dataset["Purchased"]
#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#train  model

lr = LogisticRegression()
lr.fit(x_train, y_train)

#check accuracy
lr.score(x_test, y_test)*100

#predict outcomes
lr.predict([[41]])


#plotting the graph
sns.scatterplot(x='Age', y='Purchased', data=dataset)
sns.lineplot(x = "Age", y = lr.predict(x), data = dataset, color = "red")
plt.show()