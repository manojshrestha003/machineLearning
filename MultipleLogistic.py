import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
data = {
    "cgpa": [7.5, 8.0, 6.8, 7.2, 8.5, 9.0, 5.8, 6.5, 7.8, 8.2],
    "score": [75, 80, 65, 72, 85, 90, 58, 65, 78, 82],
    "placed": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]  # 1 for placed, 0 for not placed
}

dataset = pd.DataFrame(data)

sns.scatterplot(x ="cgpa",y="score",data=dataset,hue="placed")
plt.show()

x = dataset.drop(columns = "placed")
y = dataset["placed"]
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
lr = LogisticRegression()
lr.fit(x_train,y_train)
lr.score(x_test, y_test)
lr.predict([[7.5,75]])


plot_decision_regions(x.to_numpy(), y.to_numpy(), clf = lr)
plt.show()