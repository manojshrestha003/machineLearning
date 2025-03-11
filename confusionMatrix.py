import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  confusion_matrix,precision_score, recall_score, f1_score




data = {
    "cgpa": [7.5, 8.2, 6.8, 9.1, 5.5, 7.9, 6.3, 8.7, 7.1, 9.5],
    "score": [85, 78, 67, 92, 55, 80, 62, 88, 74, 95],
    "placed": [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1: Placed, 0: Not placed
}

dataset= pd.DataFrame(data)
dataset.head(3)

x= dataset.iloc[:,:-1]
y= dataset['placed']
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=2)

lr = LogisticRegression()
lr.fit(x_train,y_train)
lr.score(x_test, y_test)
 
#Creating confusion matrix
cf = confusion_matrix(y_test,lr.predict(x_test))

sns.heatmap(cf, annot=True)
plt.show()

precision_score(y_test,lr.predict(x_test))
recall_score(y_test,lr.predict(x_test))
f1_score(y_test,lr.predict(x_test))
