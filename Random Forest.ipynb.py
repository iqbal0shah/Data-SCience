import pandas as pd

data=pd.read_csv("hd.csv")
data.head(5)
list(data.columns)

X=data.loc[:,['sqft_living','sqft_lot','sqft_above','sqft_basement']]
X.head(5)
type(X)
y=pd.DataFrame(data.iloc[:,2])

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=20,criterion="gini",random_state=1,max_depth=3)
classifier.fit(Xtrain,ytrain)

y_pred=classifier.predict(Xtest)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(accuracy_score(ytest,y_pred))

