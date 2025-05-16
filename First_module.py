import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def preprocess_data(df):
    # Convert categorical gender variable to numerical
    df['Sex'] = df['Sex'].map({'male':1, 'female':0})
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Embarked'], axis=1, inplace=True)
    return df

train_data = pd.read_csv('titanic_data/train.csv')
test_data = pd.read_csv('titanic_data/test.csv')
y_test_data = pd.read_csv('titanic_data/gender_submission.csv')
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
y_test = pd.DataFrame(y_test_data).drop(['PassengerId'], axis=1)

train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)


print(train_df.info())
print(test_df.info())
print(y_test.info())

# Select features and target variable
X_train = train_df[['Sex', 'Age', 'SibSp', 'Parch', 'Pclass', 'Fare']]
y_train = train_df['Survived']
X_test = test_df[['Sex', 'Age', 'SibSp', 'Parch', 'Pclass', 'Fare']]

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print (y_pred)
logreg_accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {logreg_accuracy:.2f}')
