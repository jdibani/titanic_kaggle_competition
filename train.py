from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix)

# Load data
data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')

#Preprocessing

# Sex
# Convert male/female into 0/1
data_train['Male'] = (data_train['Sex'] == 'male').astype(int)
data_train = data_train.drop(columns=['Sex'])
data_test['Male'] = (data_test['Sex'] == 'male').astype(int)
data_test = data_test.drop(columns=['Sex'])

# Name
# Gets title from Name, then groups rare titles into single group (Rare)
# One-hot encoding after, and drop Name

# This also didnt perform well (Think carefully about features before adding them)
"""data_train['Title'] = data_train['Name'].str.extract(r',\s*([^.]*)\.')[0]
data_test['Title'] = data_test['Name'].str.extract(r',\s*([^.]*)\.')[0]

title_map = {'Mlle': 'Miss','Ms': 'Miss','Mme': 'Mrs'}
data_train['Title'] = data_train['Title'].replace(title_map)
data_test['Title'] = data_test['Title'].replace(title_map)

common_titles = ['Mr', 'Mrs', 'Miss', 'Master']
data_train['Title'] = data_train['Title'].where(data_train['Title'].isin(common_titles),'Rare')
data_test['Title'] = data_test['Title'].where(data_test['Title'].isin(common_titles),'Rare')

data_train = pd.get_dummies(data_train,columns=['Title'],dtype=int)
data_test = pd.get_dummies(data_test,columns=['Title'],dtype=int)"""

data_train = data_train.drop(columns=['Name'])
data_test = data_test.drop(columns=['Name'])

# Embarked
# Convert Embarked to 3 different features
data_train['Embarked'] = data_train['Embarked'].fillna('Unknown')
data_test['Embarked'] = data_test['Embarked'].fillna('Unknown')
data_train = pd.get_dummies(data_train,columns=['Embarked'],dtype=int)
data_test = pd.get_dummies(data_test,columns=['Embarked'],dtype=int)

# Ticket
# Drop ticket for now, might add more later
data_train = data_train.drop(columns=['Ticket'])
data_test = data_test.drop(columns=['Ticket'])

# Age
# Replace missing Ages with median
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())
data_test['Age'] = data_test['Age'].fillna(data_train['Age'].median())

# Deck
# Extract deck from Cabin number
data_train['Deck'] = data_train['Cabin'].str[0]
data_train['Deck'] = data_train['Deck'].fillna('Unknown')
data_train = data_train.drop(columns=['Cabin'])
data_test['Deck'] = data_test['Cabin'].str[0]
data_test['Deck'] = data_test['Deck'].fillna('Unknown')
data_test = data_test.drop(columns=['Cabin'])

# Map deck level to a number
deck_map = {'A':8, 'B':7, 'C':6, 'D':5, 'E':4, 'F':3, 'G':2, 'T':1}
data_train['DeckLevel'] = data_train['Deck'].map(deck_map).fillna(0)
data_test['DeckLevel'] = data_test['Deck'].map(deck_map).fillna(0)
data_train = data_train.drop(columns=['Deck'])
data_test = data_test.drop(columns=['Deck'])

# PassengerId
# Remove PassengerId (model placed high importance on it)
test_ids = data_test['PassengerId']
data_train = data_train.drop(columns=['PassengerId'])
data_test = data_test.drop(columns=['PassengerId'])

"""
This currently made it perform worse, both with and without dropping 

# Parch & Sibsp
# Creates Familysize by combining Sibsp and Parch (+1 is person themselves)
data_train['FamilySize'] = (data_train['SibSp'] +data_train['Parch'] +1)
data_test['FamilySize'] = (data_test['SibSp'] +data_test['Parch'] +1)
data_train['IsAlone'] = (data_train['FamilySize'] == 1).astype(int)
data_test['IsAlone'] = (data_test['FamilySize'] == 1).astype(int)

data_train = data_train.drop(columns=['Parch'])
data_train = data_train.drop(columns=['SibSp'])
data_test = data_test.drop(columns=['Parch'])
data_test = data_test.drop(columns=['SibSp'])"""

# Survived
# Survived is predicted label
X = data_train.drop(columns=['Survived'])
Y = data_train['Survived']


print(data_train.head())
#print(data_test.head())

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Random Forest with best found parameters
rfc = RandomForestClassifier(n_estimators=200,max_depth=7,min_samples_split=10,random_state=42)
rfc.fit(X_train, y_train)

# Predict and evaluate
y_pred = rfc.predict(X_test)

print('accuracy score:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Feature importance
importance = pd.DataFrame({'Feature': X_train.columns,'Importance': rfc.feature_importances_})
print(importance.sort_values('Importance',ascending=False))

data_test = data_test.reindex(columns=X.columns, fill_value=0)

# Train final model on all training data
rfc_final = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    min_samples_split=10,
    random_state=42
)

rfc_final.fit(X, Y)

# Predict Kaggle test set
test_predictions = rfc_final.predict(data_test)

submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': test_predictions
})

submission.to_csv('submission.csv', index=False)