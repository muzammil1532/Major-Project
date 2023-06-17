
import pandas as pd  #Data manipulation and analysis

# Preprocessing purpose
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Splitting Data
from sklearn.model_selection import train_test_split

# For accuracy,Classification Report, Confusion Matrix
from sklearn import metrics

# For training different ML models
from sklearn import tree

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('E:/MajorProject/Code/DataSet1/UNSW_NB15_training-set.csv')

train.drop(['id'],axis=1,inplace=True)

scaler = StandardScaler()
# extract numerical attributes and scale it to have zero mean and unit variance  
cols = train.select_dtypes(include=['float64','int64']).columns
sc_train = scaler.fit_transform(train.select_dtypes(include=['float64','int64']))
sc_traindf = pd.DataFrame(sc_train, columns = cols)

LE = LabelEncoder()
# extract categorical attributes from both training and test sets
obj_train = train.select_dtypes(include=['object']).copy()
LE_obj_train = obj_train.apply(LE.fit_transform)
enctrain = LE_obj_train.drop(['attack_cat'], axis=1)

train_x = pd.concat([sc_traindf,enctrain],axis=1)
train_y = train['attack_cat']

selected_features=['dur',
 'sbytes',
 'dbytes',
 'rate',
 'sttl',
 'sload',
 'dload',
 'smean',
 'dmean',
 'ct_srv_src',
 'ct_dst_sport_ltm',
 'ct_dst_src_ltm',
 'ct_srv_dst',
 'label',
 'service']
selected_train = train_x.loc[:, selected_features]
print(selected_features)
print(selected_train.dtypes)
X_train,X_test,Y_train,Y_test = train_test_split(selected_train,train_y,train_size=0.70, random_state=2)
DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
DTC_Classifier.fit(X_train, Y_train)
models = []
models.append(('Decision Tree Classifier', DTC_Classifier))
for i, v in models:
    accuracy = metrics.accuracy_score(Y_train, v.predict(X_train))
    confusion_matrix = metrics.confusion_matrix(Y_train, v.predict(X_train))
    classification = metrics.classification_report(Y_train, v.predict(X_train))
    print()
    print('============================== {} Model Evaluation =============================='.format(i))
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()
for i, v in models:
    accuracy = metrics.accuracy_score(Y_test, v.predict(X_test))
    confusion_matrix = metrics.confusion_matrix(Y_test, v.predict(X_test))
    classification = metrics.classification_report(Y_test, v.predict(X_test))
    print()
    print('============================== {} Model Test Results =============================='.format(i))
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print() 
DTC_prediction_result = DTC_Classifier.predict(X_test)
for i in range(len(DTC_prediction_result)):
    print(DTC_prediction_result[i])

# saving model as a pickle
#import pickle
#pickle.dump(DTC_Classifier,open("Intrusion_model.sav", "wb"))
# pickle.dump(sc, open("scaler.sav", "wb"))"""