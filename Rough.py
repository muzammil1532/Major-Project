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

print(train.loc[:,selected_features].dtypes)
"""train.drop(['id'],axis=1,inplace=True)

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
df=pd.DataFrame(selected_train)
for str in selected_features:
    print(str , " -->  " , df[str].min() , " --- ",df[str].max())
#print(selected_features)
#print(selected_train.dtypes)
#df=pd.DataFrame(selected_train)
#print(df.service.unique())"""