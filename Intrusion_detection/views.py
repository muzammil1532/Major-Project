from django.shortcuts import render
import pandas as pd  
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
# our home page view
def home(request):    
    return render(request, 'index.html')


# custom method for generating predictions
def getPredictions(dur,sbytes,dbytes,rate,sttl,sload,dload,smean,dmean,ct_srv_src,ct_dst_sport_ltm,ct_dst_src_ltm,ct_srv_dst,label,service):
    train = pd.read_csv('E:/MajorProject/Code/DataSet1/UNSW_NB15_training-set.csv')
    train.drop(['id'],axis=1,inplace=True)
    scaler = StandardScaler()  
    cols = train.select_dtypes(include=['float64','int64']).columns
    sc_train = scaler.fit_transform(train.select_dtypes(include=['float64','int64']))
    sc_traindf = pd.DataFrame(sc_train, columns = cols)
    LE = LabelEncoder()
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
    X_train,X_test,Y_train,Y_test = train_test_split(selected_train,train_y,train_size=0.70, random_state=2)
    DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
    DTC_Classifier.fit(X_train, Y_train)
    data={'dur':[dur],
        'sbytes' : [sbytes],
        'dbytes' : [dbytes],
        'rate': [rate],
        'sttl': [sttl],
        'sload': [sload],
        'dload': [dload],
        'smean': [smean],
        'dmean':  [dmean],
        'ct_srv_src': [ct_srv_src],
        'ct_dst_sport_ltm': [ct_dst_sport_ltm],
        'ct_dst_src_ltm': [ct_dst_src_ltm],
        'ct_srv_dst': [ct_srv_dst],
        'label': [label],
        'service':[service]
        }
    data=pd.DataFrame(data)
    res = DTC_Classifier.predict(data)
    return res
        

# our result page view
def result(request):
    dur = (request.GET['dur'])
    sbytes = (request.GET['sbytes'])
    dbytes = (request.GET['dbytes'])
    rate = (request.GET['rate'])
    sttl = (request.GET['sttl'])
    sload = (request.GET['sload'])
    dload = (request.GET['dload'])
    smean = (request.GET['smean'])
    dmean = (request.GET['dmean'])
    ct_srv_src = (request.GET['ct_srv_src'])
    ct_dst_sport_ltm = (request.GET['ct_dst_sport_ltm'])
    ct_dst_src_ltm = (request.GET['ct_dst_src_ltm'])
    ct_srv_dst = (request.GET['ct_srv_dst'])
    label = (request.GET['label'])
    service = (request.GET['service'])
    #print(dur,sbytes,dbytes,rate,sttl,sload,dload,smean,dmean,ct_srv_src,ct_dst_sport_ltm,ct_dst_src_ltm,ct_srv_dst,label,service)

    result = getPredictions(dur,sbytes,dbytes,rate,sttl,sload,dload,smean,dmean,ct_srv_src,ct_dst_sport_ltm,ct_dst_src_ltm,ct_srv_dst,label,service)

    return render(request, 'result.html', {'result':result})

 