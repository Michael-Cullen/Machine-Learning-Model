import pandas as pd
from matplotlib import pyplot as plot
from sklearn import preprocessing, neural_network
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


fileName = "CMP3744M_ADM_Assignment 2-dataset-nuclear_plants.csv"
dataset = pd.read_csv(fileName)

def printStats(dataset, fileName):
    print("Data read from: %s\n"%fileName)
    
    print("Number of Data Features:%s\n"%len(dataset.columns))
    print("Number of Records:%s\n"%len(dataset.index))
    
    print("Feature\t\t\t  Mean")
    print(dataset.mean(axis=0))
    print()
    print("Feature\t\t\t  Std Dev")
    print(dataset.std(axis=0))
    print()
    print("Feature\t\t\t Min")
    print(dataset.min(axis=0,numeric_only=True))
    print()
    print("Feature\t\t\t Max")
    print(dataset.max(axis=0,numeric_only=True))

#------------------------Box Plot-------------------------------
def boxPlot(dataset):
    plotData1 = dataset[['Status','Power_range_sensor_1']]
    
    plotData1.boxplot(by='Status')
    plot.title("Boxplot of Power_range_sensor_1 sorted by Status")
    plot.suptitle("")
    
def boxPlot2(dataset):
    plotData1 = dataset[['Status','Pressure _sensor_1']]
    
    plotData1.boxplot(by='Status')
    plot.title("Boxplot of Pressure _sensor_1 sorted by Status")
    plot.suptitle("")

#------------------------Density Plot 1-------------------------------

def densPlot(dataset):
    plotData2 = dataset[['Status','Pressure _sensor_1']]
    gby = plotData2.groupby(['Status'],group_keys=False,as_index=False,sort=False)
    
    statusGroup = [gby.get_group(x) for x in gby.groups]
    
    normValues = ((statusGroup[0])['Pressure _sensor_1']).tolist()
    abnormValues = ((statusGroup[1])['Pressure _sensor_1']).tolist()
    
    df = pd.DataFrame({
            'Normal': normValues,
            'Abnormal': abnormValues
            })
    
    df.plot.kde()
    plot.title("Density plot of Pressure_sensor_1")


def densPlot2(dataset):
    plotData2 = dataset[['Status','Pressure _sensor_4']]
    gby = plotData2.groupby(['Status'],group_keys=False,as_index=False,sort=False)
    
    statusGroup = [gby.get_group(x) for x in gby.groups]
    
    normValues = ((statusGroup[0])['Pressure _sensor_4']).tolist()
    abnormValues = ((statusGroup[1])['Pressure _sensor_4']).tolist()
    
    df = pd.DataFrame({
            'Normal': normValues,
            'Abnormal': abnormValues
            })
    
    df.plot.kde()
    plot.title("Density plot of Pressure_sensor_4")


#-----------------------Data Standardisation-------------------------

noStatusData = dataset.drop("Status", axis=1)

scaler = StandardScaler()
std = scaler.fit_transform(noStatusData)
std2 = preprocessing.scale(noStatusData)

#print('Mean:', round(std[:,0].mean()))
#print('Standard deviation:', std[:,0].std())

#----------------------Data normalisation----------------------

normScaler = MinMaxScaler(feature_range=(0,1))
rescaledData = normScaler.fit_transform(noStatusData)

#--------------------ANN Classifier------------------------

classStatus = dataset["Status"]

x_train, x_test, y_train, y_test = train_test_split(std,classStatus, test_size= 0.1, random_state=27)

mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=8000, alpha=0.0001,
                     solver='adam', activation='logistic', verbose=False,tol=0.000000001,learning_rate_init=0.01,random_state=21)

mlp.fit(x_train, y_train)

y_pred = mlp.predict(x_test)

acc = accuracy_score(y_test, y_pred)

print("MLP Accuracy Score: ", acc)
#print("Report: ", classification_report(y_test,y_pred))

#--------------------Random forest Classifier------------------------

minLeaf = 10
rfc=RandomForestClassifier(n_estimators=100, min_samples_leaf = minLeaf, random_state=21)

rfc.fit(x_train,y_train)    

tree_y_pred = rfc.predict(x_test)

print("Tree Accuracy Score : ", accuracy_score(y_test,tree_y_pred))

#-----------------10 fold cross-validation-----------------------

def cv(model,x_train,y_train):
    if type(model) == neural_network.multilayer_perceptron.MLPClassifier:
        parameters = {'hidden_layer_sizes':[25,100,500]}
    else:
        parameters = {'n_estimators':[10,50,100]}
    
    gridsearch = GridSearchCV(model, parameters, cv=10, iid=False, return_train_score=True)
    
    gridsearch.fit(x_train,y_train)
    
    print(gridsearch.cv_results_['mean_test_score'])
    
    print(gridsearch.best_estimator_)
