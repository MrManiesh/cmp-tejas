
"Import Libaries "
from matplotlib.pyplot import text
import tensorflow as tf
import neural_structured_learning as nsl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import warnings
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import os, time
warnings.filterwarnings("ignore")


def finalpy():
        
        # delete all files in /static/img/plots
        path = os.getcwd()
        completePath = path + '/static/img/plots'
        os.chdir(r"{}".format(completePath))
        all_files = os.listdir()
        for f in all_files:
                os.remove(f)
        os.chdir(path)

        textForWeb = ''
        plots = []
        textForWeb = textForWeb + '<p align="left">==================================================</p><br><p align="left">'
        # print("==================================================")
        # print("Balancesheet Dataset")
        textForWeb = textForWeb + 'Balancesheet Dataset</p><br><p align="left">'
        # print(" Process - Balancesheet Dataset")
        textForWeb = textForWeb + ' Process - Balancesheet Dataset</p><br><p align="left">'
        # print("==================================================")
        textForWeb = textForWeb + '==================================================</p><br><p align="left">'


        ##1.data slection---------------------------------------------------
        #def main():
        dataframe=pd.read_csv('data.csv')
        dataframe=dataframe.iloc[::20]
        # print("---------------------------------------------")
        textForWeb = textForWeb + '---------------------------------------------</p><br><p align="left">'
        # print()
        # print("Data Selection")
        textForWeb = textForWeb + 'Data Selection</p><br><p align="left">'
        # print("Samples of our input data")
        textForWeb = textForWeb + 'Samples of our input data</p><br><p align="left">'
        # print(dataframe.head(10))
        # textForWeb = textForWeb + str(dataframe.head(10)) + '</p><br><p align="left">'
        # print("----------------------------------------------")
        textForWeb = textForWeb + '----------------------------------------------</p><br><p align="left">'
        # print()


        #2.pre processing--------------------------------------------------
        #checking  missing values 
        # print("---------------------------------------------")
        textForWeb = textForWeb + '---------------------------------------------</p><br><p align="left">'
        # print()
        # print("Before Handling Missing Values")
        textForWeb = textForWeb + 'Before Handling Missing Values</p><br><p align="left">'
        # print()
        tempVal = dataframe.isnull().sum()
        # print index only
        for i in range(0,len(tempVal)):
                textForWeb = textForWeb + str(tempVal.index[i]) + '    ' + str(tempVal.values[i]) + '</p><br><p align="left">'
                
        # print("----------------------------------------------")
        textForWeb = textForWeb + '----------------------------------------------</p><br><p align="left">'
        # print() 
        
        # print("-----------------------------------------------")
        textForWeb = textForWeb + '-----------------------------------------------</p><br><p align="left">'
        # print("After handling missing values")
        textForWeb = textForWeb + 'After handling missing values</p><br><p align="left">'
        # print()
        dataframe_2=dataframe.fillna(0)
        # print(dataframe_2.isnull().sum())
        tempVal = dataframe_2.isnull().sum()
        # print index only
        for i in range(0,len(tempVal)):
                textForWeb = textForWeb + str(tempVal.index[i]) + '    ' + str(tempVal.values[i]) + '</p><br><p align="left">'
        # print()
        # print("-----------------------------------------------")
        textForWeb = textForWeb + '-----------------------------------------------</p><br><p align="left">'
        

        #label encoding
        
        label_encoder = preprocessing.LabelEncoder() 
        # print("--------------------------------------------------")
        textForWeb = textForWeb + '--------------------------------------------------</p><br><p align="left">'
        # print("Before Label Handling ")
        textForWeb = textForWeb + 'Before Label Handling </p><br><p align="left">'
        # print()
        # print(dataframe_2.head(10))
        # textForWeb = textForWeb + str(dataframe_2.head(10)) + '</p><br><p align="left">'
        # print("--------------------------------------------------")
        textForWeb = textForWeb + '--------------------------------------------------</p><br><p align="left">'
        # print()

        #3.Data splitting--------------------------------------------------- 

        df_train_y=dataframe_2["method"]
        df_train_X=dataframe_2
        

        number = LabelEncoder()

        df_train_X['method'] = number.fit_transform(df_train_X['method'].astype(str))
        # print("==================================================")
        textForWeb = textForWeb + '==================================================</p><br><p align="left">'
        # print(" Preprocessing")
        textForWeb = textForWeb + 'Preprocessing</p><br><p align="left">'
        # print("==================================================")
        textForWeb = textForWeb + '==================================================</p><br><p align="left">'

        df_train_X.head(5)
        x=df_train_X
        #y=df_train_y
        

        ##4.feature selection------------------------------------------------
        ##kmeans
        

        x, y_true = make_blobs(n_samples=461, centers=2,cluster_std=0.30, random_state=0)
        plt.scatter(x[:, 0], x[:, 1], s=20);

        kmeans = KMeans(n_clusters=3)
        kmeans.fit(x)
        y_kmeans = kmeans.predict(x)

        plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=20, cmap='viridis')

        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

        plt.title("k-means")
        # plt.show()
        # Save the plot to /static/img/plots
        plt.savefig(r'{}'.format(path+'/static/img/plots/kmeans.png'))
        plots.append(r'{}'.format('/static/img/plots/kmeans.png'))
        time.sleep(1)
        plt.close()

        #---------------------------------------------------------------------------------------

        x_train,x_test,y_train,y_test = train_test_split(df_train_X,y_kmeans,test_size = 0.02,random_state = 42)

        
        rf= RandomForestClassifier(n_estimators = 100)  
        rf.fit(x_train, y_train)
        rf_prediction = rf.predict(x_test)
        Result_3=accuracy_score(y_test, rf_prediction)*100
        

        # print()
        # print("---------------------------------------------------------------------")
        textForWeb = textForWeb + '---------------------------------------------------------------------</p><br><p align="left">'
        # print("Random Forest")
        textForWeb = textForWeb + 'Random Forest</p><br><p align="left">'
        # print()
        # print(metrics.classification_report(y_test,rf_prediction))
        tempVal = metrics.classification_report(y_test,rf_prediction)

        textForWeb = textForWeb + str(metrics.classification_report(y_test,rf_prediction)) + '</p><br><p align="left">'
        # print()
        # print("Random Forest Accuracy is:",Result_3,'%')
        textForWeb = textForWeb + 'Random Forest Accuracy is:' + str(Result_3) + '%</p><br><p align="left">'
        # print()
        # print("Confusion Matrix:")
        textForWeb = textForWeb + 'Confusion Matrix:</p><br><p align="left">'
        cm2=confusion_matrix(y_test, rf_prediction)
        # print(cm2)
        textForWeb = textForWeb + str(cm2) + '</p><br><p align="left">'
        # print("-------------------------------------------------------")
        textForWeb = textForWeb + '-------------------------------------------------------</p><br><p align="left">'
        # print()
        

        sns.heatmap(cm2, annot = True, cmap ='plasma',
                linecolor ='black', linewidths = 1)
        # plt.show()
        plt.title("Random Forest")
        # Save the plot to /static/img/plots
        plt.savefig(r'{}'.format(path+'/static/img/plots/RandomForest.png'))
        plots.append(r'{}'.format('/static/img/plots/RandomForest.png'))
        time.sleep(1)
        plt.close()


        # plots.append(plt)

        # plots.append(plt)



        #---------------------------------------------------------------------------------------------


        
        dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
        dt.fit(x_train, y_train)
        dt_prediction=dt.predict(x_test)
        # print()
        # print("---------------------------------------------------------------------")
        textForWeb = textForWeb + '---------------------------------------------------------------------</p><br><p align="left">'
        # print("Decision Tree")
        textForWeb = textForWeb + 'Decision Tree</p><br><p align="left">'
        # print()
        Result_2=accuracy_score(y_test, dt_prediction)*100
        # print(metrics.classification_report(y_test,dt_prediction))
        textForWeb = textForWeb + str(metrics.classification_report(y_test,dt_prediction)) + '</p><br><p align="left">'
        # print()
        # print("DT Accuracy is:",Result_2,'%')
        textForWeb = textForWeb + 'DT Accuracy is:' + str(Result_2) + '%</p><br><p align="left">'
        # print()
        # print("Confusion Matrix:")
        textForWeb = textForWeb + 'Confusion Matrix:</p><br><p align="left">'
        
        cm1=confusion_matrix(y_test, dt_prediction)
        # print(cm1)
        textForWeb = textForWeb + str(cm1) + '</p><br><p align="left">'
        # print("-------------------------------------------------------")
        textForWeb = textForWeb + '-------------------------------------------------------</p><br><p align="left">'
        # print()
        

        sns.heatmap(cm1, annot = True, cmap ='plasma',
                linecolor ='black', linewidths = 1)
        # plt.show()
        plt.title("Decision Tree")
        plt.savefig(r'{}'.format(path+'/static/img/plots/DecisionTree.png'))
        plots.append(r'{}'.format('/static/img/plots/DecisionTree.png'))
        time.sleep(1)
        plt.close()
        # plots.append(plt.show())
        #ROC graph

        #------------------------------------------------------------------------------

        "Navie Bayies "
        
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(x_test)

        # Making the Confusion Matrix
        
        cm = confusion_matrix(y_test, y_pred)

        # print("Navie Bayies Accuracy is:",Result_2,'%')
        textForWeb = textForWeb + 'Navie Bayies Accuracy is:' + str(Result_2) + '%</p><br><p align="left">'
        # print()
        # print("Confusion Matrix:")
        textForWeb = textForWeb + 'Confusion Matrix:</p><br><p align="left">'
        
        cm1=confusion_matrix(y_test, y_pred)
        # print(cm1)
        textForWeb = textForWeb + str(cm1) + '</p><br><p align="left">'
        # print("-------------------------------------------------------")
        textForWeb = textForWeb + '-------------------------------------------------------</p><br>'
        # print()

        #---------------------------------------------------------------------------------------------
        return textForWeb, plots



# # call 
textForWeb = finalpy()
# print(textForWeb)