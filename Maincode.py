def maincodepy():
        import os, time
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
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
        import nltk
        import re
        from nltk.corpus import stopwords
        import string
        # df = pd.read_csv("dataset.csv")

        data = pd.read_csv("dataset.csv")
        df=data
        # print(data.head())
        # textForWeb = textForWeb + '<p align="left">'+ str(data.head()) +'</p><br><p align="left">'

        # print(data.columns)
        textForWeb = textForWeb + '<p align="left">' + data.columns +'</p><br><p align="left">'
        data = data[["username", "tweet", "language"]]

        # print("Data Preprocesing")
        textForWeb = textForWeb + 'Data Preprocesing</p><br><p align="left">'
        data.isnull().sum()

        data["language"].value_counts()

        nltk.download('stopwords')
        stemmer = nltk.SnowballStemmer("english")
        stopword=set(stopwords.words('english'))

        def clean(text):                
                text = str(text).lower()
                text = re.sub('\[.*?\]', '', text)
                text = re.sub('https?://\S+|www\.\S+', '', text)
                text = re.sub('<.*?>+', '', text)
                text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
                text = re.sub('\n', '', text)
                text = re.sub('\w*\d\w*', '', text)
                text = [word for word in text.split(' ') if word not in stopword]
                text=" ".join(text)
                text = [stemmer.stem(word) for word in text.split(' ')]
                text=" ".join(text)
                return text
        data["tweet"] = data["tweet"].apply(clean)

        # text = " ".join(i for i in data.tweet)
        # stopwords = set(STOPWORDS)
        # wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
        # plt.figure( figsize=(15,10))
        # plt.imshow(wordcloud, interpolation='bilinear')
        # plt.axis("off")
        # plt.show()


        nltk.download('vader_lexicon')
        sentiments = SentimentIntensityAnalyzer()
        data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["tweet"]]
        data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["tweet"]]
        data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["tweet"]]
        data = data[["tweet", "Positive", "Negative", "Neutral"]]
        # print(data.head())
        # textForWeb = textForWeb + str(data.head()) + '</p><br><p align="left">'



        import tensorflow as tf
        import neural_structured_learning as nsl
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn import metrics

        df_train_X = data[['Positive','Negative','Neutral']]

        df_train_y= df[['video']]

        x_train,x_test,y_train,y_test = train_test_split(df_train_X,df_train_y,test_size = 0.25,random_state = 42)
        from sklearn.tree import DecisionTreeClassifier 
        dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
        dt.fit(x_train, y_train)
        dt_prediction=dt.predict(x_test)
        # print()
        # print("---------------------------------------------------------------------")
        textForWeb = textForWeb + "---------------------------------------------------------------------</p><br><p align=\"left\">"
        # print("Decision Tree")
        textForWeb = textForWeb + "Decision Tree</p><br><p align=\"left\">"
        # print()
        Result_2=accuracy_score(y_test, dt_prediction)*100
        # print(metrics.classification_report(y_test,dt_prediction))
        textForWeb = textForWeb + metrics.classification_report(y_test,dt_prediction) + '</p><br><p align="left">'
        # print()
        # print("DT Accuracy is:",Result_2,'%')
        textForWeb = textForWeb + "DT Accuracy is: "+str(Result_2)+'%'+'</p><br><p align="left">'
        # print()
        # print("Confusion Matrix:")
        textForWeb = textForWeb + "Confusion Matrix:</p><br><p align=\"left\">"
        from sklearn.metrics import confusion_matrix
        cm1=confusion_matrix(y_test, dt_prediction)
        # print(cm1)
        textForWeb = textForWeb + str(cm1)+'</p><br><p align="left">'
        # print("-------------------------------------------------------")
        textForWeb = textForWeb + "-------------------------------------------------------</p><br><p align=\"left\">"
        # print()
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.heatmap(cm1, annot = True, cmap ='plasma',
                        linecolor ='black', linewidths = 1)
        plt.title("Decision Tree")
        # plt.show()
        plt.savefig(r'{}'.format(path+'/static/img/plots/DecisionTree.png'))
        plots.append(r'{}'.format('/static/img/plots/DecisionTree.png'))
        time.sleep(1)
        plt.close()
        #ROC graph
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_test, dt_prediction)
        plt.plot(fpr, tpr, marker='.', label='DT')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.title("Decision Tree 2")
        # plt.show()
        plt.savefig(r'{}'.format(path+'/static/img/plots/DecisionTree-2.png'))
        plots.append(r'{}'.format('/static/img/plots/DecisionTree-2.png'))
        time.sleep(1)
        plt.close()

        from sklearn.ensemble import RandomForestClassifier

        rf= RandomForestClassifier(n_estimators = 100)  
        rf.fit(x_train, y_train)
        rf_prediction = rf.predict(x_test)
        Result_3=accuracy_score(y_test, rf_prediction)*100
        from sklearn.metrics import confusion_matrix

        # print()
        # print("---------------------------------------------------------------------")
        textForWeb = textForWeb + "---------------------------------------------------------------------</p><br><p align=\"left\">"
        # print("Random Forest")
        textForWeb = textForWeb + "Random Forest</p><br><p align=\"left\">"
        # print()
        # print(metrics.classification_report(y_test,rf_prediction))
        textForWeb = textForWeb + metrics.classification_report(y_test,rf_prediction) + '</p><br><p align="left">'
        # print()
        # print("Random Forest Accuracy is:",Result_3,'%')
        textForWeb = textForWeb + "Random Forest Accuracy is: "+str(Result_3)+'%'+'</p><br><p align="left">'
        # print()
        # print("Confusion Matrix:")
        textForWeb = textForWeb + "Confusion Matrix:</p><br><p align=\"left\">"
        cm2=confusion_matrix(y_test, rf_prediction)
        # print(cm2)
        textForWeb = textForWeb + str(cm2)+'</p><br><p align="left">'
        # print("-------------------------------------------------------")
        textForWeb = textForWeb + "-------------------------------------------------------</p><br><p align=\"left\">"
        # print()
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.heatmap(cm2, annot = True, cmap ='plasma',
                        linecolor ='black', linewidths = 1)
        plt.title("Confusion Matrix")
        # plt.show()
        plt.savefig(r'{}'.format(path+'/static/img/plots/ConfusionMatrix-2.png'))
        plots.append(r'{}'.format('/static/img/plots/ConfusionMatrix-2.png'))
        time.sleep(1)
        plt.close()
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_test, rf_prediction)
        plt.plot(fpr, tpr, marker='.', label='RF')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.title("KNN")
        # plt.show()
        plt.savefig(r'{}'.format(path+'/static/img/plots/RandomForest-2.png'))
        plots.append(r'{}'.format('/static/img/plots/RandomForest-2.png'))
        time.sleep(1)
        plt.close()

        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier()
        knn.fit(x_train, y_train)
        knn_prediction = knn.predict(x_test)
        Result_4=metrics.accuracy_score(y_test,knn_prediction)*100
        # print()
        # print("---------------------------------------------------------------------")
        textForWeb = textForWeb + "---------------------------------------------------------------------</p><br><p align=\"left\">"
        # print("KNN ")
        textForWeb = textForWeb + "KNN </p><br><p align=\"left\">"
        # print()
        # print("Knn Acuracy is :",Result_4,'%')
        textForWeb = textForWeb + "Knn Acuracy is : "+str(Result_4)+'%'+'</p><br><p align="left">'
        # print(metrics.classification_report(y_test , knn_prediction))
        textForWeb = textForWeb + metrics.classification_report(y_test , knn_prediction) + '</p><br><p align="left">'
        from sklearn.metrics import confusion_matrix
        # print("Confusion Matrix:")
        textForWeb = textForWeb + "Confusion Matrix:</p><br><p align=\"left\">"
        cm=confusion_matrix(y_test, knn_prediction)
        # print(cm)
        textForWeb = textForWeb + str(cm)+'</p><br><p align="left">'
        # print("-------------------------------------------------------")
        textForWeb = textForWeb + "-------------------------------------------------------</p><br><p align=\"left\">"
        # print()
        import matplotlib.pyplot as plt
        plt.title("Confusion Matrix")
        # plt.imshow(cm, cmap='binary')
        plt.imsave(r'{}'.format(path+'/static/img/plots/ConfusionMatrix.png'), cm, cmap='binary')
        plots.append(r'{}'.format('/static/img/plots/ConfusionMatrix.png'))
        time.sleep(1)
        plt.close()
        import seaborn as sns
        sns.heatmap(cm, annot = True, cmap ='plasma',
                        linecolor ='black', linewidths = 1)
        # plt.show()
        plt.title("Heat Map")
        plt.savefig(r'{}'.format(path+'/static/img/plots/HeatMap.png'))
        plots.append(r'{}'.format('/static/img/plots/HeatMap.png'))
        time.sleep(1)
        plt.close()
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_test, knn_prediction)
        plt.plot(fpr, tpr, marker='.', label='KNN')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        # plt.show()
        plt.title("KNN")
        plt.savefig(r'{}'.format(path+'/static/img/plots/knn.png'))
        plots.append(r'{}'.format('/static/img/plots/knn.png'))
        time.sleep(1)
        plt.close()

        from sklearn.svm import SVC
        svclassifier = SVC()
        svclassifier.fit(x_train,y_train)
        y_pred11 = svclassifier.predict(x_test)


        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        result = confusion_matrix(y_test, y_pred11)
        # print("Confusion Matrix:")
        textForWeb = textForWeb + "Confusion Matrix:</p><br><p align=\"left\">"
        # print(result)
        textForWeb = textForWeb + str(result)+'</p><br><p align="left">'
        result1 = classification_report(y_test, y_pred11)
        # print("Classification Report:",)
        textForWeb = textForWeb + "Classification Report:</p><br><p align=\"left\">"
        # print (result1)
        textForWeb = textForWeb + str(result1)+'</p><br><p align="left">'
        # print("Accuracy:",accuracy_score(y_test, y_pred11))
        textForWeb = textForWeb + "Accuracy: "+str(accuracy_score(y_test, y_pred11))+'</p><br>'

        return textForWeb, plots

# res = maincodepy()
# print(res)
# print('Done')