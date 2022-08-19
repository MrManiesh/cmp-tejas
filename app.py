import numpy as np
from flask import Flask, request, jsonify, render_template
import final, Maincode, extractive_summarization_bert
import pickle
import sklearn
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == "POST":
        if request.form['myButton'] == 'myButton1':
            runtimeResults = final.finalpy()[0]
            plot_imgs = final.finalpy()[1]
            # add img to html
            baseImgbody = ''
            for plot_img in plot_imgs:
                baseImgbody += "<img src='" + plot_img + "'>" + "<br>"
            return render_template('webpage.html', runtimeResults=runtimeResults, plot_img=baseImgbody)


        elif request.form['myButton'] == 'myButton2':
            runtimeResults = Maincode.maincodepy()[0]
            plot_imgs = Maincode.maincodepy()[1]
            # add img to html
            baseImgbody = ''
            for plot_img in plot_imgs:
                baseImgbody += "<img src='" + plot_img + "'>" + "<br>"
            return render_template('webpage.html', runtimeResults=runtimeResults, plot_img=baseImgbody)

            
        elif request.form['myButton'] == 'myButton3':
            runtimeResults = extractive_summarization_bert.extractiveSummarizationBert()[0]
            plot_imgs = extractive_summarization_bert.extractiveSummarizationBert()[1]
            # add img to html
            baseImgbody = ''
            for plot_img in plot_imgs:
                baseImgbody += "<img src='" + plot_img + "'>" + "<br>"
            return render_template('webpage.html', runtimeResults=runtimeResults, plot_img=baseImgbody)
        else:
            myButton = 'Something went wrong'
        
if __name__ == "__main__":
    app.run(debug=True)