from flask import Flask,request,render_template, jsonify
import pandas as pd
import numpy as np
from generate import generate

app = Flask(__name__)
cols = ['composer', 'n_output']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    composer, n_output = [x for x in request.form.values()]
    n_output = int(n_output)
    l_out_midi = generate(composer, n_output)
    return render_template(
        'generated.html',
        gen1 = l_out_midi[0],
        gen2 = l_out_midi[1],
        gen3 = l_out_midi[2],
        gen4 = l_out_midi[4])
    

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data = request.get_json(force=True)
#     data_unseen = pd.DataFrame([data])
#     prediction = predict_model(model, data=data_unseen)
#     output = prediction.Label[0]
#     return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)