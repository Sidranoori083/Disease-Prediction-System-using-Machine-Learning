from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# load models
knn, symptom_cols = pickle.load(open('models/knn_model.pkl', 'rb'))
svm, _            = pickle.load(open('models/svm_model.pkl', 'rb'))

@app.route('/')
def home():
    # clean up symptom names for display
    symptoms_display = [s.replace('_', ' ').title() for s in symptom_cols]
    symptoms = list(zip(symptom_cols, symptoms_display))
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected = data.get('symptoms', [])

    # build input vector
    input_vec = np.array(
        [1 if s in selected else 0 for s in symptom_cols]
    ).reshape(1, -1)

    knn_pred = knn.predict(input_vec)[0]
    svm_pred = svm.predict(input_vec)[0]
    knn_conf = round(float(max(knn.predict_proba(input_vec)[0])) * 100, 1)
    svm_conf = round(float(max(svm.predict_proba(input_vec)[0])) * 100, 1)

    return jsonify({
        'knn_pred': knn_pred,
        'svm_pred': svm_pred,
        'knn_conf': knn_conf,
        'svm_conf': svm_conf,
        'agree': knn_pred == svm_pred
    })

if __name__ == '__main__':
    app.run(debug=True)