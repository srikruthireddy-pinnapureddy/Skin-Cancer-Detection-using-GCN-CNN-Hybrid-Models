from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

app = Flask(_name_)
model = load_model("C:\\Users\\srikr\\Downloads\\PROJECT\\PROJECT\\Alzheimers&Parkinson\\Alzheimers&Parkinson\\Alzheimers\\Project_model.h5")

class_names = ['Healthy', 'Alzheimers', 'Parkinsons']

@app.route('/')
def home():
    return render_template("home.html", img="/static/alp.jpg")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    new_data = pd.read_csv(file)
    X_new = new_data.drop(columns=['target'])  
    y_test = new_data['target']

    if len(new_data) == 1:
        prediction = model.predict(X_new)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return jsonify({
            'predicted_class': class_names[predicted_class],
            'graph3': '/static/training_validation_loss.png'
        })

    predictions = model.predict(X_new)
    predicted_class = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(y_test, predicted_class)
    precision = precision_score(y_test, predicted_class, average='weighted')
    recall = recall_score(y_test, predicted_class, average='weighted')
    f1 = f1_score(y_test, predicted_class, average='weighted')
    
    confusion = confusion_matrix(y_test, predicted_class)
    fig, ax = plt.subplots()
    sns.heatmap(confusion, annot=True, fmt='d', ax=ax, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    confusion_matrix_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    far_vs_frr_graph = generate_far_vs_frr_graph(y_test, predicted_class)
    patient_count_graph = generate_patient_count_graph(confusion)

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix_image,
        'graph1': far_vs_frr_graph,
        'graph2': patient_count_graph,
        'graph3': '/static/training_validation_loss.png'  # Reference to the saved graph image
    }
    
    return jsonify(result)

def generate_far_vs_frr_graph(y_true, y_pred):
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    far = fpr
    frr = 1 - tpr
    
    fig, ax = plt.subplots()
    ax.plot(thresholds, far, label='FAR')
    ax.plot(thresholds, frr, label='FRR')
    ax.set_xlabel('Thresholds')
    ax.set_ylabel('Rate')
    ax.set_title('FAR vs. FRR')
    ax.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return graph_image

def generate_patient_count_graph(confusion):
    correctly_classified_counts = np.diag(confusion)
    
    fig, ax = plt.subplots()
    counts = pd.Series(correctly_classified_counts, index=class_names)
    counts.plot(kind='bar', ax=ax, color=['blue', 'green', 'red'])
    
    for index, value in enumerate(counts):
        ax.text(index, value, str(value), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Number of Patients per Class')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return graph_image


app.run()