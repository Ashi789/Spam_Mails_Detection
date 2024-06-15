import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow informational messages

from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model('spam_detection_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile the model

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_spam(email_text):
    sequences = tokenizer.texts_to_sequences([email_text])
    data = pad_sequences(sequences, maxlen=100)
    prediction = model.predict(data)
    return 'spam' if prediction > 0.5 else 'ham'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_text = request.form['email_text']
        result = predict_spam(email_text)
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)