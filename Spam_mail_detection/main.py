import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report
import pickle
from tqdm import tqdm

print("Step 1: Loading the dataset")
# Load the dataset
df = pd.read_csv(r'C:\Users\Main Profile\Machine learning tasks\spam.csv', encoding='latin-1')

print("Step 2: Preprocessing the dataset")
# Drop unnecessary columns
df = df[['v1', 'v2']]

# Convert labels to binary format
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['v2'])
sequences = tokenizer.texts_to_sequences(df['v2'])
data = pad_sequences(sequences, maxlen=100)

print("Step 3: Splitting the data into training and test sets")
# Split the data
X_train, X_test, y_train, y_test = train_test_split(data, df['v1'], test_size=0.2, random_state=42)

print("Step 4: Building the model")
# Build the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print("Step 5: Training the model")
# Train the model
for epoch in tqdm(range(1, 11), desc="Training Epochs"):
    history = model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test))

print("Step 6: Evaluating the model")
# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred_classes))

print("Step 7: Saving the model and tokenizer")
# Save the model
model.save('spam_detection_model.h5')

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Step 8: Defining the function to check if an email is spam")


# Function to check if an email is spam
def check_email_spam():
    print("Loading the saved model and tokenizer")
    # Load the saved model
    model = load_model('spam_detection_model.h5')

    # Load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Get user input for the email text
    email_text = input("Enter the email text to check if it's spam: ")

    print("Processing the input email text")
    # Tokenize and pad the input email text
    sequences = tokenizer.texts_to_sequences([email_text])
    data = pad_sequences(sequences, maxlen=100)

    print("Predicting the result")
    # Predict if the email is spam or not
    prediction = model.predict(data)
    result = 'spam' if prediction > 0.5 else 'ham'
    print(f'The email is: {result}')


print("Step 9: Checking an example email")
# Example usage
check_email_spam()