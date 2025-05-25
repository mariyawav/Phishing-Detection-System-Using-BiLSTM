import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle

# Load dataset
df = pd.read_csv("combined_dataset.csv")

# Assume the dataset has 'url' and 'label' columns
url = df['url'].astype(str).values
labels = df['label'].values

# Extract additional features
df['url_length'] = df['url'].apply(len)
df['digit_count'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x))
df['special_char_count'] = df['url'].apply(lambda x: sum(not c.isalnum() for c in x))

# Normalize features
scaler = MinMaxScaler()
features = scaler.fit_transform(df[['url_length', 'digit_count', 'special_char_count']])

# Change labels: 0 -> 'legitimate', 1 -> 'phishing'
label_mapping = {0: 'legitimate', 1: 'phishing'}
df['label'] = df['label'].map(label_mapping)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['label'])

# Tokenization and padding
MAX_VOCAB_SIZE = 50000  # Limit vocab size for efficiency
MAX_LENGTH = 100  # Fixed max sequence length

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(url)
sequences = tokenizer.texts_to_sequences(url)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')

# Train-test split
X_train_text, X_test_text, X_train_feat, X_test_feat, y_train, y_test = train_test_split(padded_sequences, features, labels, test_size=0.2, random_state=42)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))

# Model inputs
text_input = Input(shape=(MAX_LENGTH,), name="text_input")
feature_input = Input(shape=(3,), name="feature_input")

# Text feature extraction
embedding_layer = Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=128, input_length=MAX_LENGTH)(text_input)
lstm_layer = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)))(embedding_layer)
dropout_layer = Dropout(0.6)(lstm_layer)
lstm_layer2 = Bidirectional(LSTM(32, kernel_regularizer=l2(0.01)))(dropout_layer)

# Feature extraction
dense_feature = Dense(32, activation='relu')(feature_input)
dropout_feature = Dropout(0.6)(dense_feature)

# Concatenate features
concatenated = Concatenate()([lstm_layer2, dropout_feature])
dense_layer = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(concatenated)
dropout_final = Dropout(0.6)(dense_layer)
output_layer = Dense(1, activation='sigmoid')(dropout_final)

# Define model
model = Model(inputs=[text_input, feature_input], outputs=output_layer)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model
history = model.fit(
    [X_train_text, X_train_feat], y_train, 
    epochs=10, 
    batch_size=32, 
    validation_data=([X_test_text, X_test_feat], y_test), 
    class_weight=class_weights, 
    callbacks=[early_stopping]
)

# Evaluate
loss, accuracy = model.evaluate([X_test_text, X_test_feat], y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()

# Save the trained model
model.save("bilstm_url_model.h5")
print("✅ Model saved successfully as 'bilstm_url_model.h5'")

# Save tokenizer for future use
with open("tokenizer.pkl", "wb") as handle:

    pickle.dump(tokenizer, handle)
print("✅ Tokenizer saved successfully as 'tokenizer.pkl'")
