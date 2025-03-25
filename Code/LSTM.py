import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize # Import label_binarize

df=pd.read_csv("filled_data2.csv")
# Assuming df is your DataFrame
data = df[['CI No [or "T. No"]', 'Estimated Central Pressure (hPa) [or "E.C.P"]', 'Maximum Sustained Surface Wind (kt) ']]
target = df['Grade (text)']

# Normalize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Encode the target variable
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)
target_categorical = to_categorical(target_encoded)
n_classes = target_categorical.shape[1] # Get number of classes

# Function to create sequences based on window size
def create_sequences(data, target, window_size):
    sequences = []
    targets = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i + window_size])
        targets.append(target[i + window_size - 1])  # Target is the label at the end of the window
    return np.array(sequences), np.array(targets)

# Create datasets for different window sizes
window_sizes = [5, 10, 15, 20, 25, 30]
datasets = {}
history_data = {}

for window_size in window_sizes:
    X, y = create_sequences(data_scaled, target_categorical, window_size)
    datasets[window_size] = (X, y)

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Store results for each window size
for window_size in window_sizes:
    X, y = datasets[window_size]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape for RNN input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Build and train the model
    model = build_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])
    history_data[window_size] = history

    # Save training history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'history_window_size_{window_size}.csv', index=False)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Window Size: {window_size}, Test Loss: {loss}, Test Accuracy: {accuracy}')

    # Make predictions (for classification report and confusion matrix)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Print confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print(f'Confusion Matrix for Window Size {window_size}:\n{cm}')

    # Print classification report
    report = classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_)
    print(f'Classification Report for Window Size {window_size}:\n{report}')

    # Save classification report to a text file
    with open(f'classification_report_window_size_{window_size}.txt', 'w') as f:
        f.write(report)

    # Plotting the confusion matrix
    plt.figure(figsize=(12, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Window Size: {window_size})')
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
    plt.yticks(tick_marks, label_encoder.classes_)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Plotting accuracy and loss metrics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy Metrics (Window Size: {window_size})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Metrics (Window Size: {window_size})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- ROC Curve and AUC ---
    y_scores = model.predict(X_test) # Use model.predict instead of predict_proba
    y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes)) # Binarize y_test

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {label_encoder.classes_[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Window Size {window_size} (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # --- Precision-Recall Curve and Average Precision ---
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        precision, recall, thresholds = precision_recall_curve(y_test_binarized[:, i], y_scores[:, i])
        average_precision = average_precision_score(y_test_binarized[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {label_encoder.classes_[i]} (AP = {average_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for Window Size {window_size} (One-vs-Rest)')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()
