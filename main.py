import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, 
                           roc_curve, auc, precision_recall_curve,
                           precision_score, recall_score, f1_score, accuracy_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================
# Custom Layer Definition
# ==============================================

class DistanceLayer(layers.Layer):
    """Custom layer to compute absolute difference between vectors"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        a, b = inputs
        return tf.abs(a - b)

    def get_config(self):
        return super().get_config()

# ==============================================
# Image Comparison Functions
# ==============================================

def highlight_differences(original_path, edited_path, output_path):
    """Highlight differences between two images using OpenCV"""
    original = cv2.imread(original_path)
    edited = cv2.imread(edited_path)
    
    if original is None or edited is None:
        print(f"Error loading images: {original_path} or {edited_path}")
        return None
        
    if original.shape != edited.shape:
        edited = cv2.resize(edited, (original.shape[1], original.shape[0]))
    
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    edited_gray = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(original_gray, edited_gray)
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    highlighted = edited.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(highlighted, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    cv2.imwrite(output_path, highlighted)
    print(f"Saved highlighted differences to {output_path}")
    return highlighted

# ==============================================
# Model Architecture
# ==============================================

def create_siamese_model(input_shape):
    """Create a Siamese neural network for image comparison"""
    def create_base_network():
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3)
        ])
        return model
    
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    base_network = create_base_network()
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = DistanceLayer()([processed_a, processed_b])
    prediction = layers.Dense(1, activation='sigmoid')(distance)
    
    return models.Model(inputs=[input_a, input_b], outputs=prediction)

# ==============================================
# Data Preparation
# ==============================================

def prepare_data(original_folder, edited_folder, img_size=(128, 128)):
    """Prepare training data with positive and negative pairs"""
    pairs = []
    labels = []
    
    original_files = sorted([f for f in os.listdir(original_folder) if f.startswith('original')])
    edited_files = sorted([f for f in os.listdir(edited_folder) if f.startswith('edited')])
    
    print(f"Found {len(original_files)} original and {len(edited_files)} edited images")
    
    # Create positive pairs (original vs edited)
    for orig_file, edit_file in zip(original_files, edited_files):
        orig_path = os.path.join(original_folder, orig_file)
        edit_path = os.path.join(edited_folder, edit_file)
        
        orig_img = cv2.imread(orig_path)
        edit_img = cv2.imread(edit_path)
        
        if orig_img is None or edit_img is None:
            continue
            
        orig_img = cv2.resize(orig_img, img_size) / 255.0
        edit_img = cv2.resize(edit_img, img_size) / 255.0
        
        pairs.append([orig_img, edit_img])
        labels.append(1)  # 1 means "edited"
    
    # Create negative pairs (original vs original)
    for i in range(len(original_files)-1):
        orig_path1 = os.path.join(original_folder, original_files[i])
        orig_path2 = os.path.join(original_folder, original_files[i+1])
        
        orig_img1 = cv2.imread(orig_path1)
        orig_img2 = cv2.imread(orig_path2)
        
        if orig_img1 is None or orig_img2 is None:
            continue
            
        orig_img1 = cv2.resize(orig_img1, img_size) / 255.0
        orig_img2 = cv2.resize(orig_img2, img_size) / 255.0
        
        pairs.append([orig_img1, orig_img2])
        labels.append(0)  # 0 means "same"
    
    print(f"Created {len(pairs)} image pairs ({sum(labels)} edited, {len(labels)-sum(labels)} original)")
    return np.array(pairs), np.array(labels)

# ==============================================
# Visualization Functions
# ==============================================

def plot_metrics(y_true, y_pred, y_prob):
    """Plot comprehensive evaluation metrics"""
    plt.figure(figsize=(15, 10))
    
    # Confusion Matrix
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Original', 'Edited'],
                yticklabels=['Original', 'Edited'])
    plt.title('Confusion Matrix')
    
    # ROC Curve
    plt.subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Precision-Recall Curve
    plt.subplot(2, 2, 3)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(recall, precision, color='blue', lw=2,
             label='Precision-Recall curve')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    # Metrics Bar Chart
    plt.subplot(2, 2, 4)
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'red', 'purple'])
    plt.title('Model Metrics')
    plt.ylim(0, 1)
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('model_metrics.png')
    plt.close()
    print("Saved model metrics visualization to model_metrics.png")

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("Saved training history visualization to training_history.png")

# ==============================================
# Model Training
# ==============================================

def train_model():
    """Train the Siamese network and save the model"""
    print("\nPreparing data...")
    pairs, labels = prepare_data('original', 'edited')
    if len(pairs) == 0:
        print("Error: No valid image pairs found for training")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.2, random_state=42)
    
    X_train_a = X_train[:, 0]
    X_train_b = X_train[:, 1]
    X_test_a = X_test[:, 0]
    X_test_b = X_test[:, 1]
    
    print("\nCreating model...")
    model = create_siamese_model((128, 128, 3))
    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
    
    print("\nTraining model for 30 epochs...")
    history = model.fit(
        [X_train_a, X_train_b], y_train,
        validation_data=([X_test_a, X_test_b], y_test),
        epochs=30,
        batch_size=32,
        verbose=1
    )
    
    print("\nSaving model...")
    model.save('image_comparison_model.keras')
    np.save('training_history.npy', history.history)
    
    print("\nGenerating visualizations...")
    plot_training_history(history)
    
    y_prob = model.predict([X_test_a, X_test_b]).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    plot_metrics(y_test, y_pred, y_prob)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Original', 'Edited']))
    
    return model

# ==============================================
# Image Comparison
# ==============================================

def compare_all_pairs(model):
    """Compare all available original/edited pairs"""
    original_files = sorted([f for f in os.listdir('original') if f.startswith('original')])
    edited_files = sorted([f for f in os.listdir('edited') if f.startswith('edited')])
    
    if len(original_files) != len(edited_files):
        print("Warning: Mismatch in number of original and edited files")
    
    for i in range(min(len(original_files), len(edited_files))):
        pair_num = i + 1  # Assuming files are named starting from 1
        original_path = os.path.join('original', f'original{pair_num}.jpeg')
        edited_path = os.path.join('edited', f'edited{pair_num}.jpeg')
        
        if not os.path.exists(original_path) or not os.path.exists(edited_path):
            print(f"Pair {pair_num} files not found")
            continue
        
        print(f"\n=== Comparing Pair {pair_num} ===")
        
        # Traditional method
        output_path = os.path.join('differences', f'highlighted_{pair_num}.jpeg')
        highlight_differences(original_path, edited_path, output_path)
        
        # ML method
        if model is not None:
            img_size = (128, 128)
            original_img = cv2.imread(original_path)
            edited_img = cv2.imread(edited_path)
            
            original_img = cv2.resize(original_img, img_size) / 255.0
            edited_img = cv2.resize(edited_img, img_size) / 255.0
            
            prediction = model.predict([np.array([original_img]), np.array([edited_img])])
            similarity = 1 - prediction[0][0]
            
            print(f"ML Similarity score: {similarity:.2%}")
            print("ML VERDICT: EDITED" if similarity < 0.5 else "ML VERDICT: ORIGINAL")

# ==============================================
# Main Execution
# ==============================================

def main():
    print("Starting image comparison system...")
    
    # Create output directories
    os.makedirs('differences', exist_ok=True)
    
    # Train or load model
    print("\nInitializing machine learning model...")
    model_path = 'image_comparison_model.keras'
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        try:
            model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={'DistanceLayer': DistanceLayer}
            )
            model.compile(optimizer='adam', 
                         loss='binary_crossentropy', 
                         metrics=['accuracy'])
            
            if os.path.exists('training_history.npy'):
                history = np.load('training_history.npy', allow_pickle=True).item()
                plot_training_history(history)
            
            # Generate metrics if test data exists
            pairs, labels = prepare_data('original', 'edited')
            if len(pairs) > 0:
                X_train, X_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.2, random_state=42)
                X_test_a = X_test[:, 0]
                X_test_b = X_test[:, 1]
                y_prob = model.predict([X_test_a, X_test_b]).ravel()
                y_pred = (y_prob > 0.5).astype(int)
                plot_metrics(y_test, y_pred, y_prob)
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred, target_names=['Original', 'Edited']))
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            model = None
    else:
        print("No existing model found. Training new model...")
        model = train_model()
    
    # Compare all available pairs
    print("\nComparing all image pairs:")
    compare_all_pairs(model)
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()