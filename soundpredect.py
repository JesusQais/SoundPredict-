import os
import numpy as np
import soundfile as sf
import librosa
import joblib
import sounddevice as sd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import time

"""
*******************************************************************************
                                Qais Ahmad
                      _             _                         _ 
         __ _  __ _(_)___    __ _| |__  _ __ ___   __ _  __| |
        / _` |/ _` | / __|  / _` | '_ \| '_ ` _ \ / _` |/ _` |
       | (_| | (_| | \__ \ | (_| | | | | | | | | | (_| | (_| |
        \__, |\__,_|_|___/  \__,_|_| |_|_| |_| |_|\__,_|\__,_|
           |_|                                                    
*******************************************************************************
"""
# Print the logo explicitly to ensure it appears in the console
logo = """
                      _             _                         _ 
         __ _  __ _(_)___    __ _| |__  _ __ ___   __ _  __| |
        / _` |/ _` | / __|  / _` | '_ \| '_ ` _ \ / _` |/ _` |
       | (_| | (_| | \__ \ | (_| | | | | | | | | | (_| | (_| |
        \__, |\__,_|_|___/  \__,_|_| |_|_| |_| |_|\__,_|\__,_|
           |_|                                                    
"""

print(logo)

# Constants
SAMPLING_RATE = 22050  # Set appropriate sample rate
DURATION = 10  # Duration for free recording mode (10 seconds)
SEGMENT_DURATION = 1  # Each segment duration (in seconds) for prediction
FILENAME = "audio_features.npy"
MODEL_FILENAME = "learning_model.pkl"
SAMPLES_DIR = "samples"

# Make sure the sample directory exists
if not os.path.exists(SAMPLES_DIR):
    os.makedirs(SAMPLES_DIR)

# Feature extraction function
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=SAMPLING_RATE)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCC features
    return np.mean(mfccs.T, axis=0)

# Record audio function
def record_audio(filename, duration=DURATION):
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(SAMPLING_RATE * duration), samplerate=SAMPLING_RATE, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    sf.write(filename, audio_data, SAMPLING_RATE)  # Save the recording as a .wav file
    print(f"Finished recording. Saved as {filename}")

# Collect audio samples for training
def collect_audio_samples():
    samples = []
    labels = []
    for label in ['a', 'b', 'c']:  # For this example, we record a, b, and c only
        filename = os.path.join(SAMPLES_DIR, f"{label}_sample.wav")
        record_audio(filename, duration=3)  # Record for 3 seconds for each character
        features = extract_features(filename)
        samples.append(features)
        labels.append(label)
    return np.array(samples), np.array(labels)

# Train the classifier using the recorded samples
def train_classifier():
    print("Training classifier...")
    samples, labels = collect_audio_samples()
    model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)  # Use SGD with SVM-like hinge loss
    label_encoder = LabelEncoder()  # Encode labels to integers
    labels_encoded = label_encoder.fit_transform(labels)
    model.fit(samples, labels_encoded)
    
    # Save the model and label encoder
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Classifier trained and saved.")

# Load the classifier (or use an incremental model if available)
def load_classifier():
    if os.path.exists(MODEL_FILENAME):
        model = joblib.load(MODEL_FILENAME)  # Load pre-trained model
        label_encoder = joblib.load('label_encoder.pkl')  # Load the label encoder
        print("Loaded pre-trained model.")
    else:
        # If no pre-trained model, use an SGD classifier for incremental learning
        model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)  # SVM-like model with incremental learning
        label_encoder = LabelEncoder()  # Empty label encoder for new training
        print("No pre-trained model found. Using incremental learning model.")
    return model, label_encoder

# Predict a given audio sample
def predict_audio(model, label_encoder, filename):
    features = extract_features(filename)
    prediction = model.predict([features])
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# Free record section to record anything and predict the character
def free_record_and_predict(model, label_encoder):
    print(f"\nFree recording mode activated. Please record for {DURATION} seconds, and the model will predict multiple keys.")
    filename = os.path.join(SAMPLES_DIR, "temp_sample.wav")
    record_audio(filename, duration=DURATION)
    
    # Split the recorded audio into segments for prediction
    y, sr = librosa.load(filename, sr=SAMPLING_RATE)
    segment_samples = int(SAMPLING_RATE * SEGMENT_DURATION)
    num_segments = len(y) // segment_samples

    print("Segmenting the recording into individual keypresses...")
    predictions = []
    segments_data = []  # To store segments data for learning
    for i in range(num_segments):
        start_sample = i * segment_samples
        end_sample = (i + 1) * segment_samples
        segment = y[start_sample:end_sample]
        
        # Save the segment as a temporary file
        temp_filename = os.path.join(SAMPLES_DIR, f"segment_{i}.wav")
        sf.write(temp_filename, segment, SAMPLING_RATE)
        
        # Predict for the segment
        prediction = predict_audio(model, label_encoder, temp_filename)
        predictions.append(prediction)
        print(f"Segment {i+1} prediction: {prediction}")
        
        segments_data.append((segment, prediction))  # Store segment data for possible learning
    
    # After prediction, allow the user to correct the predictions if needed
    print(f"Predicted sequence: {''.join(predictions)}")
    corrected_predictions = input("If any prediction is incorrect, please enter the correct sequence (e.g., 'abc'). Otherwise, press Enter to continue: ")
    
    if corrected_predictions:
        # Retrain the model with the corrected predictions
        print("Retraining the model with corrected predictions...")
        updated_samples = []
        updated_labels = []
        for (segment, correct_label) in segments_data:
            features = np.mean(librosa.feature.mfcc(y=segment, sr=SAMPLING_RATE, n_mfcc=13).T, axis=0)
            updated_samples.append(features)
            updated_labels.append(correct_label)
        
        # Encode the corrected labels
        updated_labels_encoded = label_encoder.transform(updated_labels)
        
        # Perform incremental learning
        model.partial_fit(updated_samples, updated_labels_encoded)
        print("Model updated with corrected data.")
        # Save the updated model
        joblib.dump(model, MODEL_FILENAME)
        joblib.dump(label_encoder, 'label_encoder.pkl')  # Save the updated label encoder
    
    print("Free prediction completed.")

# Main function with options to train or predict
def main():
    model, label_encoder = load_classifier()

    while True:
        print("\nSelect an option:")
        print("1. Train model (record a, b, c)")
        print("2. Predict a key (after training)")
        print("3. Free predict (record for 10 seconds and predict multiple keys)")
        print("4. Exit")
        choice = input("Enter choice (1/2/3/4): ")

        if choice == '1':
            train_classifier()
            model, label_encoder = load_classifier()  # Reload trained model after training
        elif choice == '2':
            if model:
                print("\nPlease record the key to predict (3 seconds max).")
                filename = os.path.join(SAMPLES_DIR, "temp_sample.wav")
                record_audio(filename, duration=3)
                prediction = predict_audio(model, label_encoder, filename)
                print(f"Prediction: {prediction}")
            else:
                print("No model found. Please train the model first.")
        elif choice == '3':
            if model:
                free_record_and_predict(model, label_encoder)
            else:
                print("No model found. Please train the model first.")
        elif choice == '4':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()

