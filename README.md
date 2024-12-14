# SoundPredict 🧠🎧

**SoundPredict** is an interactive machine learning tool designed to record and recognize audio features such as spoken characters using the microphone. It utilizes **Mel-frequency cepstral coefficients (MFCCs)** for feature extraction and a **Stochastic Gradient Descent (SGD) classifier** for predictions. This tool allows you to record a sequence of characters, train a model, and make predictions based on audio input.

## Features
- 📡 **Audio Recording**: Record audio samples for training or prediction.
- 🔍 **Feature Extraction**: Extract MFCCs from audio files to represent audio characteristics.
- 🤖 **Model Training**: Train a classifier to recognize audio features for specific characters (e.g., 'a', 'b', 'c').
- 🎤 **Free Prediction Mode**: Record for 10 seconds and make predictions for multiple characters.
- 🧠 **Incremental Learning**: Update and retrain the model with new predictions.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JesusQais/SoundPredict-.git

   cd soundpredict
   
   
2. Create a virtual environment (optional but recommended):

$ python3 -m venv venv
$ source venv/bin/activate  # For Windows: venv\Scripts\activate


3. Install the required dependencies:

pip install -r requirements.txt
Run the Tool 🎮

Start the tool by running the following command:


$ python soundpredict.py





You will be prompted with the following options:

1. Train model (record a, b, c)
2. Predict a key (after training)
3. Free predict (record for 10 seconds and predict multiple keys)
4. Exit 



💡 Example Usage
Train the model: Record samples for letters a, b, and c to train the classifier.
Make predictions: Record a short sample and let the tool predict the corresponding key.
Free recording: Record a free-form 10-second audio clip, and the model will predict a sequence of keys.


👨‍🏫 About the Creator

👋 Qais Ahmad

🛠️ Credits
Librosa 🎶 for audio feature extraction
Scikit-learn 📚 for machine learning
Sounddevice 🎧 for recording audio
Joblib 💾 for saving and loading the model


📄 License
This project is licensed under the MIT License - see the LICENSE file for details.






