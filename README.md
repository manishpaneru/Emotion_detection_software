Emotion Recognition Using TensorFlow and Keras
This repository contains code for an emotion recognition model built with TensorFlow and Keras. The model classifies facial images into seven emotions:
Anger
Disgust
Fear
Happy
Neutral
Sadness
Surprise
Requirements
Python 3.6+


Usage
Dataset Preparation
Organize your dataset in the following structure:
plaintext
Copy code
dataset_emotion/
├── train/
│   ├── Anger/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Neutral/
│   ├── Sadness/
│   └── Surprise/
└── test/
    ├── Anger/
    ├── Disgust/
    ├── Fear/
    ├── Happy/
    ├── Neutral/
    ├── Sadness/
    └── Surprise/

Training the Model

python model.py

This will train the model and save it as emotion_classifier.h5.
Running the GUI Application
Launch the GUI to test the model with your own images:
python app.py

License
This project is licensed under the MIT License.

