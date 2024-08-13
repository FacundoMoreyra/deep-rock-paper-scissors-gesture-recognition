# Deep Rock Paper Scissors Gesture Recognition
In this repository you will find scripts to:
- Recognize hands using Google MediaPipe.
- Record a gesture dataset.
- Train a gesture classifier using Tensorflow and Keras.
- Implement a Rock Paper Scissors recognizer using Google MediaPipe and the trained model.

## Prerequisites
- Python 3.8.x or newer
- Python venv

## Preparation

### Fetch required resources
Create a virtual enviroment:
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## Rock Paper Scissors Gesture Recognition
With the venv opened, run:
```
python3 rock-paper-scissors.py
```

## Train your own model

### Record your dataset
With the venv opened, run:
```
python3 record-dataset.py
```

### Train the gesture classifier
With the venv opened, run:
```
python3 train-gesture-classifier.py
```