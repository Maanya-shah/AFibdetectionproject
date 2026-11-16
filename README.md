AFib Detection AI
Project that detects atrial fibrillation (AFib) from ECG recordings. It compares 5 models (Logistic Regression, SVM, Random Forest, XGBoost, and a CNN-LSTM deep learning model) and includes a web portal where you can upload an ECG and get a prediction.

## Requirements
- Python 3.11
- Git

## How to run it on your computer

1. Download the project:
   ```
   git clone https://github.com/Maanya-shah/AFibdetectionproject.git
   cd AFibdetectionproject
   ```

2. Create and turn on a virtual environment:
   ```
   python -m venv afib_env
   afib_env\Scripts\activate
   ```
   (On Mac: `source afib_env/bin/activate`)

3. Install the libraries:
   ```
   pip install -r requirements.txt
   ```

4. Start the web portal:
   ```
   python app.py
   ```

5. Open the link it prints (usually http://127.0.0.1:5000) in your browser.

## Re-running the notebook (optional)
The trained models are already in the `models/` folder, so the web app works without this step.
To retrain, unzip `data/training2017.zip` into `data/cinc2017/`, then open `afib_detection.ipynb`.

## Dataset
PhysioNet/Computing in Cardiology Challenge 2017 (8,528 single-lead ECG recordings).
