# Readme
Here’s a clean and professional README.md for your Emotion-Based Courtroom Deception Detection project using the Real-Life Deception Detection (RLDD) 2016 dataset:

---

# 🎭 Emotion-Based Courtroom Deception Detection

An AI-powered multimodal system to detect deception in courtroom testimonies using emotion analysis from video, audio, and text. Built on the Real-Life Deception Detection (RLDD) 2016 dataset, this project focuses on violent crimes such as assault, homicide, and domestic abuse.

---

## 📂 Dataset

- Dataset: Real-Life Deception Detection (2016)
- Source: Real courtroom trial clips
- Modalities:
  - 🎥 Video clips of testimony
  - 🔊 Audio from the same clips
  - 📜 Transcribed text
  - 🏷️ Annotations (Truth/Lie labels)

You can find this dataset publicly available for research use.

---

## 🧠 Project Objective

Predict whether a person is telling the truth or lying by analyzing emotional cues across modalities. The key idea is:

> Extract and average emotion probabilities → Analyze emotional signatures → Predict deception (Truth vs Lie)

---

## 💼 Crime Focus

We align this work with two highly emotion-relevant crime categories:
- 🔪 Violent Crimes (e.g. murder, assault, domestic violence)
- ⚠️ Sexual Offenses (e.g. sexual assault, child abuse)

These scenarios typically involve heightened fear, guilt, anger, and sadness — ideal for emotional analysis.

---

## 🏗️ Project Structure

```bash
RLDD_EmotionDeception_Project/
├── data/
│   ├── clips/               # Video/audio clips
│   ├── transcripts/         # Courtroom testimony transcripts
│   └── annotations.csv      # Labels: Truth or Lie
├── features/
│   ├── text_emotions.csv    # Text-based emotion scores
│   ├── audio_features.csv   # Audio-based speech features
│   └── video_emotions.csv   # Visual emotion scores
├── models/
│   └── classifier.py        # Deception detection model
├── notebooks/
│   └── analysis.ipynb       # EDA and modeling experiments
├── utils/
│   └── preprocessing.py     # Feature extraction helpers
└── main.py                  # Main pipeline script
```

---

## 🛠️ Methodology

1. 🎯 Preprocessing
   - Align clips, transcripts, and labels
   - Clean and tokenize text

2. 🧪 Feature Extraction (Multimodal)
   - Text: GoEmotions BERT for emotional tone
   - Audio: Librosa/OpenSMILE for pitch, pauses, energy
   - Video: DeepFace or FER+ for facial emotion recognition

3. 🔍 Feature Fusion
   - Combine average emotion probabilities per clip
   - Merge with label data (Truth/Lie)

4. 🤖 Model Training
   - Classifier: Random Forest, XGBoost, or Logistic Regression
   - Evaluation: Accuracy, F1-score, Confusion Matrix

5. 📊 Visualization
   - Emotion heatmaps
   - Truth vs Lie emotional signature differences
   - Feature importance plots

---

## 🔍 Future Scope

- Temporal emotion shift tracking (e.g. frame-by-frame change)
- Use LSTM or Transformers on sequences
- Real-time deception probability output

---

## 📚 Dependencies

- Python 3.8+
- transformers, deepface, librosa, scikit-learn, pandas, numpy, matplotlib, seaborn, tqdm

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---





