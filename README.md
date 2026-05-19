# 🎯 Zero-Point Sentiment

> Fine-grained sentiment analysis that goes beyond positive/negative — predicting a **numerical satisfaction score (1–10)** from raw user feedback using NLP and Machine Learning.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red?logo=streamlit)](https://streamlit.io/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-lightgrey.svg)](LICENSE)

---

## 📖 Overview

Traditional sentiment analysis classifies text into broad buckets — **positive**, **negative**, or **neutral**. **Zero-Point Sentiment** takes a more nuanced approach: it transforms raw textual feedback into a **continuous satisfaction score from 1 to 10**, capturing the true intensity of user opinion.

This system ingests multiple raw datasets, preprocesses them independently, merges and explores the combined data, trains ML regression/classification models, and exposes predictions through a deployable web application.

---

## 🗂️ Project Structure

```
zero_point_sentiment/
│
├── datasets/                          # Raw and processed data
│   ├── dataset1.csv                   # Source dataset 1 (e.g. Amazon reviews)
│   ├── dataset2.csv                   # Source dataset 2 (e.g. Yelp reviews)
│   ├── dataset1_cleaned.csv           # Cleaned version of dataset 1
│   ├── dataset2_cleaned.csv           # Cleaned version of dataset 2
│   └── merged_dataset.csv             # Final merged & normalized dataset
│
├── notebooks/                         # Step-by-step Jupyter notebooks
│   ├── 01_dataset1_preprocessing.ipynb
│   ├── 02_dataset2_preprocessing.ipynb
│   ├── 03_merged_preprocessing_eda.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── models/                            # Saved trained model artifacts
│   ├── tfidf_vectorizer.pkl
│   └── sentiment_model.pkl
│
├── app/                               # Deployment app (Streamlit)
│   ├── app.py                         # Main Streamlit application
│   └── predictor.py                   # Inference logic / helper functions
│
├── viuals/                            # Plots and visualizations
│   ├── score_distribution.png
│   ├── wordcloud_positive.png
│   ├── wordcloud_negative.png
│   └── model_performance.png
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🔄 End-to-End Pipeline

```
Dataset 1 ──┐
            ├─► Individual Preprocessing ─► Merge ─► EDA ─► Model Training ─► Deployment
Dataset 2 ──┘
```

---

## 📦 Step 1 — Individual Dataset Preprocessing

Each dataset is cleaned independently before merging, to preserve source-specific quirks and ensure quality control.

### 📁 `notebooks/01_dataset1_preprocessing.ipynb`

**Source:** e.g. Amazon Product Reviews / Course Feedback CSV

```python
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# --- Load ---
df1 = pd.read_csv('datasets/dataset1.csv')
print(df1.shape)         # Inspect dimensions
df1.head()               # Preview first rows

# --- Inspect ---
df1.info()
df1.isnull().sum()       # Count missing values
df1['rating'].value_counts()   # Class balance check

# --- Drop nulls & duplicates ---
df1.dropna(subset=['review_text', 'rating'], inplace=True)
df1.drop_duplicates(subset='review_text', inplace=True)

# --- Normalize rating to 1–10 scale ---
# (if source uses 1–5, scale up)
df1['score'] = df1['rating'].apply(lambda x: round(x * 2))

# --- Text Cleaning ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()                          # Lowercase
    text = re.sub(r'<.*?>', '', text)                 # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)        # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)              # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()          # Normalize whitespace
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df1['clean_text'] = df1['review_text'].apply(clean_text)

# --- Remove very short reviews ---
df1 = df1[df1['clean_text'].str.split().str.len() >= 3]

# --- Keep relevant columns ---
df1 = df1[['clean_text', 'score']]
df1['source'] = 'dataset1'

df1.to_csv('datasets/dataset1_cleaned.csv', index=False)
print(f"Dataset 1 cleaned: {df1.shape}")
```

**Output:** `datasets/dataset1_cleaned.csv` — clean text + normalized score (1–10) + source label.

---

### 📁 `notebooks/02_dataset2_preprocessing.ipynb`

**Source:** e.g. Yelp Reviews / App Store Reviews

```python
# --- Load ---
df2 = pd.read_csv('datasets/dataset2.csv')
df2.info()
df2.isnull().sum()

# --- Handle missing values ---
df2['review_text'].fillna('', inplace=True)
df2.dropna(subset=['stars'], inplace=True)

# --- Normalize score (Yelp uses 1–5 stars → map to 1–10) ---
df2['score'] = df2['stars'].apply(lambda x: round(x * 2))

# --- Apply same cleaning pipeline ---
df2['clean_text'] = df2['review_text'].apply(clean_text)
df2 = df2[df2['clean_text'].str.split().str.len() >= 3]

# --- Keep relevant columns ---
df2 = df2[['clean_text', 'score']]
df2['source'] = 'dataset2'

df2.to_csv('datasets/dataset2_cleaned.csv', index=False)
print(f"Dataset 2 cleaned: {df2.shape}")
```

**Output:** `datasets/dataset2_cleaned.csv`

---

## 🔗 Step 2 — Merge + Preprocessing + EDA

### 📁 `notebooks/03_merged_preprocessing_eda.ipynb`

#### 2a. Merge Datasets

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

df1 = pd.read_csv('datasets/dataset1_cleaned.csv')
df2 = pd.read_csv('datasets/dataset2_cleaned.csv')

# --- Merge ---
df = pd.concat([df1, df2], ignore_index=True)
print(f"Merged dataset shape: {df.shape}")

# --- Shuffle ---
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Final null/duplicate check ---
df.dropna(inplace=True)
df.drop_duplicates(subset='clean_text', inplace=True)
print(f"After dedup: {df.shape}")

df.to_csv('datasets/merged_dataset.csv', index=False)
```

#### 2b. Exploratory Data Analysis (EDA)

```python
# --- Score Distribution ---
plt.figure(figsize=(10, 5))
sns.countplot(x='score', data=df, palette='coolwarm')
plt.title('Satisfaction Score Distribution (1–10)')
plt.xlabel('Score')
plt.ylabel('Count')
plt.savefig('viuals/score_distribution.png', dpi=150)
plt.show()

# --- Source-wise breakdown ---
df.groupby(['source', 'score']).size().unstack().plot(
    kind='bar', figsize=(12, 5), colormap='tab10'
)
plt.title('Score Distribution by Source Dataset')
plt.savefig('viuals/score_by_source.png', dpi=150)
plt.show()

# --- Review length analysis ---
df['text_length'] = df['clean_text'].str.split().str.len()
print(df['text_length'].describe())

plt.figure(figsize=(10, 4))
sns.histplot(df['text_length'], bins=50, kde=True, color='steelblue')
plt.title('Distribution of Review Word Count')
plt.xlabel('Word Count')
plt.savefig('viuals/review_length_dist.png', dpi=150)
plt.show()

# --- WordCloud: High satisfaction (score >= 8) ---
high_text = ' '.join(df[df['score'] >= 8]['clean_text'])
wc_high = WordCloud(width=800, height=400, background_color='white',
                    colormap='Greens').generate(high_text)
plt.figure(figsize=(12, 5))
plt.imshow(wc_high, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words — High Satisfaction (Score 8–10)')
plt.savefig('viuals/wordcloud_positive.png', dpi=150)
plt.show()

# --- WordCloud: Low satisfaction (score <= 3) ---
low_text = ' '.join(df[df['score'] <= 3]['clean_text'])
wc_low = WordCloud(width=800, height=400, background_color='white',
                   colormap='Reds').generate(low_text)
plt.figure(figsize=(12, 5))
plt.imshow(wc_low, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words — Low Satisfaction (Score 1–3)')
plt.savefig('viuals/wordcloud_negative.png', dpi=150)
plt.show()

# --- Correlation: Review length vs Score ---
plt.figure(figsize=(8, 4))
sns.boxplot(x='score', y='text_length', data=df, palette='muted')
plt.title('Review Length vs Satisfaction Score')
plt.savefig('viuals/length_vs_score.png', dpi=150)
plt.show()
```

**Key EDA Findings:**
- Score distribution is typically skewed toward higher ratings (5–8 range).
- High-satisfaction reviews frequently contain words like *excellent*, *amazing*, *helpful*, *recommend*.
- Low-satisfaction reviews frequently contain words like *worst*, *terrible*, *broken*, *refund*.
- Longer reviews tend to appear at score extremes (very satisfied or very dissatisfied).

---

## 🤖 Step 3 — Model Training

### 📁 `notebooks/04_model_training.ipynb`

```python
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load Merged Data ---
df = pd.read_csv('datasets/merged_dataset.csv')
X = df['clean_text']
y = df['score']

# --- Train / Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# --- TF-IDF Vectorization ---
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),      # Unigrams + bigrams
    sublinear_tf=True,       # Apply log normalization
    min_df=3,                # Ignore very rare terms
    max_df=0.9               # Ignore very common terms
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# Save vectorizer
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# --- Model Comparison ---
models = {
    'Ridge Regression':  Ridge(alpha=1.0),
    'SVR (RBF)':         SVR(kernel='rbf', C=1.0, epsilon=0.2),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
    'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = np.clip(np.round(model.predict(X_test_tfidf)), 1, 10)
    mae  = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2   = r2_score(y_test, preds)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
    print(f"{name:25s} → MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

# --- Hyperparameter Tuning on Best Model ---
param_grid = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5,
                           scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)
print(f"Best alpha: {grid_search.best_params_['alpha']}")

final_model = grid_search.best_estimator_
final_model.fit(X_train_tfidf, y_train)

# Save final model
with open('models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("Model saved to models/sentiment_model.pkl")
```

---

## 📊 Step 4 — Model Evaluation

### 📁 `notebooks/05_model_evaluation.ipynb`

```python
import pickle, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

# --- Load saved model + vectorizer ---
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- Predictions ---
preds = np.clip(np.round(model.predict(X_test_tfidf)), 1, 10)

# --- Metrics ---
print(f"MAE  : {mean_absolute_error(y_test, preds):.4f}")
print(f"RMSE : {mean_squared_error(y_test, preds)**0.5:.4f}")
print(f"R²   : {r2_score(y_test, preds):.4f}")

# --- Actual vs Predicted scatter ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, preds, alpha=0.3, color='steelblue')
plt.plot([1, 10], [1, 10], 'r--', linewidth=2)
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title('Actual vs Predicted Satisfaction Score')
plt.savefig('viuals/model_performance.png', dpi=150)
plt.show()

# --- Confusion Matrix ---
cm = confusion_matrix(y_test.astype(int), preds.astype(int), labels=list(range(1, 11)))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(1, 11), yticklabels=range(1, 11))
plt.xlabel('Predicted Score')
plt.ylabel('Actual Score')
plt.title('Confusion Matrix — Satisfaction Score Prediction')
plt.savefig('viuals/confusion_matrix.png', dpi=150)
plt.show()

# --- Sample Predictions ---
samples = [
    "Absolutely fantastic experience, would highly recommend!",
    "It was okay, nothing special but not bad either.",
    "Terrible service. Completely disappointed and frustrated.",
]
for text in samples:
    vec = tfidf.transform([text])
    score = int(np.clip(round(model.predict(vec)[0]), 1, 10))
    print(f"Score: {score}/10 → \"{text}\"")
```

**Sample output:**
```
Score: 9/10 → "Absolutely fantastic experience, would highly recommend!"
Score: 5/10 → "It was okay, nothing special but not bad either."
Score: 2/10 → "Terrible service. Completely disappointed and frustrated."
```

---

## 🚀 Step 5 — Deployment (Streamlit App)

### 📁 `app/predictor.py`

```python
import pickle
import numpy as np

def load_artifacts():
    with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('../models/sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return tfidf, model

def predict_score(text: str, tfidf, model) -> int:
    vec = tfidf.transform([text])
    score = model.predict(vec)[0]
    return int(np.clip(round(score), 1, 10))
```

### 📁 `app/app.py`

```python
import streamlit as st
from predictor import load_artifacts, predict_score

st.set_page_config(page_title="Zero-Point Sentiment", page_icon="🎯", layout="centered")

@st.cache_resource
def get_model():
    return load_artifacts()

tfidf, model = get_model()

st.title("🎯 Zero-Point Sentiment Analyzer")
st.markdown("Enter any user feedback and get a **satisfaction score from 1 to 10**.")

user_input = st.text_area("✍️ Paste your feedback here:", height=150,
                           placeholder="e.g. The product quality was outstanding...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        score = predict_score(user_input, tfidf, model)
        if score >= 8:
            st.success(f"😊 Satisfaction Score: **{score} / 10** — Highly Positive")
        elif score >= 5:
            st.warning(f"😐 Satisfaction Score: **{score} / 10** — Neutral / Mixed")
        else:
            st.error(f"😞 Satisfaction Score: **{score} / 10** — Negative")
        st.progress(score / 10)
    else:
        st.warning("Please enter some feedback text.")

st.markdown("---")
st.caption("Built with Python · scikit-learn · NLTK · Streamlit")
```

### Running Locally

```bash
git clone https://github.com/solid-metal/zero_point_sentiment.git
cd zero_point_sentiment
pip install -r requirements.txt
streamlit run app/app.py
```

Open **http://localhost:8501** in your browser.

### Deploying to Streamlit Cloud

1. Push your repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New App** → select your repo → set main file to `app/app.py`.
4. Click **Deploy** — your app is live in minutes with a public URL.

---

## 🛠️ Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.8+ |
| ML Framework | scikit-learn |
| NLP | NLTK, TF-IDF Vectorizer |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Experimentation | Jupyter Notebook |
| Deployment | Streamlit |

---

## ⚙️ Installation

```bash
git clone https://github.com/solid-metal/zero_point_sentiment.git
cd zero_point_sentiment
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
wordcloud
streamlit
jupyter
```

---

## 📄 License

Licensed under the **GNU Affero General Public License v3.0**. See [LICENSE](LICENSE) for details.

---

## 👤 Authors

**Abhinav Singh** — [GitHub](https://github.com/Anthony-3000)
**Saksham** — [GitHub](https://github.com/SakshamG-014)
**Anurag Vaibhav** — [GitHub](https://github.com/solid-metal)


---

## 🌟 Acknowledgements

- [scikit-learn](https://scikit-learn.org/) for the ML toolkit
- [NLTK](https://www.nltk.org/) for NLP preprocessing
- [Streamlit](https://streamlit.io/) for rapid deployment
- Open-source review datasets from Yelp, Amazon, and similar platforms
