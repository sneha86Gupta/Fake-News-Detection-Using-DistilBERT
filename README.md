# Fake-News-Detection-Using-DistilBERT
This project aims to classify news articles as FAKE or REAL using both traditional machine learning algorithms and deep learning with DistilBERT. We compare multiple models, analyze performance, and visualize results to identify the best-performing approach for fake news detection.
The project experiments with:  
- **Traditional ML models** (Logistic Regression, SVM, Random Forest, XGBoost, Decision Tree)  
- **Transformer-based deep learning model** (DistilBERT fine-tuned for text classification)  

It also provides detailed evaluation with **confusion matrices, accuracy comparisons, and visualizations** to understand performance.  

---

## 🎯 Objectives  
1. Preprocess the dataset and prepare it for classification.  
2. Train and compare different ML models.  
3. Fine-tune DistilBERT for fake news detection.  
4. Evaluate models with metrics such as Accuracy, Precision, Recall, F1-score.  
5. Visualize results using plots and confusion matrices.  

---
## 📂 Dataset  
- The dataset used contains **news articles** labeled as:  
  - `REAL` → Genuine news content  
  - `FAKE` → False or misleading content  
- After preprocessing, the dataset includes cleaned **text features** and corresponding **labels**.  

### Preprocessing Steps:  
- Removal of stopwords, punctuation, and special characters.  
- Tokenization and lowercasing.  
- Label encoding (`FAKE = 0`, `REAL = 1`).  
- Train-test split for model training.  

---
## 🛠️ Tech Stack  

### **Languages**  
- Python 3.8+  

### **Libraries & Frameworks**  
- **Data Handling & Processing**  
  - `pandas`, `numpy`  

- **Text Preprocessing**  
  - `nltk` (stopwords, lemmatization, tokenization)  
  - `re`, `string` (text cleaning)  

- **Classical ML Models**  
  - `scikit-learn` (`LogisticRegression`, `LinearSVC`, `RandomForestClassifier`, `DecisionTreeClassifier`, `TfidfVectorizer`)  
  - `xgboost`  

- **Deep Learning (Transformers)**  
  - `torch` (PyTorch)  
  - `transformers` (Hugging Face: DistilBERT, tokenizer, scheduler)  

- **Visualization**  
  - `matplotlib`, `seaborn`  

---
## 🧠 Models Implemented  

### 🔹 Classical ML Models  
- **Logistic Regression**  
- **Support Vector Machine (SVM)**  
- **Random Forest Classifier**  
- **Decision Tree Classifier**  
- **XGBoost Classifier**  

### 🔹 Transformer Model  
- **DistilBERT** (fine-tuned on our dataset)  
  - Tokenization handled by Hugging Face’s `DistilBertTokenizer`.  
  - Pre-trained model loaded using `DistilBertForSequenceClassification`.  
  - Fine-tuned with CrossEntropyLoss and AdamW optimizer.  
  - Trained on GPU (if available).  

---

## 📊 Results  

### **Classical ML Models**  
| Model               | Accuracy |
|----------------------|----------|
| Logistic Regression  | 82.31%   |
| Linear SVM           | 81.76%   |
| Random Forest        | 84.56%   |
| XGBoost              | 83.76%   |
| Decision Tree        | 81.42%   |

### **DistilBERT (Fine-tuned)**  
- Accuracy: **87.74%**  
- Precision/Recall (REAL): **0.91 / 0.91**  
- Precision/Recall (FAKE): **0.78 / 0.78**  

✅ DistilBERT clearly outperforms traditional ML models.  

---

## 📈 Visualizations  

1. **Training Loss Curve**  
   Shows how loss decreases across epochs while training DistilBERT.  

2. **Accuracy Comparison Chart**  
   Comparison of all ML models vs DistilBERT.  

3. **Confusion Matrix**  
   Helps visualize misclassifications for FAKE vs REAL news.
   ---
   👩‍💻 Author

Sneha Gupta
📧 Contact: [snehagupta8611@gmail.com]
🌐 GitHub: : https://github.com/sneha86Gupta
