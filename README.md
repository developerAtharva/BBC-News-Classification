# 📰 BBC News Classification using Machine Learning

## 🔍 Overview
This project classifies BBC News articles into different categories using **Natural Language Processing (NLP)** and **Machine Learning**. The dataset consists of BBC news articles labeled into predefined categories. The model learns to classify articles based on textual features.  

## 🎯 Objective
- **Preprocess** and clean text data for better model accuracy.  
- **Train Machine Learning models** to classify news articles into categories.  
- **Evaluate model performance** using accuracy, precision, recall, and F1-score.  

## 📊 Dataset
The dataset consists of BBC News articles, each labeled into one of the following categories:  
- **Business** 🏢  
- **Entertainment** 🎭  
- **Politics** 🏛️  
- **Sport** ⚽  
- **Tech** 💻  

Each article contains raw text data that is processed to extract meaningful features for classification.  

## 🛠️ Technologies Used
- **Programming Language:** Python 🐍  
- **Libraries:** Pandas, NumPy, Scikit-Learn, NLTK, Matplotlib, Seaborn  
- **Machine Learning Models:**  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Random Forest  
  - Naïve Bayes (Best performing model)  

## ⚡ Steps Taken
1. **Data Preprocessing:**  
   - Removed stopwords and punctuation.  
   - Performed tokenization and lemmatization.  
   - Converted text into numerical format using TF-IDF Vectorization.  
2. **Model Training & Evaluation:**  
   - Experimented with multiple classification models.  
   - **Naïve Bayes** achieved the highest accuracy.  
3. **Performance Metrics:**  
   - Evaluated models using **accuracy**.  

## 🏆 Best Model: Naïve Bayes  
- Achieved the **highest accuracy** in news classification.  
- Works well for **text classification** tasks due to its probabilistic nature.  

## 🔥 How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/bbc-news-classification.git
   cd bbc-news-classification
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Execute the notebook cells to preprocess data, train models, and analyze results.  

## 📌 Future Improvements
- Implement **Deep Learning models (LSTMs, Transformers)** for improved accuracy.  
- Deploy the model using **Flask or Streamlit** for an interactive web app.  
- Expand dataset to cover more diverse news sources.  

## 📢 Contributing
Feel free to fork this repository and contribute! Open issues and pull requests are welcome. 😊  


---

⭐ **If you found this project useful, don't forget to star the repo!** ⭐
