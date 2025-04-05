# Hate Speech Detection using BERT

This project focuses on detecting hate speech in textual data using a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model. Hate speech detection is a critical task in natural language processing (NLP), especially for promoting healthier online communication and flagging harmful content.

## 🔍 Overview

The project implements a complete machine learning pipeline to:

- Preprocess and clean text data  
- Tokenize and encode input using the BERT tokenizer  
- Fine-tune a pretrained BERT model on a labeled hate speech dataset  
- Evaluate performance using standard classification metrics  

## 🚀 Technologies Used

- Python 3.x  
- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- PyTorch  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib  

## 🧠 Model

The model is based on the `bert-base-uncased` architecture and is fine-tuned for **binary classification**:  
- **Hate Speech**  
- **Not Hate Speech**

Using BERT embeddings allows for deep semantic understanding, improving accuracy over traditional models.

## 📊 Evaluation

The model's performance is evaluated using the following metrics:

- ✅ Accuracy  
- 🎯 Precision  
- 🔁 Recall  
- 📏 F1 Score  
- 📉 Confusion Matrix  

## 💻 How to Run the Project Locally

To run the project on your machine:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/moinuddin-ahamed/Hate-AI-Model
   cd Hate-AI-Model
   ```
