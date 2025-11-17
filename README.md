# 📘 Sentiment Analysis with DistilBERT  
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface)]()
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20Accelerated-blue?logo=kaggle)]()
[![Python](https://img.shields.io/badge/Python-3.10-green?logo=python)]()
[![Model](https://img.shields.io/badge/Model-DistilBERT-yellow)]()

📘 Sentiment Analysis with DistilBERT — FastAPI Web App

This project demonstrates a complete end-to-end NLP pipeline using HuggingFace Transformers, DistilBERT, and FastAPI.
It includes:

✔ Dataset loading

✔ Tokenization

✔ Fine-tuning DistilBERT

✔ Saving the model

✔ Building a modern Bootstrap UI

✔ Deploying an API for real-time sentiment prediction

Perfect for beginners, AI/ML engineers, and portfolio projects.

🚀 Features
🔍 1. Sentiment Classification

Supports 2-class (Positive/Negative)

Supports 3-class (Positive/Neutral/Negative)

Uses DistilBERT, a light & fast Transformer model

⚙️ 2. FastAPI Web Server

/predict → JSON sentiment prediction API

/ui → Modern Bootstrap UI for user input

Colored results + emojis 🙂 😡 😐

🎨 3. Clean UI

Built with Bootstrap 5, includes:

Centered card layout

Professional design

Mobile-friendly

📦 4. Easy-to-Train

Just run:

python train.py


Model is saved to:

./results/

📁 Project Structure
📦 DistilBERT_Sentiment_Repo
│
├── app.py                # FastAPI server with Bootstrap UI
├── train.py              # Training script (fine-tunes DistilBERT)
├── results/              # Saved model + tokenizer
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── training_args.bin
│
├── train.csv             # Training dataset (your file)
├── test.csv              # Testing dataset (your file)
│
├── README.md             # Documentation
└── requirements.txt      # Python dependencies

📦 Installation
1️⃣ Clone the repo
git clone https://github.com/yourusername/DistilBERT_Sentiment_Repo.git
cd DistilBERT_Sentiment_Repo

2️⃣ Create virtual env
python3 -m venv .venv
source .venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

🧠 Training the Model

Your dataset should look like:

train.csv / test.csv
text,label
"I love this product!",1
"This is terrible.",0

Run training:
python train.py


What happens:

Tokenizer loads

Dataset is tokenized

DistilBERT is fine-tuned

Metrics (Accuracy, F1) are computed

Model is saved into ./results/

🌐 Running the Web App

Start the FastAPI server:

uvicorn app:app --reload


Now visit:

🎨 UI
http://127.0.0.1:8000/ui

🧪 API (JSON)
http://127.0.0.1:8000/docs

✨ UI Preview (Description)

Input box for text

Bootstrap card layout

Color-coded results:

Green = Positive 🙂

Red = Negative 😡

Orange = Neutral 😐

🧪 Example Predictions
Text	Output
“I love this!”	POSITIVE 🙂
“This is the worst.”	NEGATIVE 😡
“It works.”	NEUTRAL 😐
📈 Model Performance

After training you will see:

Epoch 1/2 – Accuracy: 0.89, F1: 0.88
Epoch 2/2 – Accuracy: 0.92, F1: 0.91

🛠 Customization

You can modify:

Learning rate

Batch size

Number of labels

Model architecture

Or replace DistilBERT with:

BERT-base

RoBERTa

DeBERTa

ALBERT

📤 Deployment Options

You can deploy this app on:

🔹 HuggingFace Spaces (Free)

Supports Gradio & FastAPI

🔹 AWS EC2

Production + scaling

🔹 Docker
docker build -t sentiment-app .
docker run -p 8000:8000 sentiment-app

❤️ Credits

Built using:

HuggingFace Transformers

FastAPI

Bootstrap

PyTorch

⭐ Contribute

Pull requests welcome!
You can:

Improve UI

Add datasets

Add multi-language support

Add ONNX optimization

🎯 This Project Is Perfect For:

ML Portfolio

Job Applications

Learning Transformers

Understanding NLP pipelines

Real-time prediction apps


---

## 🚀 Features
- Fine‑tune DistilBERT in 3–5 minutes on Kaggle GPU  
- HuggingFace `datasets` + `transformers`  
- Mixed precision FP16 training  
- Clean inference pipeline  
- Optional FastAPI deployment  
- Beginner‑friendly explanations

---

## 📁 Project Structure
```
repo/
│── README.md
│── requirements.txt
│── app.py
│── src/
│   └── inference_example.py
│── notebook_train/
│   └── main.py
```

---

## 🧠 Model: DistilBERT
DistilBERT is 40% smaller, 60% faster, and retains 97% of BERT’s accuracy.  
Ideal for learning NLP and running on free GPUs.

---

## 🛠 Installation
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training (Kaggle Recommended)
Use the Kaggle notebook for:
- GPU acceleration  
- FP16 mixed precision  
- Fast dataset loading  

---

## 🔍 Inference
```python
from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="./results")
print(pipe("This movie was great!"))
```

---

## 🌐 FastAPI Deployment
Run:
```bash
uvicorn app:app --reload
```

---

## 📝 License
MIT License
