🚀 Intelligent Job Recommendation System
Hybrid TF-IDF + SBERT Based Job Recommender with Bias & Efficiency Analysis

📌 Overview
This project presents a hybrid job recommendation system that combines lexical matching (TF-IDF) and semantic understanding (SBERT) to deliver accurate and relevant job suggestions.

Unlike traditional systems that rely only on keywords or deep learning, this approach balances precision and semantic meaning, making it highly effective for structured job datasets.

🎯 Problem Statement
Most job recommendation systems fail due to:

❌ Over-reliance on keywords → misses relevant jobs

❌ Over-reliance on semantic models → shows irrelevant jobs

👉 This project solves this by combining both approaches into a hybrid retrieval system.

💡 Key Idea
Final recommendation score:

Score = α × Semantic Similarity + (1 - α) × Lexical Similarity
TF-IDF → captures exact keyword match

SBERT → captures contextual meaning

α (alpha) → controls balance

🏗️ System Architecture
User enters a query

Query is processed using:

TF-IDF (lexical similarity)

SBERT embeddings (semantic similarity)

SBERT embeddings are searched using FAISS

Scores are normalized and combined

Top-K jobs are returned

🧠 Technologies Used
Python

FastAPI

Scikit-learn (TF-IDF)

Sentence Transformers (SBERT)

FAISS (Approximate Nearest Neighbor Search)

Pandas, NumPy

📂 Project Structure
├── app.py                  # FastAPI backend
├── job_recommendation.py   # Core recommendation engine
├── evaluation.py           # Evaluation & metrics
├── admin.html              # Admin panel (add jobs)
├── data/                   # Dataset files
├── artifacts/              # Saved models & embeddings
├── results/                # Evaluation outputs
├── sbert_model/            # SBERT model directory
└── README.md
📊 Dataset
Source: Kaggle

Total Jobs: ~84,000

Fields:

Title

Company

City

Industry

Description

⚙️ Features
🔍 Hybrid job recommendation (TF-IDF + SBERT)

⚡ Fast retrieval using FAISS

➕ Dynamic job addition via API

📊 Evaluation with real metrics

⚖️ Bias & fairness analysis

📈 Alpha tuning for optimal performance

📈 Evaluation Metrics
Precision@K → accuracy of results

Recall@K → coverage of relevant jobs

NDCG@K → ranking quality

✅ Best Performance (α = 0.1)
Metric	Value
Precision@10	0.6466
Recall@10	0.9894
NDCG@10	0.8795
⚖️ Bias & Fairness Analysis
Evaluated across:

Industry

City

Position

Metrics used:

Entropy

HHI (Herfindahl Index)

Gini Coefficient

Jensen-Shannon Divergence

👉 Result: Hybrid model does not increase bias compared to TF-IDF.

⚡ Performance
Metric	Value
Avg Query Time	~73 ms
p95 Latency	~95 ms
Add Job Time	~10 sec
Memory Usage	~1 GB
🚀 How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Run Backend
python app.py
3️⃣ Open Admin Panel
Open admin.html in browser to add jobs.

4️⃣ Test API
POST request:

/api/recommend
🧪 Run Evaluation
Alpha Sweep
python evaluation.py --mode sweep
Manual Evaluation
python evaluation.py --mode manual --labels_csv manual_labels.csv
📌 Key Insights
Pure SBERT performs worst on structured job data

TF-IDF remains a strong baseline

Hybrid approach gives best balance

Better ranking does not increase bias

⚠️ Limitations
Dataset from Kaggle (not benchmark dataset)

Manual labeling may introduce subjectivity

SBERT not fine-tuned for job domain

🔮 Future Work
Fine-tune SBERT on job data

Add user personalization

Optimize update latency

Incorporate feedback learning

👨‍💻 Author
Mohit Pandey
Engineering Student | AI/ML Enthusiast

📜 License
This project is for academic and research purposes.
