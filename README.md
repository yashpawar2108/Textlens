# 📚 TextLens – AI-Powered Text Analysis

**TextLens** is an AI-powered text analysis and summarization tool built with **FastAPI** and integrated with **Groq’s LLM API** for lightning-fast, context-aware results.  
It allows users to **analyze, summarize, and transform** text through an intuitive web interface.

---

## 🚀 Features
- **Text Summarization** – Condense large text into concise summaries.
- **Text Analysis** – Extract key points, sentiment, and context.
- **Text Transformation** – Rephrase or adapt text for different needs.
- **FastAPI Backend** – High-performance, asynchronous API server.
- **Groq LLM Integration** – Harness the speed and intelligence of Groq’s AI models.
- **Secure Config** – Store API keys in `.env` to keep credentials safe.

---

## 🛠️ Tech Stack
**Backend**:
- [FastAPI](https://fastapi.tiangolo.com/) – Async, fast, Python web framework.
- [Groq API](https://groq.com/) – Large language model API.
- [aiohttp](https://docs.aiohttp.org/) – Async HTTP client.
- [pdfplumber](https://github.com/jsvine/pdfplumber) – Extract text from PDFs.

**Frontend**:
- HTML, CSS, JavaScript – Simple, responsive UI.
- Fetch API for real-time interaction with FastAPI endpoints.

**Other Tools**:
- `python-dotenv` – Environment variable management.
- ThreadPoolExecutor – Efficient background processing.

---

## 📂 Project Structure
```txt
Textlens/
│── app/
│ ├── main.py # FastAPI entry point
│ ├── routes/ # API routes
│ ├── services/ # LLM and PDF processing logic
│── static/ # CSS, JS, images
│── templates/ # HTML frontend
│── requirements.txt # Python dependencies
│── .env.example # Example environment variables
│── README.md # Project documentation
```

## ⚡ Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yashpawar2108/Textlens.git
cd Textlens
```
2️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
