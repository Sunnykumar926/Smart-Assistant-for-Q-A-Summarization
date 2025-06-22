# 🤖 Smart Assistant for Research Summarization

A **Streamlit-based GenAI application** that reads and understands content from user-uploaded `.pdf` and `.txt` files. It provides **contextual question answering**, **logic-based question generation**, and **source-grounded feedback**, powered by **Google Gemini 1.5** via LangChain.

📁 Upload research papers, legal documents, or technical reports — let the assistant read, reason, and quiz you!

---

## 🎯 Project Objective

This project was developed as part of a GenAI recruitment task to build an **AI-powered assistant** that:

- **Understands** complex documents
- **Answers** free-form user questions with supporting evidence
- **Generates** logic-based questions to test comprehension
- **Evaluates** answers with contextual feedback
- **Justifies** all outputs with references from the document

---

## 🚀 Features

- 📄 **Document Upload**: Accepts `.pdf` and `.txt` files for analysis.
- ✍️ **Auto Summary**: Instantly generates a <150 word summary per document.
- 🧠 **Ask Anything**: Ask detailed questions; the assistant answers with quotes from the document.
- 🎯 **Challenge Me Mode**:
  - Auto-generates logic-based questions from document content
  - Lets users attempt answers
  - Evaluates and justifies with reference-backed feedback
- 🔍 **Document Grounding**: Every response is traceable to its source — no hallucination.

---

## 🧠 How It Works

### 📝 Document Parsing
- Uses **PyPDF2** and UTF-8 decoding to extract structured content.
- Groups content by document name and page number for traceability.

### 🔍 Vector Store
- Documents are chunked using `RecursiveCharacterTextSplitter`.
- Stored locally using **FAISS** and **Google GenerativeAI Embeddings**.

### 💬 Q&A + Feedback Pipeline
- Uses **LangChain** with `ChatGoogleGenerativeAI` (Gemini 1.5 Flash).
- Custom prompts ensure grounded, accurate, and referenced responses.
- User answers are evaluated with a structured rubric and source citations.

---

## 🏗️ Tech Stack

- **Streamlit** – Interactive web UI
- **LangChain** – Chain-based Q&A and prompt engineering
- **Google Gemini 1.5** – LLM backend (via `langchain-google-genai`)
- **FAISS** – Local vector storage and semantic retrieval
- **PyPDF2** – PDF parsing
- **dotenv** – Environment variable management

---

## 📦 Installation

### 🧰 Prerequisites

- Python 3.8+
- Google Gemini API Key (add to `.env` as `GOOGLE_API_KEY`)

### 🔧 Setup
1. Clone repository:
```bash
git clone https://github.com/yourusername/smart-assistant-genai.git
cd smart-assistant-genai
```

2. Create virtual environment:
```bash
python -m venv my_env
source my_env/bin/activate  # Linux/MacOS
my_env\Scripts\activate    # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create `.env` file with your API key:
```bash
GOOGLE_API_KEY=your_api_key_here
```
5. Run your file
```bash
streamlit run app.py
```















