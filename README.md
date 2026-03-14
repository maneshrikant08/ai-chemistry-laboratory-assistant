# AI Chemistry Laboratory Assistant

Streamlit-based chatbot built for the NeoStats AI Engineer case study. The project turns chemistry laboratory manuals into a grounded assistant using local RAG, live web search fallback, and configurable response modes.

## Use Case Objective

The goal of this chatbot is to help students and lab users interact with chemistry manuals more effectively by:

- retrieving answers from local chemistry lab documents
- falling back to web search when local evidence is weak or unavailable
- supporting concise and detailed response styles
- providing a cleaner user experience with citations, short-term memory, and uploadable documents

## Features Implemented

### Mandatory case-study features

- `RAG integration`
  - local PDF knowledge base
  - embeddings-based retrieval
  - FAISS vector index
- `Live web search integration`
  - Tavily-based fallback for external/current information
- `Response modes`
  - concise
  - detailed

### Additional implemented features

- `Search modes`
  - `Auto`
  - `Documents`
  - `Web`
- `Multi-provider chat models`
  - OpenAI
  - Groq
  - Gemini
- `Short-term memory`
  - recent messages reused for conversational continuity
- `User document uploads`
  - PDF, DOCX, TXT, PNG, JPG, JPEG
- `Citations`
  - `[S#]` for local document sources
  - `[W#]` for web sources
- `General chat handling`
  - greetings and personal follow-ups bypass source-heavy RAG rendering
- `Improved UI`
  - dark-mode presentation styling for demo use

## Project Structure

```text
.
├── AI_UseCase/
│   ├── app.py
│   ├── rebuild_index.py
│   ├── requirements.txt
│   ├── config/
│   ├── models/
│   └── utils/
├── data/
│   ├── kelm201.pdf ... kelm207.pdf
│   └── index/
└── README.md
```

## Local Setup

### 1. Install dependencies

```powershell
python -m pip install -r AI_UseCase\requirements.txt
```

### 2. Create environment file

Create `AI_UseCase/.env` from `AI_UseCase/.env.example` and fill in the keys you want to use.

Required for the current implementation:

- `OPENAI_API_KEY`

Optional depending on usage:

- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `TAVILY_API_KEY`
- `TESSERACT_CMD`

### 3. Run the app

```powershell
streamlit run AI_UseCase\app.py
```

## Rebuild the Base Index

If you change source PDFs, chunk settings, or embeddings, rebuild the FAISS index:

```powershell
python AI_UseCase\rebuild_index.py
```

## Recommended Demo Flow

Use these prompts during demonstration:

1. `What is in unit 4?`
2. Switch to `Web` mode and ask: `What is CNG?`
3. Say `Hi I am Shrikant` and then ask: `What is my name?`
4. Upload a custom file and click `Upload`
5. Toggle between `concise` and `detailed`

## Streamlit Cloud Deployment

Deploy the app with:

- repository: this GitHub repository
- app entrypoint: `AI_UseCase/app.py`
- secrets configured in Streamlit Cloud, not committed to GitHub

### Deployment Link

Add your Streamlit Cloud URL here after deployment:

`[Add deployment link here]`

## Challenges Faced

- improving retrieval for short structural queries like `unit 4`
- separating document-backed answers from web-backed answers cleanly
- preventing unsupported LLM answers when evidence is weak
- balancing grounded question answering with normal conversational behavior
- making the output clearer through source mode, citations, and UI improvements

## Submission Notes

For NeoStats submission, keep the repository public and reviewer-friendly:

- do not commit real API keys
- keep the README clear and runnable
- include the deployment link in the final PPT
- create the final zip without the virtual environment
