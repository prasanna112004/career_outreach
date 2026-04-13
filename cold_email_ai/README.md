## AI Cold Email Generator – Personalized Outreach Assistant

Generate highly personalized cold emails by combining a live job posting with a structured portfolio knowledge base using Retrieval‑Augmented Generation (RAG).

This project is designed as a **portfolio‑quality AI engineering project**: the codebase is modular, the architecture is explicit, and the RAG pipeline is implemented with production‑style patterns.

---

### 1. Problem Statement

Most cold emails sent for jobs are generic and ignore the details of the role and the candidate’s actual experience. This leads to:

- **Low relevance**: emails don’t reflect the specific responsibilities or stack.
- **Weak personalization**: they don’t highlight the most relevant projects or achievements.
- **Inconsistent quality**: tone, clarity, and structure vary a lot from one email to the next.

This app solves those problems by:

- Scraping a **job posting URL**.
- Retrieving **portfolio snippets** that are semantically aligned with the job description.
- Feeding both into an LLM with a **structured prompt**.
- Generating a **concise, personalized cold email** in a chosen tone.

---

### 2. High‑Level Architecture

**Tech Stack**

- **Python**
- **LangChain** – orchestration, documents, prompt templates
- **OpenAI API** – embeddings + GPT chat model
- **ChromaDB** – vector database for portfolio documents
- **Streamlit** – web UI
- **BeautifulSoup** – web scraping
- **tiktoken** – token counting and prompt introspection

**Project Layout**

`cold_email_ai/`

- `app.py` – Streamlit UI and orchestration.
- `scraper.py` – job description scraping with BeautifulSoup.
- `portfolio_data.py` – static portfolio dataset as LangChain `Document`s.
- `rag_pipeline.py` – text chunking and retrieval over ChromaDB.
- `vector_store.py` – vector store creation and loading.
- `prompt_templates.py` – structured prompt for cold email generation.
- `utils.py` – token counting and email quality scoring.
- `requirements.txt` – Python dependencies.
- `README.md` – documentation (this file).

---

### 3. RAG Pipeline Design

The RAG flow is intentionally simple but mirrors production‑grade designs:

1. **Portfolio Knowledge Base**
   - `portfolio_data.py` defines a small but realistic portfolio:
     - projects (RAG system, sales outreach assistant, MLOps platform),
     - skills (Python, LangChain, OpenAI, ChromaDB, Streamlit, MLOps),
     - experience and achievements.
   - Data is stored as a list of LangChain `Document` objects with rich metadata (`section`, `project_name`, `candidate_name`).

2. **Chunking**
   - `rag_pipeline.chunk_documents` uses `RecursiveCharacterTextSplitter` to turn long portfolio entries into **overlapping chunks**.
   - Parameters (`chunk_size=800`, `chunk_overlap=120`) provide enough context while keeping each chunk embedding‑friendly.

3. **Vector Store Construction**
   - `vector_store.build_vector_store`:
     - Uses `OpenAIEmbeddings` (`text-embedding-3-small` by default).
     - Stores vectors in **ChromaDB** in a local `chroma_db/` directory.
     - Persists the index to disk so it can be reused across app runs.

4. **Retriever**
   - `rag_pipeline.build_portfolio_retriever`:
     - Indexes the chunked documents in Chroma.
     - Exposes a **similarity‑based retriever** (`search_kwargs={"k": 4}` by default).
   - `rag_pipeline.retrieve_relevant_portfolio_context`:
     - Given the job description text, returns the **top‑k portfolio chunks**.

5. **LLM‑Driven Generation**
   - `prompt_templates.build_cold_email_prompt`:
     - Combines:
       - candidate profile,
       - scraped job description,
       - retrieved portfolio context,
       - tone specification (`professional`, `friendly`, `direct`).
     - Builds a `ChatPromptTemplate` with **system + human messages** that constrain style and length.
   - `app.generate_email`:
     - Instantiates `ChatOpenAI` (`gpt-4o-mini` by default).
     - Pipes the prompt into the model.
     - Produces a personalized cold email.

---

### 4. Modules and Responsibilities

- **`scraper.py`**
  - `scrape_job_posting(url) -> JobDescription`
  - Uses `requests` + `BeautifulSoup` to:
    - fetch and parse an arbitrary job posting URL,
    - extract title (`<h1>` / `<title>`),
    - best‑effort company name (meta tags, text heuristics),
    - responsibilities / description (paragraphs and list items),
    - naive required skills via keyword matching.
  - Returns a `JobDescription` dataclass with `.as_plain_text()` for LLM usage.

- **`portfolio_data.py`**
  - `load_portfolio_documents() -> List[Document]`
  - Encodes:
    - candidate profile,
    - 3 key projects,
    - core skills,
    - experience and achievements.
  - All stored as `Document` instances with metadata, enabling **fine‑grained retrieval**.

- **`vector_store.py`**
  - `build_vector_store(documents, persist_dir)` – builds a Chroma index and persists it.
  - `load_vector_store(persist_dir)` – loads an existing Chroma store with the same embedding model.
  - Uses `OpenAIEmbeddings` to stay consistent with the generation model family.

- **`rag_pipeline.py`**
  - `chunk_documents(documents)` – RecursiveCharacterTextSplitter.
  - `build_portfolio_retriever(documents)` – chunks → embeds → Chroma → retriever.
  - `load_portfolio_retriever()` – loads an existing store as a retriever.
  - `retrieve_relevant_portfolio_context(job_text, retriever)` – top‑k portfolio chunks per job.

- **`prompt_templates.py`**
  - `build_cold_email_prompt(job, portfolio_docs, tone)` – main high‑level prompt builder.
  - `format_portfolio_context(docs)` – pretty‑prints retrieved documents into a contextual block.
  - Supports tones: **professional**, **friendly**, **direct**.

- **`utils.py`**
  - `count_tokens(text, model)` – token estimation via `tiktoken`.
  - `score_email_quality(email_text, job, portfolio_docs)` – simple **1–10 scoring** based on:
    - relevance (mentions job title, company, and overlapping skills),
    - personalization (mentions candidate and portfolio‑specific terms),
    - clarity (length band around ~120–300 words).

- **`app.py`**
  - Streamlit entrypoint that coordinates:
    - UI input (job URL, tone),
    - job scraping,
    - portfolio retrieval (cached),
    - prompt construction,
    - LLM call,
    - scoring and display.

---

### 5. Streamlit UI Flow

**Inputs**

- `Job posting URL` (text input).
- `Tone selector` (`professional`, `friendly`, `direct`).

**Actions**

- `Generate Email` – run the full pipeline with the current inputs.
- `Regenerate` – re‑run generation using the same URL and tone.

**Outputs**

- **Generated cold email**
  - Displayed in a text area for easy copying/editing.
  - Includes an estimated **quality score (1–10)**.
- **Retrieved portfolio context preview**
  - Shows which portfolio chunks were surfaced by the retriever.
  - Helps debug and explain personalization.
- **Scraped job description**
  - Shows the flattened input passed to the model.
  - Helpful for understanding scraper behavior.
- **Token estimate**
  - Approximate prompt token count using `tiktoken`.

---

### 6. Running the Project

#### 6.1. Prerequisites

- Python 3.10+ recommended.
- An OpenAI API key with access to:
  - `gpt-4o-mini` (or compatible chat model),
  - `text-embedding-3-small` (or equivalent embedding model).

#### 6.2. Setup

From the root of your workspace:

```bash
cd cold_email_ai
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set your OpenAI key:

```bash
export OPENAI_API_KEY="sk-..."  # macOS/Linux
# or on Windows PowerShell:
$env:OPENAI_API_KEY="sk-..."
```

Alternatively, create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
```

#### 6.3. Run the App

```bash
streamlit run app.py
```

Then open the provided `localhost` URL in your browser.

---

### 7. Extending the System (Portfolio‑Ready Talking Points)

Ideas for how you might extend this project in an AI engineer portfolio:

- **Richer scraping layer**
  - Add site‑specific parsers for LinkedIn, Greenhouse, Lever, and Workday.
  - Use a headless browser (Playwright) for dynamic content.

- **Improved RAG**
  - Add more granular metadata (tech stack, domain, role type).
  - Implement hybrid search (keyword + vector) for better recall.
  - Introduce query transformation or job‑aware re‑ranking steps.

- **Evaluation & Feedback**
  - Log generations and scores in a database.
  - Collect human ratings and tune prompts or routing logic.
  - Add A/B tests over different prompt variants or tones.

- **Multi‑candidate support**
  - Load portfolio data from a database or config instead of a static module.
  - Support multiple candidate profiles in the same app.

---

### 8. Summary

This project demonstrates a **complete, production‑style RAG application**:

- A realistic stack: Python, LangChain, OpenAI, ChromaDB, Streamlit, BeautifulSoup, tiktoken.
- Clear separation of concerns: scraping, data, retrieval, prompting, UI.
- A transparent scoring heuristic to reason about output quality.

It is intentionally simple enough to understand in one sitting, but designed so you can **confidently showcase it in an AI engineering portfolio** and extend it into a more robust product. 

