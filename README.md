# Advanced RAG System

A local Retrieval-Augmented Generation (RAG) application that answers questions using a document knowledge base. The pipeline combines semantic vector search with BM25 keyword retrieval, fuses results, reranks passages, grades evidence, and generates answers with source references. It also supports query transformation, corrective retrieval, and conversational memory.

The project includes a Streamlit interface for interactive Q&A, source inspection, conversation history, and session-state inspection.

The application uses the Gemini API. Successful operation depends on network connectivity, model access, and available API quota.

## Features

- **Document ingestion:** Reads supported text and PDF documents, splits them into overlapping chunks, and stores processed chunks for retrieval.
- **Semantic retrieval:** Embeds document chunks using `sentence-transformers/all-MiniLM-L6-v2` and searches by vector similarity.
- **Hybrid retrieval:** Combines vector retrieval and BM25 keyword retrieval using Reciprocal Rank Fusion (RRF).
- **Query transformation:** Uses Gemini to rewrite or decompose questions into retrieval-friendly queries.
- **Reranking:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank candidate passages.
- **Grounded answer generation:** Uses Gemini to generate answers from retrieved evidence and return source references.
- **Evidence grading and corrective retrieval:** Grades whether the retrieved context is sufficient and can retry retrieval once when evidence is weak.
- **Abstention:** The answer pipeline is designed to indicate when the supplied documents do not contain enough evidence instead of inventing an answer.
- **Conversation memory:** Stores conversation turns, exposes recent history, and supports transcript and session-state inspection.
- **Streamlit UI:** Provides interactive chat, suggested questions, source panels, transcript export, and developer diagnostics.
- **Evaluation harness:** Runs a golden-question dataset and reports source hit rate, keyword coverage, abstention accuracy, and service errors.

## Architecture

```text
Documents (TXT/PDF)
        |
        v
   Ingestion and chunking
        |
        v
 Processed chunks (JSONL)
        |
        +----------------------------+
        |                            |
        v                            v
 Sentence embeddings              BM25 index
        |                            |
        v                            v
  Vector retrieval             Keyword retrieval
        |                            |
        +-------------+--------------+
                      |
                      v
             Reciprocal Rank Fusion
                      |
                      v
                Cross-encoder
                  reranking
                      |
                      v
              Evidence grading
                      |
             +--------+--------+
             |                 |
       Sufficient evidence   Weak evidence
             |                 |
             |             Corrective retry
             |                 |
             +--------+--------+
                      |
                      v
             Gemini answer generation
                      |
                      v
          Answer with source references
          or an insufficient-evidence reply
```

Conversation history is supplied to follow-up resolution and maintained by the CLI or Streamlit session layer.

## Technology Stack

- Python
- Streamlit
- Google Gen AI SDK (Gemini API)
- Sentence Transformers
- PyTorch
- NumPy
- `rank-bm25`
- `pypdf`
- `python-dotenv`
- pytest

## Repository Structure

```text
dotkonnekt-advanced-rag/
├── app/
│   ├── ingestion.py
│   ├── vector_store.py
│   ├── hybrid_retriever.py
│   ├── query_transformer.py
│   ├── reranker.py
│   ├── answer_generator.py
│   ├── evidence_grader.py
│   ├── corrective_rag.py
│   ├── followup_resolver.py
│   ├── conversation_memory.py
│   ├── evaluate.py
│   └── streamlit_app.py
├── data/
│   ├── documents/
│   ├── processed/
│   └── embeddings/
├── evaluation/
│   ├── golden_questions.json
│   └── results.json
├── tests/
├── transcripts/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Some data and output files are generated when ingestion, indexing, and evaluation are run. Ensure the file and folder names above match the final repository.

## Prerequisites

- Python 3.12 (the project was developed with Python 3.12.4).
- A Gemini API key with access to the configured model.
- Internet access for API calls and initial model downloads.

## Setup and Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/KavyaPoddar23/AdvancedRAGsystem.git
cd dotkonnekt-advanced-rag
```

### 2. Create and Activate a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use the activation method appropriate for your environment or activate the environment through VS Code.

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch or torchvision import errors occur, install compatible versions for your Python version and operating system, then verify the imports before launching the application.

### 4. Configure the Gemini API Key

Copy `.env.example` to `.env` and add your own API key:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env` or publish your API key. Ensure `.gitignore` excludes `.env`.

### 5. Add Documents

Place `.txt` or supported PDF documents inside `data/documents/`.

The practice knowledge base contains documents covering:

- Python fundamentals and data structures
- Python testing
- REST APIs and API security
- SQL and SQL injection
- Machine learning
- Embeddings and vector search
- RAG architecture
- Hybrid retrieval
- RAG evaluation and grounding

### 6. Prepare the Document Index

Run these commands from the repository root if they match the entry points in your implementation:

```powershell
python -m app.ingestion
python -m app.vector_store
```

These should create processed chunks and embedding/index artifacts. If the final modules use different command-line entry points, update the commands to match the checked-in code.

### 7. Launch the Streamlit Application

```powershell
python -m py_compile app/streamlit_app.py
python -m streamlit run app/streamlit_app.py
```

Streamlit will print a local URL in the terminal and normally open the application in your browser.

### 8. Run Automated Tests

```powershell
python -m pytest -q
```

Run the tests after making code changes and record the result from your final run.

## Using the Application

1. Start the Streamlit application.
2. Choose a suggested question or enter your own question.
3. Review the generated answer and expand the evidence/source section.
4. Ask a follow-up question to test conversational context.
5. Use the sidebar to inspect the conversation transcript or session state.
6. Export the conversation if you need a sample transcript.

### Suggested Questions

- `How can SQL injection be prevented?`
- `What is the difference between vector search and BM25?`
- `What is the purpose of embeddings in a RAG system?`
- `What evidence should a RAG system use to produce a grounded answer?`
- `What is the population of Mars?`

The last question is intended to test abstention because the knowledge base does not provide the requested information.

## Prompt Design

The project uses task-specific prompts for different stages of the RAG pipeline. The prompt implementations are maintained in their respective source modules.

### 1. Query Transformation Prompt

**Purpose:** Transform the user's question into one or more retrieval-friendly queries while preserving its intended meaning.

**Design rationale:**

- Generate a small set of useful query variants.
- Decompose multi-part questions when separate retrieval queries can improve coverage.
- Preserve the intent of the original question.
- Avoid introducing unsupported facts.

### 2. Follow-Up Resolution Prompt

**Purpose:** Rewrite follow-up questions as standalone questions using the previous conversation.

**Design rationale:**

- Resolve references such as “it,” “that,” or “the first method.”
- Preserve the user's intended meaning.
- Do not answer the question during the resolution step.
- Do not invent context that is absent from the conversation.
- Keep already-standalone questions semantically unchanged.

### 3. Evidence Grading Prompt

**Purpose:** Determine whether retrieved passages provide enough relevant evidence to answer the question.

**Design rationale:**

- Evaluate retrieved context against the user's question.
- Return a structured sufficiency decision and reason.
- Use the reason to guide a corrective retrieval attempt when evidence is weak.
- Avoid treating an empty context as sufficient evidence.

### 4. Answer Generation Prompt

**Purpose:** Generate an answer grounded in retrieved passages and identify the sources supporting it.

**Design rationale:**

- Use retrieved context instead of unsupported outside knowledge.
- Associate claims with source identifiers from the retrieved evidence.
- Validate returned citations against the available context.
- Abstain when evidence is missing or citations cannot be validated.

For the final submission, review the actual prompt strings in `app/query_transformer.py`, `app/followup_resolver.py`, `app/evidence_grader.py`, and `app/answer_generator.py`. Keep this documentation synchronized with the implemented prompts.

## Evaluation

The project includes `evaluation/golden_questions.json`, a six-question evaluation dataset covering:

1. SQL injection prevention
2. Vector search versus BM25
3. The purpose of embeddings
4. Python testing practices
5. An unanswerable question about the population of Mars
6. Evidence requirements for grounded RAG answers

### Recorded Evaluation Results

The evaluation run on **10 October 2026** reported the following results:

| Metric                   | Recorded Result |
| ------------------------ | --------------: |
| Total cases              |               6 |
| Successful cases         |               4 |
| Service errors           |               2 |
| Source hit rate          |           1.000 |
| Average keyword coverage |           0.889 |
| Abstention accuracy      |           1.000 |

### Interpretation

Two cases encountered Gemini API errors: a `503 UNAVAILABLE` error and a `429 RESOURCE_EXHAUSTED` quota error.

The reported source-hit, keyword-coverage, and abstention metrics were calculated from eligible successful cases, not from six successful model responses. These results provide an initial baseline, not proof of perfect overall reliability.

Re-run the evaluation when API quota is available and update this section with the new results if they differ.

### Run the Evaluation Harness

```powershell
python -m app.evaluate
```

To run a smaller subset first:

```powershell
python -m app.evaluate --limit 1
```

The evaluation harness writes its results to `evaluation/results.json`.

To see the supported command-line options:

```powershell
python -m app.evaluate --help
```

## Sample Transcript

The following is a representative example of the intended behavior. Exact wording may vary because responses are generated by the model.

```text
User: How can SQL injection be prevented?

Assistant: SQL injection can be prevented by using parameterized queries
or prepared statements, validating inputs, and applying least-privilege
database permissions.

Sources: 07_sql_injection.txt

User: What is the population of Mars?

Assistant: I don't have enough information in the provided documents
to answer that reliably.
```

## Known Limitations

- **External API dependency:** Gemini availability, latency, model access, and quota affect answer generation, query transformation, follow-up resolution, and evidence grading.
- **Small evaluation dataset:** Six questions provide an initial smoke-test baseline rather than a comprehensive benchmark.
- **Evaluation interruptions:** Service errors are recorded separately and should not be interpreted as successful answers or ordinary answer-quality failures.
- **Local document collection:** Answer quality is limited by the coverage and quality of the indexed documents.
- **Model downloads:** Sentence Transformer models may need to be downloaded on first use.
- **Citation quality:** Citation correctness depends on retrieval quality and citation validation. Source references should be inspected when evaluating responses.

## Security Considerations

- Keep the Gemini API key in `.env`.
- Commit `.env.example`, not `.env`.
- Ensure `.gitignore` excludes `.env`, `.venv/`, Python cache files, and local logs.
- Review staged files before pushing to GitHub.
- If an API key was ever committed, revoke or rotate it. Removing it in a later commit does not remove it from Git history.

## Submission

GitHub Repository: https://github.com/KavyaPoddar23/AdvancedRAGsystem.git
