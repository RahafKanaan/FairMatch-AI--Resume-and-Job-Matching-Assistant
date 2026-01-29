# FairMatch AI
Privacy-First & Explainable Resume–Job Matching System

---

## Overview
FairMatch AI is an ethical, privacy-aware resume–job matching system designed to support transparent and responsible recruitment workflows.

The system performs semantic matching between resumes and job descriptions while enforcing:
- strict input validation,
- personal data anonymization,
- explainable AI outputs,
- and mandatory human oversight.

FairMatch AI **supports decision-making**; it does not automate hiring decisions.

---

## Core Capabilities

### Input Validation
- Automatically detects whether an input is a valid:
  - Resume (CV)
  - Job Description
- Rejects invalid inputs with clear error messages
- Supports:
  - Plain text
  - PDF documents (with OCR fallback)

---

### Privacy-Preserving Processing
Before any analysis:
- Names are removed
- Email addresses are removed
- Phone numbers are removed
- Social links are removed

Matching and ranking operate **only on anonymized content**.

Contact details are:
- Stored separately
- Never used for scoring
- Revealed only upon explicit user request

---

### Semantic Matching Engine
- Text is converted into embeddings using large language models
- Similarity is computed using cosine similarity
- Results are expressed as a clear match percentage
- Optional weighted matching emphasizes job responsibilities and required skills

---

### Explainability & Transparency
For every match, the system generates:
- A structured explanation describing:
  - Aligned skills and experience
  - Missing or weakly represented requirements
- Neutral language with no suitability judgment
- No ranking bias or hiring recommendation

---

## Services

### 1. Individual Service
Designed for individual users.

Workflow:
1. Upload or paste a CV
2. Upload or paste a Job Description
3. Input validation (CV vs Job)
4. Anonymization
5. Match score generation
6. AI explanation
7. Optional detailed alignment analysis

---

### 2. Company Service
Designed for organizations managing multiple resumes.

Workflow:
1. Upload and maintain a persistent CV pool
2. Upload or paste a Job Description
3. Select Top-K matching CVs
4. Review anonymized results
5. Reveal original CV files or contact details only for selected CVs
6. Optional detailed alignment analysis per CV

---

## Data Preparation & Processing Pipeline

1. Input acquisition (Text / PDF)
2. PDF text extraction
3. OCR fallback for image-based PDFs
4. Input validation (CV / Job Description)
5. Text cleaning and normalization
6. PII removal (anonymization)
7. Embedding generation
8. Similarity computation
9. Explainable output generation

---

## Evaluation & Benchmarking
FairMatch AI supports scenario-based evaluation for robustness testing.

Evaluation includes:

- Testing with diverse CV formats
- Image-based and low-quality PDFs
- Non-job or marketing-style inputs
- Manual review of explanation consistency and clarity

Similarity scores are used strictly as alignment indicators, not predictive metrics.

Evaluation focuses on behavioral correctness, safe failure, and responsible output rather than numerical performance benchmarks.

---

## Installation

### System Requirements
- Python 3.9+
- Tesseract OCR installed and accessible via PATH
- Poppler utilities (required for PDF to image conversion)

Poppler is required by the `pdf2image` library to process PDF files,
especially image-based or scanned resumes. Without Poppler, OCR-based
text extraction from PDFs will fail.

### Tesseract Language Support

To enable OCR for Arabic resumes, the Arabic language data file must be installed.

Ensure that `ara.traineddata` is present in the Tesseract `tessdata` directory.

You can verify installed languages by running:
`tesseract --list-langs`

If Arabic is missing, download `ara.traineddata` from:
https://github.com/tesseract-ocr/tessdata


#### Poppler Installation (Windows)

1. Download the latest Poppler release from:
   https://github.com/oschwartz10612/poppler-windows/releases

2. Extract the archive (e.g., to `C:\poppler`)

3. Add the following path to your system PATH:
   `C:\poppler\Library\bin`

4. Verify installation by running:
   `pdftoppm -h`

### Python Dependencies

pip install streamlit openai numpy pypdf pdf2image pytesseract

---

## Configuration

Set the OpenAI API key as an environment variable.

### Windows (PowerShell)

setx OPENAI_API_KEY "your-api-key"

Restart the terminal after setting the variable.

---

## Running the System

### Launch the Web Interface

python -m streamlit run app.py

The application opens automatically in the browser.


---

## Data Handling & Storage
- CV pools are stored persistently for company usage
- Embeddings are cached locally to reduce redundant computation
- Anonymized text is used for all matching operations
- No personal data is included in similarity computations or explanations

The system provides runtime feedback through warnings and validation messages to inform users when inputs are skipped, rejected, or processed with reduced reliability. These messages are intended to improve transparency and user awareness during interaction.

---

## Ethical & Responsible AI Principles
- No automated hiring decisions
- No candidate ranking beyond similarity metrics
- No inference of soft skills or personal traits
- Mandatory human oversight
- Privacy by design
- Full transparency and explainability

---

## Disclaimer
Similarity scores represent semantic alignment only.  
They do not predict performance, competence, or suitability.

Final decisions must always rely on comprehensive human evaluation.

---

## Maintenance & Extension
FairMatch AI is modular by design and can be:
- Extended with additional models
- Integrated into enterprise recruitment systems
- Adapted to new regulatory or ethical requirements
