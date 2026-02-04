import re
import os
import hashlib
import tempfile
import numpy as np
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from openai import OpenAI


# ===============================
# OpenAI Client
# ===============================
# Initialize OpenAI client for embeddings and LLM calls
client = OpenAI()

# ===============================
# LLM CALLING
# ===============================
# Send a prompt to the language model and return the generated response
def call_llm(prompt, temperature=0.2):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

# ===============================
# TEXT CHUNKING
# ===============================
# Split long text into manageable chunks for embedding generation
def chunk_text(text, max_chars=3000):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

# ===============================
# EMBEDDING CACHE
# ===============================
# In-memory cache to avoid recomputing embeddings for the same text
_embedding_cache = {}

# Generate a stable hash for a given text
def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Safely generate and cache embeddings for long texts
def get_safe_embedding(text):
    key = text_hash(text)
    if key in _embedding_cache:
        return _embedding_cache[key]

    chunks = chunk_text(text)
    embeddings = []

    for chunk in chunks:
        res = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(res.data[0].embedding)

    vec = np.mean(embeddings, axis=0)
    _embedding_cache[key] = vec
    return vec

# ===============================
# OCR IMPROVEMENT
# ===============================
# Clean and normalize OCR output text
def clean_ocr_text(text):
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

# Extract text from scanned PDFs using OCR
def extract_text_with_ocr(uploaded_file):
    text = ""

    temp_dir = tempfile.mkdtemp()
    temp_pdf_path = os.path.join(temp_dir, "temp.pdf")

    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    poppler_bin = os.getenv("POPPLER_PATH", None)

    try:
        images = convert_from_path(
            temp_pdf_path,
            dpi=300,
            poppler_path=poppler_bin
        )

        for img in images:
            page_text = pytesseract.image_to_string(
                img,
                lang="ara+eng",
                config="--oem 3 --psm 6"
            )
            text += "\n" + page_text

    except Exception as e:
        print("OCR failed:", e)

    finally:
        try:
            os.remove(temp_pdf_path)
            os.rmdir(temp_dir)
        except Exception:
            pass

    return clean_ocr_text(text)

# ===============================
# PDF READER
# ===============================
# Compute a simple quality score for extracted text
def text_quality_score(text):
    return len(text.strip())

# Extract text from PDF with OCR fallback if needed
def read_pdf(uploaded_file):
    text = ""

    try:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += "\n" + t
    except Exception:
        pass

    if len(text.split()) < 30:
        ocr_text = extract_text_with_ocr(uploaded_file)

        if len(ocr_text.split()) > len(text.split()):
            return ocr_text

    return text

# ===============================
# INPUT VALIDATION
# ===============================
# Validate whether input text represents a meaningful job description
def is_valid_job_description(text):
    """
    Validate whether text represents a meaningful job-related description
    and not a personal, narrative, or promotional document.
    Supports Arabic and English.
    """
    text_lower = text.lower()

    personal_style_indicators = [
        
        "i am", "i have", "i would", "i believe",
        "my experience", "my background",
        "i am writing", "i look forward",
        "thank you", "sincerely", "best regards",
        "dear",

        
        "أنا", "لدي", "أمتلك", "أرغب",
        "خبرتي", "مسيرتي", "أعتقد",
        "يسعدني", "أتطلع", "أشكركم",
        "مع فائق الاحترام", "تحية"
    ]

    personal_score = sum(k in text_lower for k in personal_style_indicators)

    if personal_score >= 3:
        return False

    job_indicators = [
        
        "responsibilities", "requirements", "qualifications",
        "role", "position", "skills", "experience",
        "we are looking for", "you will",

        
        "المهام", "المسؤوليات", "المسؤوليات الوظيفية",
        "المتطلبات", "المؤهلات",
        "الدور الوظيفي", "الخبرة", "المهارات", "يشترط"
    ]

    role_signals = [
        
        "looking for", "seeking", "responsible for",
        "manage", "coordinate", "support",

        
        "نبحث عن", "مسؤول عن", "إدارة",
        "تقديم الدعم", "العمل على", "التعامل مع"
    ]

    branding_indicators = [
        
        "about us", "who we are", "our culture",
        "our values", "mission", "vision",
        "fast-growing", "high-growth",

        
        "من نحن", "عن الشركة",
        "ثقافتنا", "قيمنا",
        "رؤيتنا", "رسالتنا"
    ]

    job_score = sum(k in text_lower for k in job_indicators)
    role_score = sum(k in text_lower for k in role_signals)
    branding_score = sum(k in text_lower for k in branding_indicators)

    if branding_score >= 2 and job_score == 0 and role_score == 0:
        return False

    if job_score >= 2:
        return True

    if job_score >= 1 and role_score >= 1:
        return True

    if role_score >= 2:
        return True

    return False

# Validate whether input text is a real resume (CV)
def is_valid_cv(text):
    """
    Validate whether the input text is a resume (CV)
    and not a cover letter, portfolio, or unrelated document.
    Supports English and Arabic.
    """
    text_lower = text.lower()

    cv_indicators = [
        
        "education", "experience", "skills", "projects",
        "certification", "work experience", "professional experience",

        
        "التعليم", "الخبرة", "الخبرات",
        "المهارات", "المشاريع",
        "الشهادات", "الدورات", "الخبرة العملية"
    ]

    non_cv_indicators = [
        
        "cover letter", "dear hiring manager",
        "portfolio", "case study", "personal statement", "about me",

        
        "رسالة", "رسالة تغطية",
        "نبذة", "نبذة شخصية", "عنّي", "عني",
        "دراسة حالة", "ملف أعمال"
    ]

    cv_score = sum(k in text_lower for k in cv_indicators)
    non_cv_score = sum(k in text_lower for k in non_cv_indicators)

    if non_cv_score >= 2 and cv_score == 0:
        return False

    if cv_score < 2:
        return False

    return True

# Use the language model as a fallback to validate ambiguous CVs
def llm_validate_cv(text):
    """
    Fallback LLM-based CV validation.
    Language-agnostic (supports Arabic and English).
    """
    prompt = f"""
You are validating whether the following text is a resume (CV).

Text:
{text[:3000]}

Answer only with:
VALID_CV or NOT_CV
"""
    return call_llm(prompt, temperature=0).strip() == "VALID_CV"
#---------------------------------------
#add new
#----------------------------------------
def extract_education_text(cv_text):
    """
    Extract education-related lines from a CV.
    Supports Arabic and English.
    """
    education_keywords = [
        # English
        "education", "degree", "bachelor", "master", "phd",
        "university", "college",

        # Arabic
        "التعليم", "بكالوريوس", "ماجستير", "دكتوراه",
        "جامعة", "كلية"
    ]

    lines = cv_text.splitlines()
    education_lines = [
        line for line in lines
        if any(k.lower() in line.lower() for k in education_keywords)
    ]

    return "\n".join(education_lines)

def extract_job_role_representation(job_text):
    """
    Build a semantic representation of the job role.
    - Uses explicit job titles if present
    - Otherwise infers role identity from responsibilities and skills
    - Supports Arabic and English
    """

    lines = [l.strip() for l in job_text.splitlines() if l.strip()]
    text_lower = job_text.lower()

    # ----------------------------------------
    # Explicit job title indicators
    # ----------------------------------------
    explicit_title_keywords = [
        # English
        "position", "role", "job title", "we are looking for",
        "seeking", "hiring",

        # Arabic
        "المسمى الوظيفي", "نبحث عن", "مطلوب", "الوظيفة"
    ]

    explicit_titles = []

    for line in lines[:5]:  # search only early lines
        if any(k in line.lower() for k in explicit_title_keywords):
            explicit_titles.append(line)

    # ----------------------------------------
    # If explicit title exists → trust it
    # ----------------------------------------
    if explicit_titles:
        return " ".join(explicit_titles)

    # ----------------------------------------
    # Otherwise infer role from job content
    # ----------------------------------------
    inferred_role_lines = []

    role_signals = [
        # English
        "responsible for", "will work on", "manage", "develop",
        "design", "analyze", "implement", "support",

        # Arabic
        "مسؤول عن", "العمل على", "تطوير", "تصميم",
        "تحليل", "تنفيذ", "إدارة", "تقديم الدعم"
    ]

    for line in lines:
        if any(k in line.lower() for k in role_signals):
            inferred_role_lines.append(line)

    # Fallback: use whole description (trimmed)
    if not inferred_role_lines:
        return job_text[:500]

    return "\n".join(inferred_role_lines)

    
# ===============================
# JOB KEY POINT EXTRACTION
# ===============================
# Extract key responsibilities and required skills from a job description
def extract_job_key_points(job_text):
    lines = job_text.splitlines()
    responsibilities, skills = [], []

    responsibility_keywords = [
        
        "responsib", "duties", "tasks",

        
        "المهام", "المسؤوليات", "الواجبات"
    ]

    skill_keywords = [
        
        "skill", "require", "qualification",

        
        "المهارات", "المتطلبات", "المؤهلات", "يشترط"
    ]

    for line in lines:
        l = line.lower()
        if any(w in l for w in responsibility_keywords):
            responsibilities.append(line)
        elif any(w in l for w in skill_keywords):
            skills.append(line)

    return {
        "responsibilities": responsibilities,
        "skills": skills
    }

# ===============================
# PRIVACY
# ===============================
# Remove personal identifiable information (PII) from resume text
def deidentify_text(text):
    lines = text.splitlines()

    if lines:
        first_line = lines[0].strip()
        if 1 < len(first_line.split()) <= 4:
            lines[0] = "[NAME_REMOVED]"

    text = "\n".join(lines)

    text = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', '[NAME_REMOVED]', text)

    text = re.sub(
        r'[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.-]+\s*\.\s*[a-zA-Z]{2,}',
        '[EMAIL_REMOVED]',
        text
    )

    text = re.sub(
        r'\+?\d[\d\s\-()]{7,}\d',
        '[PHONE_REMOVED]',
        text
    )

    text = re.sub(
        r'(https?://)?(www\.)?linkedin\.com/\S+',
        '[LINKEDIN_REMOVED]',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r'(https?://)?(www\.)?github\.com/\S+',
        '[GITHUB_REMOVED]',
        text,
        flags=re.IGNORECASE
    )

    return text

# ===============================
# SIMILARITY & MATCHING
# ===============================
# Compute cosine similarity between two embedding vectors
def cosine_similarity(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return np.dot(v1, v2) / denom

# Compute a weighted semantic match score between a CV and a job description
def compute_weighted_match(cv_text, job_text):
    """
    Compute a weighted semantic match score between a CV and a job description.

    The final score combines:
    - General semantic alignment between the CV and the full job description (50%)
    - Focused alignment with job responsibilities and required skills (40%)
    - Educational alignment with the inferred job role (10%)

    All comparisons are semantic (embedding-based), not keyword or title matching.
    """

    # ---------------------------------------
    # 1. Extract focused job content
    # ---------------------------------------
    job_points = extract_job_key_points(job_text)
    focus_text = "\n".join(
        job_points["responsibilities"] + job_points["skills"]
    )

    # ---------------------------------------
    # 2. Base semantic similarity (CV ↔ Job)
    # ---------------------------------------
    cv_vec = get_safe_embedding(cv_text)
    job_vec = get_safe_embedding(job_text)
    base_score = cosine_similarity(cv_vec, job_vec)

    # ---------------------------------------
    # 3. Focused similarity (CV ↔ Skills & Responsibilities)
    # ---------------------------------------
    if focus_text.strip():
        focus_vec = get_safe_embedding(focus_text)
        focus_score = cosine_similarity(cv_vec, focus_vec)
    else:
        focus_score = base_score  # graceful fallback

    # ---------------------------------------
    # 4. Education ↔ Job Role semantic alignment
    # ---------------------------------------
    education_text = extract_education_text(cv_text)
    job_role_text = extract_job_role_representation(job_text)

    if education_text.strip() and job_role_text.strip():
        edu_vec = get_safe_embedding(education_text)
        role_vec = get_safe_embedding(job_role_text)
        education_score = cosine_similarity(edu_vec, role_vec)
    else:
        # fallback if education or role cannot be inferred
        education_score = base_score

    # ---------------------------------------
    # 5. Final weighted score
    # ---------------------------------------
    final_score = (
        0.50 * base_score +
        0.40 * focus_score +
        0.10 * education_score
    )

    return round(final_score * 100, 2)


# ===============================
# PROMPTS
# ===============================
EXPLAIN_PROMPT_FINAL = """
You are an AI assistant supporting resume–job matching analysis for HR review.
You operate strictly in a supportive role and do NOT make hiring decisions.
Do NOT mention or infer any personal names. Refer to the person only as "the candidate".

Language Policy:
- Detect the language of the job description.
- If the resume language differs from the job description, ignore the resume language and follow the job description language strictly.
- Generate the entire output in the same language as the job description.
- If the job description is written in Arabic, respond in clear, formal Modern Standard Arabic suitable for professional HR documentation.
- Do not mix languages in the output.


Resume (anonymized):
{cv_text}

Job Description:
{job_text}

Match Score: {match_percentage}%

Task:
Provide a clear and structured explanation of the match by addressing:
1. Key skills and experiences from the resume that directly align with the job requirements
2. Important job requirements that are missing or weakly represented in the resume
3. The main factors that contributed positively or negatively to the match score

Guidelines:
- Be factual, neutral, and descriptive
- Do NOT rank the candidate or suggest suitability
- Do NOT recommend hiring or rejection

Closing Statement (must be included at the end of the output):
Note that soft skills cannot be reliably matched or evaluated through automated text analysis.
In addition, technical skills identified in the resume require further validation through
human-led assessment methods such as technical interviews, coding tasks, or practical tests.
Final decisions must always rely on comprehensive human evaluation.

Important Note:
This explanation is intended to support human evaluation only and must not be used
as a standalone hiring decision.
"""

ALIGNMENT_PROMPT = """
You are an AI assistant performing a detailed resume–job alignment analysis
for internal HR review purposes only.
Do NOT mention or infer any personal names. Refer to the person only as "the candidate".

Language Policy:
- Detect the language of the job description.
- If the resume language differs from the job description, ignore the resume language and follow the job description language strictly.
- Generate the entire output in the same language as the job description.
- If the job description is written in Arabic, respond in clear, formal Modern Standard Arabic suitable for professional HR documentation.
- Do not mix languages in the output.

Resume (anonymized):
{cv_text}

Job Description:
{job_text}

Task:
Step 1: Explicitly extract and list the job requirements that are clearly addressed by the resume.
Step 2: Explicitly extract and list the job requirements that are partially addressed or missing.

Instructions:
- Focus strictly on skills, tools, technologies, experience, and qualifications
- Separate extraction from interpretation
- Do NOT evaluate candidate suitability
- Do NOT make hiring recommendations

Return the result using clear bullet points under each step.

Closing Statement (must be included at the end of the output):
Soft skills, interpersonal abilities, and contextual competencies cannot be accurately
assessed through automated resume analysis. Furthermore, the presence of technical skills
in a resume does not guarantee proficiency and must be validated through interviews,
practical assessments, or other human-led evaluation methods.

Important Note:
This analysis supports transparency and fairness and must be reviewed by a human decision-maker.
"""

COMPANY_ALIGNMENT_PROMPT = """
You are an AI assistant conducting a structured resume–job alignment analysis
to support ethical and transparent HR screening.
Do NOT mention or infer any personal names. Refer to the person only as "the candidate".

Language Policy:
- The output language MUST be the same as the job description language.
- EVEN IF the resume is written in a different language, you MUST ignore the resume language completely.
- Do NOT translate resume content verbatim.
- Do NOT mix languages under any circumstances.
- Any violation of this rule is considered an incorrect output.


Resume (anonymized):
{cv_text}

Job Description:
{job_text}

Match Score: {match_percentage}%

Task:
Produce the analysis using the following structure only:

Aligned Skills and Experience:
- List specific skills, tools, or experiences from the resume that directly match the job requirements

Missing or Weakly Represented Requirements:
- List required or preferred skills, tools, or experiences from the job description
  that are not clearly demonstrated in the resume

Overall Alignment Summary:
- Provide a neutral, high-level summary of alignment without ranking, scoring,
  or suitability judgment

Constraints:
- Do NOT compare candidates
- Do NOT recommend hiring or rejection
- Do NOT infer potential performance or future success

Closing Statement (must be included at the end of the output):
Automated analysis cannot reliably assess soft skills or contextual factors such as
communication, teamwork, and problem-solving in real-world settings. In addition,
technical skills identified through text-based analysis require further validation
through human evaluation, interviews, and practical assessments before any decision
is made.

Important Note:
This output is designed to support responsible HR decision-making and must
always be interpreted alongside human judgment.
"""


# ===============================
# ANALYSIS FUNCTIONS
# ===============================
# Generate a high-level explanation of the CV–job match score
def explain_match(cv_text, job_text, match_percentage):
    prompt = EXPLAIN_PROMPT_FINAL.format(
        cv_text=cv_text,
        job_text=job_text,
        match_percentage=match_percentage
    )
    return call_llm(prompt)

# Generate a detailed alignment analysis between CV and job requirements
def analyze_alignment(cv_text, job_text):
    prompt = ALIGNMENT_PROMPT.format(
        cv_text=cv_text,
        job_text=job_text
    )
    return call_llm(prompt)

# Generate alignment analysis for company-level resume screening
def analyze_company_alignment(cv_text, job_text, match_percentage):
    prompt = COMPANY_ALIGNMENT_PROMPT.format(
        cv_text=cv_text,
        job_text=job_text,
        match_percentage=match_percentage
    )
    return call_llm(prompt)

# ===============================
# TEXT QUALITY CHECK
# ===============================
# Detect whether extracted text is likely low-quality or unreliable OCR output
def is_low_quality_text(text):
    """
    Detect whether extracted text is likely low-quality OCR output.
    Works for both Arabic and English text.
    """
    if not text:
        return True

    char_count = len(text)
    word_count = len(text.split())

    # Very short content (likely broken OCR or empty)
    if char_count < 200:
        return True

    # Too few words for a real CV
    if word_count < 20:
        return True

    return False

