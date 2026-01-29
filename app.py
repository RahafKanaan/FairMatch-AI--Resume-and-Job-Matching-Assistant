import streamlit as st
import os
import pickle
import uuid
import hashlib

from backend import (
    read_pdf,
    deidentify_text,
    compute_weighted_match,
    explain_match,
    analyze_alignment,
    analyze_company_alignment,
    is_valid_cv,
    llm_validate_cv,
    is_valid_job_description,
    is_low_quality_text
)

# ===============================
# Config
# ===============================
# Application configuration and static resources
# Defines file paths and UI assets used across the app
POOL_FILE = "company_cv_pool.pkl"
LOGO_FILE = "logo.png"

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="FairMatch AI",
    layout="wide"
)

# ===============================
# Session State
# ===============================
# Initialize Streamlit session state variables
# Used for navigation and multi-step workflows
if "page" not in st.session_state:
    st.session_state.page = "home"

if "company_done" not in st.session_state:
    st.session_state.company_done = False

if "company_results" not in st.session_state:
    st.session_state.company_results = []

if "job_text_company" not in st.session_state:
    st.session_state.job_text_company = ""

# ===============================
# Helpers
# ===============================
# Load stored company CV pool from disk
# Returns an empty list if no pool exists
def load_cv_pool():
    if not os.path.exists(POOL_FILE):
        return []
    try:
        with open(POOL_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return []

# Save the updated company CV pool to disk
def save_cv_pool(pool):
    with open(POOL_FILE, "wb") as f:
        pickle.dump(pool, f)

# Render the application header and branding
def header():
    col1, col2 = st.columns([1, 6])
    with col1:
        if os.path.exists(LOGO_FILE):
            st.image(LOGO_FILE, width=160)
    with col2:
        st.markdown("## **FairMatch AI**")
        st.markdown(
            "<span style='color:gray'>Ethical, Explainable & Multilingual Resume–Job Matching</span>",
            unsafe_allow_html=True
        )
    st.divider()

# ===============================
# HOME PAGE
# ===============================
# Render the landing page and main navigation options
def home_page():
    header()

    st.markdown(
        """
FairMatch AI helps you understand how resumes and job descriptions align in a clear and transparent way.
The system focuses on explaining alignment and gaps rather than making hiring decisions, ensuring privacy, fairness, and human oversight throughout the process.
"""
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Individual Matching")
        st.write(
            """
Designed for individuals who want to see how their resume matches a specific job description.
FairMatch AI highlights alignment and missing requirements to support self-assessment and preparation, without ranking or hiring recommendations.
"""
        )
        if st.button("Start Individual Matching"):
            st.session_state.page = "individual"

    with col2:
        st.subheader("Company Matching")
        st.write(
            """
Built for organizations reviewing multiple resumes against a single job description.
FairMatch AI provides anonymized, explainable insights to support internal review while keeping personal data protected and decisions in human hands. 
"""
        )
        if st.button("Start Company Matching"):
            st.session_state.page = "company"

# ===============================
# INDIVIDUAL PAGE
# ===============================
# Handle individual resume–job matching workflow
# Includes input, validation, anonymization, scoring, and explanation
def individual_page():
    header()
    st.subheader("Individual CV – Job Matching")

    st.info(
        "This tool provides informational insights only and is intended to support understanding, not hiring decisions."
        
    )

    # Select how the CV is provided (text or PDF)
    cv_type = st.radio("CV Input Method", ["Text", "PDF"])
    if cv_type == "Text":
        cv_text = st.text_area("Paste CV text", height=200)
    else:
        cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])
        cv_text = read_pdf(cv_file) if cv_file else ""

    # Select how the job description is provided
    job_type = st.radio("Job Description Input Method", ["Text", "PDF"])
    if job_type == "Text":
        job_text = st.text_area("Paste Job Description", height=200)
    else:
        job_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
        job_text = read_pdf(job_file) if job_file else ""

    # Run validation and matching pipeline
    if st.button("Run Individual Matching"):

        if not cv_text or len(cv_text.strip()) < 30:
            st.error("Unable to extract sufficient text from the CV.")
            return

        if not job_text or len(job_text.strip()) < 30:
            st.error("Unable to extract sufficient text from the job description.")
            return

        if not is_valid_cv(cv_text) and not llm_validate_cv(cv_text):
            st.error("The provided document does not appear to be a valid CV.")
            return

        if not is_valid_job_description(job_text):
            st.error("The provided text does not appear to be a valid job description.")
            return

        # Warn user if extracted CV text quality is low
        if is_low_quality_text(cv_text):
            st.warning(
                "Low-quality text extraction detected. "
                "The CV may be highly visual or minimally structured. "
                "Matching results may be less reliable."
            )

        # Anonymize CV text before similarity computation
        clean_cv = deidentify_text(cv_text)
        # Compute semantic match score
        score = compute_weighted_match(clean_cv, job_text
        )

        st.subheader(f"Match Score: {score}%")
        st.write(explain_match(clean_cv, job_text, score))

        if st.checkbox("Show detailed alignment analysis"):
            st.write(analyze_alignment(clean_cv, job_text))

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"

# ===============================
# COMPANY PAGE
# ===============================
# Company-level resume pool matching workflow
# Supports multiple CVs and Top-K analysis
def company_page():
    header()
    st.subheader("Company Resume Pool Matching")

    st.info(
        "Resumes are anonymized before analysis, and original files are never used for scoring."
        "Results are provided to support internal review and human judgment."
    )

    pool = load_cv_pool()

    st.markdown("### Resume Pool Management")
    # Upload and manage company CV pool
    uploaded = st.file_uploader(
        "Upload CVs (PDF only)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:
        for file in uploaded:
            # Extract text from uploaded CV
            text = read_pdf(file)

            if not text or len(text.strip()) < 30:
                st.warning(f"Initial text extraction failed for '{file.name}' Retrying with alternative extraction.")
                continue

            # Validate that uploaded document is a real CV
            if not is_valid_cv(text) and not llm_validate_cv(text):
                st.warning(f" CV '{file.name}' is not a valid CV and was skipped.")
                continue

            if is_low_quality_text(text):
                st.warning(
                    "One uploaded CV has low-quality extracted text "
                    "and may produce less reliable matching results."
                )

            pdf_bytes = file.getvalue()
            # Prevent duplicate CV uploads using file hash
            pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

            if any(cv["pdf_hash"] == pdf_hash for cv in pool):
                continue

            pool.append({
                "cv_id": str(uuid.uuid4())[:8],
                "clean_text": deidentify_text(text),
                "pdf_bytes": pdf_bytes,
                "pdf_hash": pdf_hash
            })

        # Save updated CV pool for future use
        save_cv_pool(pool)
        st.success(f" CV '{file.name}' processed and added successfully.")

    st.divider()

    st.markdown("### Job Description")
    job_text = st.text_area("Paste Job Description", height=200)
    top_k = st.number_input("Number of top CVs to review", 1, 20, 3)

    # Run semantic matching against the CV pool
    if st.button("Run Matching"):
        if not job_text or len(job_text.strip()) < 30:
            st.error("Job description is too short.")
            return

        if not is_valid_job_description(job_text):
            st.error("The provided text does not appear to be a valid job description.")
            return

        if not pool:
            st.error("No CVs available in the pool.")
            return

        results = []
        for cv in pool:
            # Compute match score for each CV in the pool
            score = compute_weighted_match(cv["clean_text"], job_text)
            results.append({**cv, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        st.session_state.company_results = results[:top_k]
        st.session_state.job_text_company = job_text
        st.session_state.company_done = True

    # Display explainable Top-K matching results
    if st.session_state.company_done:
        for cv in st.session_state.company_results:
            with st.expander(f"CV {cv['cv_id']} — Match {cv['score']}%"):
                st.write(
                    explain_match(
                        cv["clean_text"],
                        st.session_state.job_text_company,
                        cv["score"]
                    )
                )

                if st.checkbox(
                    f"Show detailed alignment ({cv['cv_id']})",
                    key=f"detail_{cv['cv_id']}"
                ):
                    st.write(
                        analyze_company_alignment(
                            cv["clean_text"],
                            st.session_state.job_text_company,
                            cv["score"]
                        )
                    )

                st.download_button(
                    "Download Original CV (PDF)",
                    cv["pdf_bytes"],
                    file_name=f"CV_{cv['cv_id']}.pdf",
                    mime="application/pdf"
                )

    # Route user to the selected page
    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"

# ===============================
# ROUTER
# ===============================
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "individual":
    individual_page()
elif st.session_state.page == "company":
    company_page()