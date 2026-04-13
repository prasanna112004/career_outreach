"""
Streamlit app entrypoint for the AI Cold Email Generator – Personalized Outreach Assistant.

Uses shared services also exposed via FastAPI (``api/main.py``) for multi-client access.
"""

from __future__ import annotations

import streamlit as st

from prompt_templates import ALLOWED_TONES, OUTPUT_TYPES, format_portfolio_context
from resume_parser import clean_resume_text, extract_text_from_upload
from scraper import JobDescription, scrape_job_posting
from services.content_generation import generate_outreach_content
from services.match_analysis import analyze_resume_job_match
from services.model_provider import DEFAULT_CHAT_MODEL
from services.refinement import refine_generated_text
from utils import count_tokens

REFINE_PRESETS = {
    "Improve tone": "Improve the tone to be more polished and professional while staying natural.",
    "Make shorter": "Make the text noticeably shorter; remove redundancy; keep key facts.",
    "Make more confident": "Strengthen wording to sound more confident and decisive; avoid arrogance.",
}


def main() -> None:
    """Main Streamlit UI definition."""
    st.set_page_config(
        page_title="AI Cold Email Generator",
        page_icon="✉️",
        layout="centered",
    )
    st.title("AI Cold Email Generator – Career Outreach Assistant")
    st.write(
        "Generate tailored outreach from a job URL plus **your resume** (or demo portfolio). "
        "Optional: **match insights**, **email / LinkedIn / cover letter**, and **refine** without full regeneration."
    )

    ss = st.session_state
    if "resume_raw_text" not in ss:
        ss.resume_raw_text = None
    if "resume_file_name" not in ss:
        ss.resume_file_name = None
    if "bundle" not in ss:
        ss.bundle = None  # dict with generated, context_text, job_text, quality, match, output_type

    with st.sidebar:
        st.header("Configuration")
        tone = st.selectbox(
            "Tone",
            options=sorted(ALLOWED_TONES),
            index=0,
        )
        output_type = st.selectbox(
            "Output format",
            options=sorted(OUTPUT_TYPES),
            format_func=lambda x: {
                "email": "Email",
                "linkedin": "LinkedIn DM",
                "cover_letter": "Cover letter",
            }.get(x, x),
            index=0,
        )
        chat_model = st.text_input(
            "Groq chat model",
            value=DEFAULT_CHAT_MODEL,
        )
        st.caption("Requires `GROQ_API_KEY` in environment or Streamlit secrets.")

    st.subheader("1. Your background")
    source = st.radio(
        "Content source",
        options=["My resume (upload)", "Demo portfolio (sample)"],
        horizontal=True,
    )

    uploaded = st.file_uploader(
        "Upload resume",
        type=["pdf", "docx", "txt", "md"],
    )
    pasted_resume = st.text_area(
        "Or paste resume text directly",
        height=180,
        placeholder="Paste resume text here if upload parsing fails or you prefer direct input...",
    )

    if uploaded is not None:
        try:
            data = uploaded.getvalue()
            text = extract_text_from_upload(uploaded.name, data)
            ss.resume_raw_text = text
            ss.resume_file_name = uploaded.name
            st.success(f"Loaded **{uploaded.name}** ({len(text):,} characters extracted).")
        except Exception as exc:
            st.error(f"Could not read resume: {exc}")
            ss.resume_raw_text = None

    if pasted_resume.strip():
        ss.resume_raw_text = clean_resume_text(pasted_resume)
        ss.resume_file_name = "pasted_resume"

    if source == "My resume (upload)" and ss.resume_raw_text:
        with st.expander("Preview extracted resume text", expanded=False):
            st.text(ss.resume_raw_text[:8000])

    st.subheader("2. Job posting")
    job_url = st.text_input(
        "Job posting URL",
        placeholder="https://jobs.example.com/role/123",
    )
    jd_manual = st.text_area(
        "Or paste job description directly (recommended for LinkedIn)",
        height=180,
        placeholder="Paste the full JD text here if URL scraping is blocked or incomplete...",
    )
    col_j1, col_j2 = st.columns(2)
    with col_j1:
        manual_title = st.text_input("Manual job title (optional)")
    with col_j2:
        manual_company = st.text_input("Manual company (optional)")

    col1, col2 = st.columns([1, 1])
    with col1:
        generate_clicked = st.button("Generate", type="primary")
    with col2:
        regenerate_clicked = st.button("Regenerate")

    use_resume = source == "My resume (upload)"
    resume_text = ss.resume_raw_text if use_resume else None

    if generate_clicked or regenerate_clicked:
        if not job_url and not jd_manual.strip():
            st.error("Enter a job URL or paste a job description.")
            return
        if use_resume and not resume_text:
            st.error("Upload a resume first.")
            return

        with st.spinner("Scraping job, retrieving context, generating..."):
            job = None
            if jd_manual.strip():
                job = JobDescription.from_manual_input(
                    title=(manual_title or "").strip() or "LinkedIn Role",
                    company=(manual_company or "").strip() or "Company",
                    responsibilities=jd_manual.strip(),
                    skills=[],
                    url=job_url.strip() if job_url.strip() else "manual://input",
                )
            else:
                try:
                    job = scrape_job_posting(job_url)
                except Exception as exc:
                    st.error(f"Failed to scrape job posting: {exc}")
                    return

            try:
                gen_text, quality_score, context_docs, _ = generate_outreach_content(
                    job,
                    tone=tone,
                    chat_model=chat_model,
                    use_resume=use_resume,
                    resume_text=resume_text,
                    output_type=output_type,
                )
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                return

            context_text = format_portfolio_context(context_docs)
            match_data = None
            if use_resume and resume_text:
                try:
                    match_data = analyze_resume_job_match(
                        resume_text,
                        job.as_plain_text(),
                        chat_model=chat_model,
                    )
                except Exception:
                    match_data = None

            ss.bundle = {
                "generated": gen_text,
                "context_text": context_text,
                "job_text": job.as_plain_text(),
                "quality": quality_score,
                "match": match_data,
                "output_type": output_type,
            }
            # Reset editable output widget so regenerated content is shown immediately.
            if "out_edit_area" in ss:
                del ss.out_edit_area

    if ss.bundle:
        b = ss.bundle
        st.subheader("Match insights")
        if b.get("match"):
            m = b["match"]
            st.progress(min(100, max(0, m["match_score"])) / 100.0)
            st.metric("Match score (skill coverage vs JD)", f"{m['match_score']}%")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Strong matches**")
                st.caption(", ".join(m["strong_skills"]) or "—")
            with c2:
                st.markdown("**Missing vs JD**")
                st.caption(", ".join(m["missing_skills"]) or "—")
        else:
            st.caption("Upload a resume to see resume vs job match insights.")

        st.subheader("Generated output")
        st.caption(f"Format: **{b.get('output_type', 'email')}**")
        st.markdown(f"**Heuristic quality score:** {b['quality']} / 10")

        edited_key = "out_edit_area"
        edited = st.text_area(
            "Edit or copy",
            value=b["generated"],
            height=280,
            key=edited_key,
        )

        st.markdown("**Refine (rewrite in place)**")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            b1 = st.button("Improve tone", key="r1")
        with rc2:
            b2 = st.button("Make shorter", key="r2")
        with rc3:
            b3 = st.button("More confident", key="r3")
        custom = st.text_input("Custom refine instruction", key="r_custom")
        apply_custom_refine = st.button("Apply custom refine", key="r_custom_apply")

        trigger_refine = None
        if b1:
            trigger_refine = REFINE_PRESETS["Improve tone"]
        elif b2:
            trigger_refine = REFINE_PRESETS["Make shorter"]
        elif b3:
            trigger_refine = REFINE_PRESETS["Make more confident"]
        elif apply_custom_refine and custom and custom.strip():
            trigger_refine = custom.strip()

        if trigger_refine:
            with st.spinner("Refining..."):
                try:
                    refined = refine_generated_text(
                        edited,
                        trigger_refine,
                        chat_model=chat_model,
                    )
                    ss.bundle["generated"] = refined
                    if "out_edit_area" in ss:
                        del ss.out_edit_area
                    st.success("Refinement applied.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Refine failed: {exc}")

        st.subheader("Retrieved context (RAG)")
        st.text_area("Context", value=b["context_text"], height=220, disabled=True)

        st.subheader("Job description (scraped)")
        st.text_area("JD", value=b["job_text"], height=220, disabled=True)

        extra = resume_text or ""
        st.caption(
            f"Approx. tokens (rough): ~{count_tokens(b['job_text']) + count_tokens(b['context_text']) + count_tokens(extra[:2000])}"
        )

    elif not (job_url or jd_manual.strip()):
        st.info("Paste a job URL or JD text, configure options, then **Generate**.")


if __name__ == "__main__":
    main()
