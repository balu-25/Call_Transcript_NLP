"""
app.py
-------
Streamlit application: Call Transcript Analyzer (NLP)

Requirements covered:
1) Upload transcript via PDF or direct text typing. Single / two person / conference call.
2) Trained on a labeled call-transcript dataset (Kaggle-style; see data/prepare_dataset.py).
3) NLP classification pipeline producing the 4 extracted attributes below.
4) Extracts:
    i)   Tone (Neutral / High / Low)
    ii)  Number of people on the call
    iii) Purpose (Family / Friend / Complaint / Business / Fraud Call)
    iv)  Redacts passwords/confidential info before display or export (security)
    v)   Terrorism / anti-national content -> forced "Emergency" classification
"""

import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.text_extractor import extract_text_from_pdf, clean_transcript_text
from utils.redaction import redact_text, contains_sensitive_keywords
from utils.tone_analyzer import classify_tone
from utils.speaker_detector import detect_speakers
from utils.purpose_classifier import classify_purpose


st.set_page_config(
    page_title="Call Transcript Analyzer",
    page_icon="📞",
    layout="wide",
)

MODEL_MISSING_MSG = (
    "Trained model not found. Run these commands once in the project folder:\n\n"
    "    python data/prepare_dataset.py\n"
    "    python models/train_model.py\n"
)


def render_header():
    st.title("📞 Call Transcript Analyzer")
    st.caption(
        "Upload a call transcript (PDF or typed text) to automatically extract "
        "tone, speaker count, and call purpose — with sensitive data redacted for privacy."
    )


def render_sidebar():
    with st.sidebar:
        st.header("About")
        st.markdown(
            "This tool analyzes call transcripts using NLP:\n\n"
            "- **Tone**: Neutral / High / Low\n"
            "- **Speakers**: Single / Two person / Conference call\n"
            "- **Purpose**: Family, Friend, Complaint, Business, Fraud, or **Emergency**\n"
            "- **Privacy**: passwords, OTPs, card numbers, and other confidential "
            "data are automatically redacted before analysis is shown or exported.\n"
        )
        st.divider()
        st.markdown(
            "**Tip:** For best speaker detection, format multi-person transcripts as:\n\n"
            "```\nJohn: Hello, how are you?\nMary: I'm good, thanks!\n```"
        )
        st.divider()
        st.caption("Tech stack: Python · Streamlit · scikit-learn · NLTK · pdfplumber")


def get_input_text():
    tab_upload, tab_type = st.tabs(["📄 Upload PDF", "⌨️ Type / Paste Text"])

    text = ""
    with tab_upload:
        uploaded_file = st.file_uploader("Upload a transcript PDF", type=["pdf"])
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                try:
                    raw = extract_text_from_pdf(uploaded_file.read())
                    text = clean_transcript_text(raw)
                    if text:
                        st.success(f"Extracted {len(text)} characters from PDF.")
                        with st.expander("Preview extracted text"):
                            st.text(text[:3000])
                    else:
                        st.warning("No extractable text found in this PDF (it may be scanned/image-based).")
                except Exception as e:
                    st.error(f"Could not read PDF: {e}")

    with tab_type:
        typed = st.text_area(
            "Paste or type the transcript here",
            height=250,
            placeholder="Example:\nAgent: Thank you for calling support...\nCustomer: My internet has been down for 3 days...",
        )
        if typed.strip():
            text = typed.strip()

    return text


def render_results(text: str):
    # Step 1: redact sensitive info FIRST — nothing downstream sees raw secrets
    redacted_text, redacted_categories = redact_text(text)
    sensitive_hits = contains_sensitive_keywords(text)

    if redacted_categories:
        st.warning(
            f"🔒 Sensitive information detected and redacted before analysis: "
            f"{', '.join(redacted_categories)}"
        )

    # Step 2: run NLP extraction on the redacted text (purpose classifier still
    # sees enough context even with secrets masked, since masking replaces only
    # the sensitive span, not the whole message)
    with st.spinner("Analyzing transcript..."):
        tone_result = classify_tone(redacted_text)
        speaker_result = detect_speakers(redacted_text)
        try:
            purpose_result = classify_purpose(redacted_text)
        except FileNotFoundError as e:
            st.error(str(e))
            st.info(MODEL_MISSING_MSG)
            return

    # Emergency banner takes priority over everything else visually
    if purpose_result["is_emergency"]:
        st.error(
            "🚨 **EMERGENCY ALERT**: This transcript contains content matching "
            "terrorism / anti-national / threat-related patterns. Flagged terms: "
            f"{', '.join(purpose_result['emergency_terms'])}. "
            "Please escalate to appropriate authorities."
        )

    st.subheader("📊 Analysis Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🎭 Tone", tone_result["tone"], f"confidence {tone_result['confidence']:.0%}")
    with col2:
        st.metric("👥 Speakers", speaker_result["call_type"],
                   f"{speaker_result['num_speakers']} detected")
    with col3:
        purpose_display = purpose_result["purpose"]
        st.metric("🎯 Purpose", purpose_display, f"confidence {purpose_result['confidence']:.0%}")

    st.divider()

    tab_details, tab_redacted, tab_export = st.tabs(["🔍 Details", "🔒 Redacted Transcript", "⬇️ Export"])

    with tab_details:
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Tone signal breakdown**")
            st.json(tone_result["details"])
            st.markdown("**Speaker detection method**")
            st.write(f"Method: `{speaker_result['method']}`")
            if speaker_result["speaker_labels"]:
                st.write("Detected labels:", speaker_result["speaker_labels"])
        with d2:
            st.markdown("**Purpose class probabilities**")
            st.bar_chart(purpose_result["class_probabilities"])
            if sensitive_hits:
                st.markdown("**⚠️ Sensitive keywords originally present**")
                st.write(sensitive_hits)

    with tab_redacted:
        st.markdown(
            "This is the version of the transcript used for analysis — "
            "confidential fields are masked and never stored or sent elsewhere in plain form."
        )
        st.text_area("Redacted transcript", redacted_text, height=300, disabled=True)

    with tab_export:
        report = build_report(tone_result, speaker_result, purpose_result, redacted_text)
        st.download_button(
            "Download Analysis Report (.txt)",
            data=report,
            file_name="call_analysis_report.txt",
            mime="text/plain",
        )
        st.caption("The export contains only the redacted transcript and analysis results — no raw sensitive data.")


def build_report(tone_result, speaker_result, purpose_result, redacted_text) -> str:
    lines = [
        "CALL TRANSCRIPT ANALYSIS REPORT",
        "=" * 40,
        f"Tone: {tone_result['tone']} (confidence {tone_result['confidence']:.0%})",
        f"Speakers: {speaker_result['call_type']} ({speaker_result['num_speakers']} detected, method={speaker_result['method']})",
        f"Purpose: {purpose_result['purpose']} (confidence {purpose_result['confidence']:.0%})",
    ]
    if purpose_result["is_emergency"]:
        lines.append(f"** EMERGENCY FLAG ** matched terms: {', '.join(purpose_result['emergency_terms'])}")
    lines.append("")
    lines.append("Class probabilities:")
    for cls, prob in purpose_result["class_probabilities"].items():
        lines.append(f"  - {cls}: {prob:.1%}")
    lines.append("")
    lines.append("-" * 40)
    lines.append("REDACTED TRANSCRIPT (sensitive info masked):")
    lines.append("-" * 40)
    lines.append(redacted_text)
    return "\n".join(lines)


def main():
    render_header()
    render_sidebar()

    if not os.path.exists(os.path.join("models", "purpose_classifier.joblib")):
        st.error("⚠️ Model not trained yet.")
        st.code(MODEL_MISSING_MSG)
        st.stop()

    text = get_input_text()

    if text:
        st.divider()
        render_results(text)
    else:
        st.info("Upload a PDF or paste a transcript above to begin analysis.")


if __name__ == "__main__":
    main()
