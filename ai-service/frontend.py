import streamlit as st
import requests
import time

# Page config
st.set_page_config(
    page_title="AI Document Summarizer", 
    page_icon="📄", 
    layout="wide"
)

# --- CUSTOM UI MAGIC ---
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
    }
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        line-height: 1.6;
        color: #1e293b;
    }
    .keyword-tag {
        background: #e0f2fe;
        color: #0369a1;
        padding: 6px 12px;
        border-radius: 20px;
        margin: 4px;
        display: inline-block;
        font-weight: 500;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.info("This tool uses **FLAN-T5** for abstractive summarization and **KeyBERT** for keyword extraction.")
    st.divider()
    st.caption("Developed for engineering portfolio project.")

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>📄 AI Document Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Transform long PDFs into concise, actionable insights</p>", unsafe_allow_html=True)
st.divider()

# --- UPLOAD SECTION ---
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader("📂 Upload your PDF", type=["pdf"], help="Limit to 10MB for best results.")

    if uploaded_file is not None:
        st.success(f"✅ **{uploaded_file.name}** uploaded successfully!")
        
        if st.button("🚀 Generate Insights", use_container_width=True):
            # Using a status container for a more professional feel
            with st.status("🧠 AI is processing...", expanded=True) as status:
                st.write("Extracting text from PDF...")
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                
                try:
                    # Increased timeout because AI summarization takes time
                    response = requests.post(
                        "http://127.0.0.1:8000/summarize",
                        files=files,
                        timeout=180 
                    )

                    if response.status_code == 200:
                        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                        data = response.json()

                        summary = data.get("summary", "No summary generated.")
                        bullets = data.get("bullets", "No key points found.")
                        keywords = data.get("keywords", "")

                        st.divider()

                        # --- RESULTS TABS ---
                        tab1, tab2, tab3 = st.tabs(["📌 Executive Summary", "📊 Key Takeaways", "🏷 Semantic Keywords"])

                        with tab1:
                            st.markdown(f'<div class="card">{summary}</div>', unsafe_allow_html=True)

                        with tab2:
                            # Split bullets if they come as a single string to ensure clean rendering
                            formatted_bullets = bullets.replace("\n", "<br>")
                            st.markdown(f'<div class="card">{formatted_bullets}</div>', unsafe_allow_html=True)

                        with tab3:
                            if keywords:
                                # Convert comma-separated string to tags
                                kw_list = keywords.split(", ")
                                kw_html = "".join([f'<span class="keyword-tag">{k}</span>' for k in kw_list])
                                st.markdown(f'<div class="card">{kw_html}</div>', unsafe_allow_html=True)
                            else:
                                st.write("No keywords identified.")

                        st.divider()

                        # --- ACTIONS ---
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.download_button(
                                label="📥 Download as TXT",
                                data=f"SUMMARY:\n{summary}\n\nKEY POINTS:\n{bullets}\n\nKEYWORDS:\n{keywords}",
                                file_name=f"Summary_{uploaded_file.name.split('.')[0]}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        with col_b:
                            if st.button("🔄 Clear and Restart", use_container_width=True):
                                st.rerun()

                    else:
                        error_detail = response.json().get("detail", "Unknown API Error")
                        st.error(f"❌ API Error: {error_detail}")
                        status.update(label="❌ Failed", state="error")

                except requests.exceptions.Timeout:
                    st.error("⚠️ The request timed out. The document might be too large for the current CPU settings.")
                    status.update(label="⚠️ Timeout", state="error")
                except Exception as e:
                    st.error(f"⚠️ Connection Error: Could not connect to the backend. Is FastAPI running? ({e})")
                    status.update(label="⚠️ Error", state="error")