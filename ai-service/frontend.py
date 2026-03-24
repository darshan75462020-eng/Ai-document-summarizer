import streamlit as st
import requests
import base64

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Document Summarizer", 
    page_icon="📄", 
    layout="wide"
)

# --- MODERN UI STYLING (FIXED VISIBILITY) ---
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    
    /* Metrics Card Styling */
    .stMetric {
        background-color: white !important;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* FIX: Force metric text visibility */
    [data-testid="stMetricValue"] {
        color: #1e293b !important;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600 !important;
    }

    /* Results Card Styling */
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        line-height: 1.6;
        color: #1e293b;
        transition: transform 0.2s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
    }
    .keyword-tag {
        background: #e0f2fe;
        color: #0369a1;
        padding: 6px 12px;
        border-radius: 20px;
        margin: 4px;
        display: inline-block;
        font-weight: 500;
        font-size: 0.85em;
        border: 1px solid #bae6fd;
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def display_pdf(file_bytes):
    """Encodes and displays a PDF preview."""
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ System Status")
    st.success("Model: **FLAN-T5-Base**")
    st.info("Extraction: **Hybrid Extractive-Abstractive**")
    st.divider()
    st.caption("v2.1 - UI Color Fix")

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>📄 AI Document Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Advanced NLP Pipeline for Professional Documents</p>", unsafe_allow_html=True)
st.divider()

# --- UPLOAD SECTION ---
uploaded_file = st.file_uploader("📂 Drop your PDF here", type=["pdf"], help="Best for resumes and research papers.")

if uploaded_file:
    col_pre, col_act = st.columns([1, 1])
    
    with col_pre:
        with st.expander("🔍 Preview Document", expanded=True):
            display_pdf(uploaded_file.getvalue())

    with col_act:
        st.write("### 🚀 Actions")
        if st.button("Generate AI Insights", use_container_width=True, type="primary"):
            with st.status("🧠 Processing...", expanded=True) as status:
                st.write("Connecting to FastAPI Backend...")
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                
                try:
                    # Increased timeout for local CPU processing
                    response = requests.post("http://127.0.0.1:8000/summarize", files=files, timeout=180)
                    
                    if response.status_code == 200:
                        data = response.json()
                        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                        
                        summary = data.get("summary", "")
                        bullets = data.get("bullets", "")
                        keywords = data.get("keywords", "")

                        # --- METRICS DASHBOARD (UI FIX APPLIED) ---
                        st.divider()
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Est. Reading Time", f"{max(1, len(summary.split()) // 200)} min", "⏱️")
                        m2.metric("Insights Depth", "High", "📈")
                        m3.metric("Key Terms", len(keywords.split(", ")) if keywords else 0, "🏷️")

                        # --- RESULTS TABS ---
                        tab1, tab2, tab3 = st.tabs(["📌 Summary", "📊 Takeaways", "🏷 Keywords"])

                        with tab1:
                            st.markdown(f'<div class="card">{summary}</div>', unsafe_allow_html=True)

                        with tab2:
                            formatted_bullets = bullets.replace("\n", "<br>")
                            st.markdown(f'<div class="card">{formatted_bullets}</div>', unsafe_allow_html=True)

                        with tab3:
                            if keywords:
                                kw_list = keywords.split(", ")
                                kw_html = "".join([f'<span class="keyword-tag">{k}</span>' for k in kw_list])
                                st.markdown(f'<div class="card">{kw_html}</div>', unsafe_allow_html=True)
                            else:
                                st.write("No keywords identified.")

                        # --- EXPORT ---
                        st.divider()
                        st.download_button(
                            label="📥 Download TXT Report",
                            data=f"SUMMARY:\n{summary}\n\nKEY POINTS:\n{bullets}\n\nKEYWORDS:\n{keywords}",
                            file_name=f"AI_Report_{uploaded_file.name.split('.')[0]}.txt",
                            use_container_width=True
                        )

                    else:
                        st.error(f"Error: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")