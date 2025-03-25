import streamlit as st
import openai
import PyPDF2
import docx
import io
import os
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import faiss
import numpy as np
import pickle
from openai import OpenAI

# OpenAI API Configuration
openai.api_key = st.secrets["OPENAI_API_KEY"]

class SOPComplianceAnalyzer:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
    
    def extract_text_from_document(self, uploaded_file):
        """
        Extract text from PDF or DOCX files
        """
        if uploaded_file.type == 'application/pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = " ".join([page.extract_text() for page in pdf_reader.pages])
            return text
        
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            doc = docx.Document(uploaded_file)
            text = " ".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        
        else:
            st.error("Unsupported file type. Please upload a PDF or DOCX file.")
            return None

    def get_embedding(self, text, model="text-embedding-3-small"):
        response = self.client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
        
    def get_local_regulations(self, location: str, query: str, top_k: int = 5):
        # Force location to be camel case for now and match
        location_key = {
            "New York, NY": "new_york",
            "Chicago, IL": "chicago",
            "Los Angeles, CA": "los_angeles"
        }
        location = location_key.get(location)
        # Load index and metadata
        index = faiss.read_index("regulations.index")
        with open("regulations_meta.pkl", "rb") as f:
            docs, metadata = pickle.load(f)
        
        # Embed query
        query_embedding = self.get_embedding(query, model="text-embedding-3-small")
        D, I = index.search(np.array([query_embedding]).astype("float32"), top_k)

        # Filter by location
        results = []
        for idx in I[0]:
            if idx < len(metadata) and metadata[idx]['location'].lower() == location.lower():
                results.append(docs[idx])
        
        return results  # List of regulation text snippets
    
    def analyze_sop_compliance(self, sop_text, facility_location, manufacturing_type):
        """
        Use OpenAI API to analyze SOP compliance
        """

        # Get local regulations (if available)
        local_regulations = self.get_local_regulations(facility_location, sop_text)
        if local_regulations:
            regulations_context = "\n\n".join(local_regulations)
            regulations_note = f"\n\nRelevant Local Regulations for {facility_location}:\n{regulations_context}"
        else:
            regulations_note = "\n\nNote: No relevant local regulations were found for this location."
        prompt = f"""
        Comprehensive Regulatory Compliance Analysis for Standard Operating Procedure (SOP)

        Context:
        - Facility Location: {facility_location}
        - Manufacturing Type: {manufacturing_type}

        Current SOP Text:
        {sop_text}

        Provide a detailed analysis with:
        1. Potential Compliance Issues
           - Categorize issues by severity (❌ Critical, ⛔️ Major, ⚠️ Minor)
           - Reference specific regulatory standards if relevant
           - Explain potential risks and implications

        2. Detailed Recommendations
           - Specific actions to address each identified issue
           - Best practices for compliance
           - Suggested improvements to current SOP structure

        Focus on:
        - Regulatory alignment
        - Risk mitigation
        - Operational efficiency
        - Industry-specific compliance standards
        - Ensuring that any relevant location-specific regulations are addressed

        Note: If you cite a location-specific regulation, please use "[📍{facility_location}-specific]" to identify it as such
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a senior regulatory compliance expert specializing in pharmaceutical manufacturing SOPs with extensive knowledge of global regulatory frameworks."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096
        )
        
        return response.choices[0].message.content
    
    def turbocharge_sop(self, original_sop, compliance_issues):
        todays_date = datetime.today().strftime('%Y-%m-%d')

        # Generate an improved SOP based on compliance analysis
        
        prompt = f"""
        Objective: Generate a Comprehensive, Compliant Standard Operating Procedure (SOP)

        Original SOP:
        {original_sop}

        Compliance Issues and Recommendations:
        {compliance_issues}

        Instructions for SOP Revision:
        1. Integrate all compliance recommendations
        2. Maintain the core structure of the original SOP
        3. Clearly highlight and explain new compliance-driven modifications
        4. Ensure the document is:
           - Comprehensive
           - Clear
           - Aligned with current regulatory standards
           - Actionable for facility staff

        Output Format:
        - Use markdown for formatting
        - Include sections:
          * Purpose
          * Scope
          * Definitions
          * Responsibilities
          * Procedure Steps
          * Compliance Notes
          * References

        Critical:
        - Make sure that document is user ready - do not include any commentary at the end. Stick to business

        For reference, today's data is {todays_date}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert SOP writer specializing in creating compliant, clear, and comprehensive procedural documents for pharmaceutical manufacturing."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096
        )
        
        return response.choices[0].message.content
    
    def generate_quiz_questions(self, sop_text):
        prompt = f"""
        Based on the following SOP, generate 5 short-answer quiz questions that test understanding of its content, especially compliance, procedures, and safety practices. Return the output in JSON format like this:

        [
            {{"question": "..." }},
            ...
        ]

        SOP:
        {sop_text}
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional instructional designer for pharmaceutical SOPs."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def grade_quiz_responses(self, sop_text, qa_pairs):
        grading_prompt = f"""
        You are a regulatory training evaluator. Rigorously grade each of the following short-answer responses (0=incorrect, 1=partially correct, 2=correct) based on the SOP provided. If you sense that the employee is not familiar with the SOP, provide constructive feedback to guide them towards the correct answer. However, if you feel that the employee is intentionally providing incorrect answers or thinks that this is a joke, mark them as such and be forceful in your tone.

        Format response as JSON:
        [
            {{"question": "...", "answer": "...", "score": 0, "feedback": "..." }},
            ...
        ]

        SOP:
        {sop_text}

        Q&A Pairs:
        {qa_pairs}
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a compliance training evaluator."},
                {"role": "user", "content": grading_prompt}
            ]
        )
        return response.choices[0].message.content

def create_pdf_from_markdown(markdown_text, filename):
    """
    Convert markdown to PDF using ReportLab
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Convert markdown-like text to PDF-compatible paragraphs
    story = []
    for line in markdown_text.split('\n'):
        if line.startswith('# '):
            story.append(Paragraph(line[2:], styles['Title']))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], styles['Heading2']))
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], styles['Heading3']))
        elif line.strip():
            story.append(Paragraph(line, styles['Normal']))
        story.append(Spacer(1, 12))

    def set_metadata(canvas, doc):
        canvas.setTitle(filename)
    
    doc.build(story, onFirstPage=set_metadata)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

def main():
    st.set_page_config(page_title="Project Qualia", page_icon="⚡️")
    
    st.title("Project Qualia")
    st.subheader("Turbocharge SOP Compliance ⚡️")
    
    # Initialize session state for storing SOP and compliance report
    if 'original_sop' not in st.session_state:
        st.session_state.original_sop = None
    if 'compliance_report' not in st.session_state:
        st.session_state.compliance_report = None
    if 'turbocharged_sop' not in st.session_state:
        st.session_state.turbocharged_sop = None
    
    # Facility Information
    st.header("Facility Information")
    facility_location = st.selectbox(
        "Facility Location", 
        [
            "New York, NY", 
            "Chicago, IL", 
            "Los Angeles, CA"
        ]
    )
    manufacturing_type = st.selectbox(
        "Manufacturing Type", 
        [
            "Pharmaceutical Drugs", 
            "Medical Devices", 
            "Biologics", 
            "Vaccines", 
            "Diagnostic Reagents",
            "Pharmaceuticals API",
            "Biotechnology Products"
        ]
    )
    
    # File upload section
    st.header("Upload SOP Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file", 
        type=['pdf', 'docx']
    )

    if uploaded_file is not None:
        base_filename = os.path.splitext(uploaded_file.name)[0]
    
    # First Analysis Stage
    if st.button("🔭 Analyze SOP") and uploaded_file is not None:
        # Initialize compliance analyzer
        analyzer = SOPComplianceAnalyzer(st.secrets["OPENAI_API_KEY"])
        
        # Extract text from uploaded document
        sop_text = analyzer.extract_text_from_document(uploaded_file)
        
        if sop_text:
            # Store original SOP in session state
            st.session_state.original_sop = sop_text
            
            # Perform compliance analysis
            with st.spinner('👩‍🔬 Analyzing SOP for Compliance...'):
                compliance_report = analyzer.analyze_sop_compliance(
                    sop_text, 
                    facility_location, 
                    manufacturing_type
                )
            
            # Store compliance report in session state
            st.session_state.compliance_report = compliance_report

             # === Stage 2: Turbocharge SOP ===
            # Now automatically turbocharge the SOP
            if st.session_state.original_sop and st.session_state.compliance_report:
                analyzer = SOPComplianceAnalyzer(st.secrets["OPENAI_API_KEY"])
                with st.spinner('🏎️ Turbocharging Your SOP...'):
                    turbocharged_sop = analyzer.turbocharge_sop(
                        st.session_state.original_sop, 
                        st.session_state.compliance_report
                    )
                st.session_state.turbocharged_sop = turbocharged_sop


    # === Persistent Display via Tabs ===
    if st.session_state.original_sop or st.session_state.compliance_report or 'turbocharged_sop' in st.session_state:
        st.markdown("---")
        st.header("📋 SOP Studio")

        tabs = st.tabs(["🔎 Compliance Analysis", "🚀 Turbocharged SOP", "📚 Classroom"])

        unrendered_tab_message = "⚠️ Let’s analyze your SOP and turbocharge first."

        with tabs[0]:
            if st.session_state.compliance_report:
                try:
                    if st.session_state.compliance_report:
                        st.markdown(st.session_state.compliance_report)

                    col1, col2 = st.columns(2)
                    with col1:
                        compliance_pdf_bytes = create_pdf_from_markdown(
                            st.session_state.compliance_report, "compliance_analysis.pdf"
                        )
                        st.download_button(
                            label="Download PDF",
                            data=compliance_pdf_bytes,
                            file_name=f"{base_filename}_compliance_analysis.pdf",
                            mime="application/pdf"
                        )

                    with col2:
                        st.download_button(
                            label="Download Raw Markdown",
                            data=st.session_state.compliance_report,
                            file_name=f"{base_filename}_compliance_analysis.md",
                            mime="text/markdown"
                        )
                except:
                    st.info(unrendered_tab_message)
            else:
                st.info(unrendered_tab_message)

        with tabs[1]:
            if 'turbocharged_sop' in st.session_state:
                try:
                    if st.session_state.turbocharged_sop:
                        st.markdown(st.session_state.turbocharged_sop)

                    col1, col2 = st.columns(2)
                    with col1:
                        turbo_pdf = create_pdf_from_markdown(
                            st.session_state.turbocharged_sop, "turbocharged_sop.pdf"
                        )
                        st.download_button(
                            label="Download PDF",
                            data=turbo_pdf,
                            file_name=f"{base_filename}_turbocharged_sop.pdf",
                            mime="application/pdf"
                        )
                    with col2:
                        st.download_button(
                            label="Download Raw Markdown",
                            data=st.session_state.turbocharged_sop,
                            file_name=f"{base_filename}_turbocharged_sop.md",
                            mime="text/markdown"
                        )
                except:
                    st.info(unrendered_tab_message)
            else:
                st.info(unrendered_tab_message)
        
        with tabs[2]:
            if 'turbocharged_sop' not in st.session_state and st.session_state.turbocharged_sop is None:
                st.info("Please turbocharge your SOP first before accessing the classroom.")
            else:
                analyzer = SOPComplianceAnalyzer(st.secrets["OPENAI_API_KEY"])
                
                if st.session_state.turbocharged_sop and 'quiz_questions' not in st.session_state:
                    with st.spinner("🧠 Generating Quiz Questions..."):
                        quiz_json = analyzer.generate_quiz_questions(st.session_state.turbocharged_sop).replace("```json", "").replace("```", "")
                        import json
                        try:
                            st.session_state.quiz_questions = json.loads(quiz_json)
                        except:
                            st.error("Error parsing quiz questions.")
                
                if 'quiz_questions' in st.session_state:
                    st.subheader("📝 Short Answer Quiz")
                    responses = []

                    for i, q in enumerate(st.session_state.quiz_questions):
                        st.markdown(f"**Q{i+1}: {q['question']}**")
                        response = st.text_area(f"Your Answer {i+1}", key=f"answer_{i}")
                        responses.append({
                            "question": q['question'],
                            "answer": response
                        })

                    if st.button("📤 Submit Quiz"):
                        with st.spinner("🔍 Grading..."):
                            import json
                            grading_json = json.dumps(responses)
                            graded = analyzer.grade_quiz_responses(st.session_state.turbocharged_sop, grading_json)
                            graded = graded.replace("```json", "").replace("```", "")
                            try:
                                graded_results = json.loads(graded)
                                st.subheader("📊 Results")
                                for i, result in enumerate(graded_results):
                                    st.markdown(f"**Q{i+1}: {result['question']}**")
                                    st.markdown(f"**Your Answer:** {result['answer']}")
                                    st.markdown(f"**Score:** {result['score']}/2")
                                    st.markdown(f"**Feedback:** {result['feedback']}")
                                    st.markdown("---")
                            except:
                                st.error("Failed to parse grading response.")

if __name__ == "__main__":
    main()