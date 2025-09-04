import streamlit as st
from crewai import Crew, Task, Agent, LLM
from crewai_tools import SerperDevTool, SeleniumScrapingTool
from dotenv import load_dotenv
import os
import time
import json
from datetime import datetime
import pandas as pd
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

# Set up page configuration
st.set_page_config(
    page_title="AI/ML Use Case Proposer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        color: #155724;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        color: #856404;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        color: #721c24;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'api_keys_valid' not in st.session_state:
        st.session_state.api_keys_valid = None

def validate_api_keys() -> tuple[bool, list]:
    """Validate required API keys"""
    missing_keys = []
    required_keys = ["GEMINI_API_KEY", "SERPER_API_KEY"]
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    return len(missing_keys) == 0, missing_keys

def save_analysis_to_history(company_name: str, result: str, duration: float):
    """Save analysis results to session history"""
    analysis_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'company': company_name,
        'duration': round(duration, 2),
        'result': result,
        'id': len(st.session_state.analysis_history)
    }
    st.session_state.analysis_history.append(analysis_data)

def create_agents_and_tasks(company_name: str, llm: LLM, tool: SerperDevTool, tool_kag: SeleniumScrapingTool):
    """Create and configure agents and tasks"""
    
    # Enhanced agent definitions with more specific roles
    researcher_agent = Agent(
        llm=llm,
        role="Senior Market Research Analyst",
        goal=f"Conduct comprehensive research on {company_name} including industry analysis, competitive landscape, business model, recent developments, and strategic initiatives.",
        backstory="""You are a seasoned market research analyst with 15+ years of experience in technology and business intelligence. 
        You excel at gathering comprehensive information about companies, understanding their market position, 
        identifying growth opportunities, and analyzing industry trends that could impact AI/ML adoption.""",
        tools=[tool, tool_kag],
        memory=True,
        verbose=1,
        max_iter=3,
        max_execution_time=300
    )

    analyst_agent = Agent(
        llm=llm,
        role="AI/ML Strategy Consultant",
        goal="Analyze research findings to identify high-impact, feasible AI/ML use cases tailored to the company's specific context, capabilities, and industry requirements.",
        backstory="""You are an expert AI/ML strategy consultant with deep knowledge of machine learning applications across industries. 
        You specialize in identifying practical AI/ML use cases that align with business objectives, 
        considering technical feasibility, ROI potential, and implementation complexity.""",
        tools=[tool],
        memory=True,
        verbose=1,
        max_iter=3,
        max_execution_time=300
    )

    proposal_agent = Agent(
        llm=llm,
        role="Technical Proposal Specialist",
        goal="Create a comprehensive, actionable AI/ML implementation proposal with detailed recommendations, timelines, and success metrics.",
        backstory="""You are a technical proposal specialist with expertise in AI/ML project planning and implementation. 
        You excel at translating complex technical concepts into clear business proposals, 
        including implementation roadmaps, resource requirements, and measurable success criteria.""",
        tools=[tool],
        memory=True,
        verbose=1,
        max_iter=3,
        max_execution_time=300
    )

    # Enhanced task definitions
    research_task = Task(
        description=f"""Conduct comprehensive research on {company_name}. Your research should include:
        
        1. Company Overview: Size, revenue, business model, key products/services
        2. Industry Analysis: Market trends, competitive landscape, regulatory environment
        3. Technology Stack: Current technology usage, digital maturity level
        4. Recent Developments: Latest news, partnerships, investments, strategic initiatives
        5. Pain Points: Known challenges or inefficiencies in their operations
        6. Growth Areas: Expansion plans, new market opportunities
        
        Provide detailed, actionable insights that will inform AI/ML use case identification.""",
        agent=researcher_agent,
        expected_output="""A comprehensive research report containing:
        - Executive summary of the company
        - Industry context and market position
        - Current technology landscape
        - Identified business challenges and opportunities
        - Strategic priorities and growth areas
        All findings should be well-sourced and include relevant statistics where available.""",
        output_file="research_findings.md"
    )

    analysis_task = Task(
        description=f"""Based on the research findings, identify and analyze potential AI/ML use cases for {company_name}. For each use case:
        
        1. Clearly define the business problem it solves
        2. Explain the AI/ML approach and techniques required
        3. Assess technical feasibility (High/Medium/Low)
        4. Estimate potential business impact and ROI
        5. Identify implementation challenges and risks
        6. Suggest success metrics and KPIs
        7. Provide rough timeline and resource estimates
        
        Prioritize use cases based on impact vs. complexity matrix.""",
        agent=analyst_agent,
        expected_output="""A detailed analysis report containing:
        - 5-8 prioritized AI/ML use cases with comprehensive details
        - Impact vs. complexity assessment for each use case
        - Technical requirements and implementation considerations
        - Risk assessment and mitigation strategies
        - Resource and timeline estimates
        - Success metrics and measurement framework""",
        output_file="use_case_analysis.md"
    )

    proposal_task = Task(
        description=f"""Create a professional AI/ML implementation proposal for {company_name} that includes:
        
        1. Executive Summary highlighting key recommendations
        2. Prioritized use case roadmap with 3 phases (Quick Wins, Medium-term, Long-term)
        3. Detailed implementation plan for top 3 use cases including:
           - Technical architecture and approach
           - Required tools, technologies, and infrastructure
           - Team composition and skill requirements
           - Project timeline with milestones
           - Budget estimates and ROI projections
           - Risk management plan
        4. Success measurement framework
        5. Next steps and recommendations
        
        Format as a professional business proposal.""",
        agent=proposal_agent,
        expected_output="""A comprehensive, professional AI/ML implementation proposal including:
        - Executive summary with key recommendations
        - Detailed 3-phase implementation roadmap
        - Technical specifications and architecture
        - Resource requirements and budget estimates
        - Timeline with key milestones
        - Success metrics and KPI framework
        - Risk assessment and mitigation plan
        - Clear next steps and action items
        
        The proposal should be ready for presentation to C-level executives.""",
        output_file="final_proposal.md"
    )

    return researcher_agent, analyst_agent, proposal_agent, research_task, analysis_task, proposal_task

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI/ML Use Case Proposal Generator</h1>
        <p>Generate comprehensive AI/ML implementation strategies tailored to your company</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key validation
        api_valid, missing_keys = validate_api_keys()
        if not api_valid:
            st.markdown(f"""
            <div class="error-box">
                <strong>⚠️ Missing API Keys:</strong><br>
                {', '.join(missing_keys)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
                ✅ API Keys Configured
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Company input with validation
        company_name = st.text_input(
            "🏢 Company Name",
            placeholder="Enter company name (e.g., Tesla, Microsoft, Netflix)",
            help="Enter the full company name for best results"
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            max_tokens = st.slider("Max Tokens per Agent", 1000, 8000, 5000, 500)
            verbose_output = st.checkbox("Verbose Output", value=True)
            include_competitor_analysis = st.checkbox("Include Competitor Analysis", value=True)
        
        # Run button with validation
        run_button = st.button(
            "🚀 Generate Proposal",
            disabled=not (api_valid and company_name.strip()),
            help="Generate AI/ML use case proposal" if api_valid and company_name.strip() else "Please configure API keys and enter company name"
        )
        
        st.markdown("---")
        
        # Analysis history
        if st.session_state.analysis_history:
            st.subheader("📊 Recent Analyses")
            for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
                with st.expander(f"{analysis['company']} - {analysis['timestamp']}"):
                    st.write(f"**Duration:** {analysis['duration']}s")
                    if st.button(f"View Results", key=f"view_{analysis['id']}"):
                        st.session_state.current_analysis = analysis

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # Metrics and info
        if st.session_state.analysis_history:
            st.markdown("### 📈 Analytics")
            
            total_analyses = len(st.session_state.analysis_history)
            avg_duration = sum(a['duration'] for a in st.session_state.analysis_history) / total_analyses
            
            st.metric("Total Analyses", total_analyses)
            st.metric("Avg Duration", f"{avg_duration:.1f}s")
            
            # Create dataframe for visualization
            if len(st.session_state.analysis_history) > 1:
                df = pd.DataFrame(st.session_state.analysis_history)
                st.line_chart(df.set_index('timestamp')['duration'])

    with col1:
        if run_button and company_name.strip():
            start_time = time.time()
            
            try:
                # Initialize LLM
                llm = LLM(
                    model="gemini/gemini-1.5-flash",
                    verbose=verbose_output,
                    google_api_key=os.getenv("GEMINI_API_KEY"),
                    max_tokens=max_tokens
                )

                # Initialize tools
                tool = SerperDevTool()
                tool_kag = SeleniumScrapingTool()

                # Create agents and tasks
                with st.status("🤖 Initializing AI Agents...", expanded=True) as status:
                    researcher_agent, analyst_agent, proposal_agent, research_task, analysis_task, proposal_task = create_agents_and_tasks(
                        company_name, llm, tool, tool_kag
                    )
                    status.update(label="✅ Agents Initialized Successfully!", state="complete")

                # Execute workflow
                with st.status("🔄 Running AI Analysis Workflow...", expanded=True) as status:
                    crew = Crew(
                        agents=[researcher_agent, analyst_agent, proposal_agent],
                        tasks=[research_task, analysis_task, proposal_task],
                        verbose=verbose_output,
                        max_rpm=10,
                        share_crew=False
                    )
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate progress updates
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        if i < 30:
                            status_text.text("🔍 Researching company and industry...")
                        elif i < 70:
                            status_text.text("🧠 Analyzing AI/ML opportunities...")
                        else:
                            status_text.text("📝 Generating final proposal...")
                        time.sleep(0.1)
                    
                    result = crew.kickoff(inputs={'company': company_name})
                    status.update(label="✅ Analysis Complete!", state="complete")

                end_time = time.time()
                duration = end_time - start_time

                # Save to history
                save_analysis_to_history(company_name, str(result), duration)

                # Display results
                st.success(f"✅ Analysis completed in {duration:.2f} seconds!")
                
                # Results tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📋 Final Proposal", "🔍 Research Findings", "📊 Use Case Analysis", "💾 Export"])
                
                with tab1:
                    st.subheader(f"AI/ML Implementation Proposal for {company_name}")
                    st.markdown(str(result))
                
                with tab2:
                    try:
                        with open("research_findings.md", "r", encoding='utf-8') as f:
                            st.markdown(f.read())
                    except FileNotFoundError:
                        st.warning("Research findings file not found")
                
                with tab3:
                    try:
                        with open("use_case_analysis.md", "r", encoding='utf-8') as f:
                            st.markdown(f.read())
                    except FileNotFoundError:
                        st.warning("Use case analysis file not found")
                
                with tab4:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        try:
                            with open("final_proposal.md", "r", encoding='utf-8') as f:
                                st.download_button(
                                    label="📄 Download Full Proposal",
                                    data=f.read(),
                                    file_name=f"{company_name.replace(' ', '_')}_AI_ML_Proposal.md",
                                    mime="text/markdown"
                                )
                        except FileNotFoundError:
                            pass
                    
                    with col2:
                        try:
                            with open("research_findings.md", "r", encoding='utf-8') as f:
                                st.download_button(
                                    label="🔍 Download Research",
                                    data=f.read(),
                                    file_name=f"{company_name.replace(' ', '_')}_Research.md",
                                    mime="text/markdown"
                                )
                        except FileNotFoundError:
                            pass
                    
                    with col3:
                        try:
                            with open("use_case_analysis.md", "r", encoding='utf-8') as f:
                                st.download_button(
                                    label="📊 Download Analysis",
                                    data=f.read(),
                                    file_name=f"{company_name.replace(' ', '_')}_Analysis.md",
                                    mime="text/markdown"
                                )
                        except FileNotFoundError:
                            pass

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

        elif st.session_state.current_analysis:
            # Display selected historical analysis
            analysis = st.session_state.current_analysis
            st.subheader(f"📋 Analysis: {analysis['company']}")
            st.write(f"**Generated:** {analysis['timestamp']}")
            st.write(f"**Duration:** {analysis['duration']}s")
            st.markdown("---")
            st.markdown(analysis['result'])
            
            if st.button("🔄 Run New Analysis"):
                st.session_state.current_analysis = None
                st.rerun()

        else:
            # Welcome message
            st.markdown("""
            ### 👋 Welcome to the AI/ML Use Case Proposal Generator!
            
            This tool helps you discover and prioritize AI/ML opportunities for any company by:
            
            1. **🔍 Researching** the company's industry, business model, and current challenges
            2. **🧠 Analyzing** potential AI/ML applications and their feasibility
            3. **📋 Generating** a comprehensive implementation proposal with timelines and ROI projections
            
            **To get started:**
            - Ensure your API keys are configured (check sidebar)
            - Enter a company name
            - Click "Generate Proposal" and wait for the analysis
            
            The process typically takes 2-5 minutes depending on the complexity of the company and industry.
            """)
            
            # Example companies
            st.markdown("### 💡 Example Companies to Try:")
            example_companies = ["Tesla", "Netflix", "Shopify", "Zoom", "Peloton", "DoorDash"]
            
            cols = st.columns(3)
            for i, company in enumerate(example_companies):
                with cols[i % 3]:
                    if st.button(f"🏢 {company}", key=f"example_{company}"):
                        st.session_state.example_company = company
                        st.rerun()

if __name__ == "__main__":
    main()
