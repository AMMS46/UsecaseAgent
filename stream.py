import streamlit as st
from crewai import Crew, Task, Agent, LLM
from crewai_tools import SerperDevTool, SeleniumScrapingTool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up page configuration
st.set_page_config(
    page_title="AI/ML Use Case Proposer",
    page_icon="🤖",
    layout="wide"
)

# Initialize tools
tool = SerperDevTool()
tool_kag = SeleniumScrapingTool()

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    company_name = st.text_input(" ")
    run_button = st.button("Generate Proposal")
    
# Main content area
st.title("AI/ML Use Case Proposal Generator")

if run_button:
    # Validate environment variables
    if not os.getenv("GEMINI_API_KEY") or not os.getenv("SERPER_API_KEY"):
        st.error("Missing API keys in environment variables!")
        st.stop()

    # Initialize LLM
    llm = LLM(
        model="gemini/gemini-1.5-flash",
        verbose=True,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        max_tokens=5000
    )

    # Define agents
    with st.status("Initializing AI Agents...", expanded=True) as status:
        researcher_agent = Agent(
            llm=llm,
            role="Researcher: Gathers company and industry insights.",
            goal="Collect data about the company's industry segment, key offerings, and strategic focus areas.",
            backstory="Experienced market researcher specializing in analyzing industries and companies.",
            tools=[tool, tool_kag],
            memory=True,
            verbose=1
        )

        analyst_agent = Agent(
            llm=llm,
            role="Analyst: Processes research findings.",
            goal="Analyze research data to identify AI/ML use cases.",
            backstory="Seasoned data analyst with expertise in AI/ML applications.",
            tools=[tool],
            memory=True,
            verbose=1
        )

        proposal_agent = Agent(
            llm=llm,
            role="Proposal Writer: Creates final report.",
            goal="Generate detailed proposal with prioritized use cases.",
            backstory="Professional technical writer specializing in AI/ML concepts.",
            tools=[tool],
            memory=True,
            verbose=1
        )
        status.update(label="Agents Initialized!", state="complete")

    # Define tasks
    with st.status("Creating Workflow Tasks...", expanded=True) as status:
        task1 = Task(
            description=f"Conduct in-depth research on {company_name}",
            agent=researcher_agent,
            expected_output="Detailed summary of industry segment, key offerings, and strategic focus areas.",
            output_file="task1output.txt"
        )

        task2 = Task(
            description="Analyze research findings to identify AI/ML use cases",
            agent=analyst_agent,
            expected_output="List of actionable AI/ML use cases with feasibility analysis.",
            output_file="task2output.txt"
        )

        task3 = Task(
            description="Create final proposal listing top AI/ML use cases",
            agent=proposal_agent,
            expected_output="Structured report with prioritized use cases and references.",
            output_file="final_proposal.txt"
        )
        status.update(label="Tasks Configured!", state="complete")

    # Execute crew
    with st.status("Running AI Workflow...", expanded=True) as status:
        crew = Crew(
            agents=[researcher_agent, analyst_agent, proposal_agent],
            tasks=[task1, task2, task3],
            verbose=1
        )
        
        result = crew.kickoff(inputs={'company': company_name})
        status.update(label="Analysis Complete!", state="complete")

    # Display results
    st.subheader("Final Proposal")
    st.markdown(result)

    # Add download buttons
    with open("final_proposal.txt", "r") as f:
        st.download_button(
            label="Download Full Report",
            data=f,
            file_name="ai_ml_proposal.md",
            mime="text/markdown"
        )

    # Show intermediate outputs
    with st.expander("View Research Findings"):
        with open("task1output.txt", "r") as f:
            st.write(f.read())

    with st.expander("View Analysis Results"):
        with open("task2output.txt", "r") as f:
            st.write(f.read())
