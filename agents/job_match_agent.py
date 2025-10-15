# LangChain agent for matching resume content to job roles using Llama 2

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.model_loader import load_local_llm

# Load the locally hosted LLM
llm = load_local_llm()

# Define the prompt template for job matching
job_match_prompt = PromptTemplate(
    input_variables=["resume_text"],
    template="""
Based on the resume below, suggest 3 job roles or titles that best match the candidate's skills, experience, and interests. Include a brief rationale for each suggestion.

Resume:
{resume_text}

Suggested Job Matches:
"""
)

# Create the LangChain pipeline
job_match_chain = LLMChain(llm=llm, prompt=job_match_prompt)

# Function to generate job match suggestions
def suggest_job_matches(resume_text):
    return job_match_chain.run({"resume_text": resume_text})
