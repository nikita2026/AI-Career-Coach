# LangChain agent for generating tailored cover letters using Llama 2

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.model_loader import load_local_llm

# Load the locally hosted Llama 2 model
llm = load_local_llm()

# Define the prompt template for cover letter generation
cover_letter_prompt = PromptTemplate(
    input_variables=["resume_text", "job_description"],
    template="""
Using the resume below and the job description provided, generate a personalized cover letter that highlights relevant skills, experience, and enthusiasm for the role.

Resume:
{resume_text}

Job Description:
{job_description}

Cover Letter:
"""
)

# Create the LangChain pipeline
cover_letter_chain = LLMChain(llm=llm, prompt=cover_letter_prompt)

# Function to generate the cover letter
def generate_cover_letter(resume_text, job_description):
    return cover_letter_chain.run({
        "resume_text": resume_text,
        "job_description": job_description
    })
