# Resume enhancement agent using LangChain and Llama 2
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.model_loader import load_local_llm

# Load the local LLM
llm = load_local_llm()

# Define prompt template for resume polishing
resume_prompt = PromptTemplate(
    input_variables=["resume_text", "position"],
    template="Improve this resume for a {position} role:\n\n{resume_text}"
)

# Create LangChain pipeline
resume_chain = LLMChain(llm=llm, prompt=resume_prompt)

# Function to run the enhancement
def enhance_resume(resume_text, position):
    return resume_chain.run({"resume_text": resume_text, "position": position})
