# Gradio interface for AI Career Coach
import gradio as gr
from agents.resume_agent import enhance_resume

# Wrapper function for Gradio
def run_resume_tool(position, resume_text):
    return enhance_resume(resume_text, position)

# Launch Gradio app
gr.Interface(
    fn=run_resume_tool,
    inputs=[
        gr.Textbox(label="Position"),
        gr.Textbox(label="Resume Text", lines=20)
    ],
    outputs=gr.Textbox(label="Enhanced Resume"),
    title="AI Career Coach",
    description="Polish your resume for a specific role using Llama 2 and LangChain."
).launch()
