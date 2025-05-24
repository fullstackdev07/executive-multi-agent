import os
from typing import Optional
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

class ClientRepresentativeAgent:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            print("OpenAI API key not found.")
        self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7)
        self._create_extraction_prompt()
        self._create_feedback_prompt()
        self._create_chains()

    def _create_extraction_prompt(self):
        self.extraction_template = PromptTemplate(
            input_variables=["input_statement"],
            template="""
    From the following statement, infer the client's characteristics:

    Statement:
    {input_statement}

    Return as JSON:
    {{
        "client_persona": "...",
        "client_priorities": "...",
        "client_values": "...",
        "client_tone": "..."
    }}
    """
        )

    def _create_feedback_prompt(self):
        self.feedback_template = PromptTemplate(
            input_variables=[
                "client_persona",
                "client_priorities",
                "client_values",
                "client_tone",
                "input_to_review",
            ],
            template="""
    You are a Client Representative in a project. Review the following input as if you were the client.

    Client Persona: {client_persona}  
    Client Priorities: {client_priorities}  
    Client Values: {client_values}  
    Client Tone and Style: {client_tone}  

    Input to Review:  
    {input_to_review}

    Your job:
    - Emulate the client's tone
    - Reflect their priorities and values
    - Offer constructive feedback
    - Ask for clarity or express concerns
    - Be specific and helpful

    Write a thorough response of approximately **500 words** that reflects how the client would engage with this input.
    Include:
    - Clear feedback on alignment with goals and tone
    - Specific strengths and weaknesses
    - Clarifying questions if needed
    - Suggestions for improvement or refinement

    Client Representative Feedback:
    """
        )

    def _create_chains(self):
        self.extraction_chain = LLMChain(llm=self.llm, prompt=self.extraction_template)
        self.feedback_chain = LLMChain(llm=self.llm, prompt=self.feedback_template)
            
    def run(self, input_statement: str, files: Optional[list] = None) -> str:
        combined_input = input_statement or ""
        
        # If files are provided, read and append their content
        if files:
            for file_path in files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_text = f.read()
                    combined_input += "\n" + file_text
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to read file {file_path}: {e}")

        # Step 1: Extract client persona, tone, values, priorities
        extracted_json_str = self.extraction_chain.run({"input_statement": combined_input})
        if self.verbose:
            print("Extracted Traits:\n", extracted_json_str)

        try:
            extracted_data = eval(extracted_json_str)
        except Exception as e:
            raise ValueError(f"Failed to parse extracted client data: {e}")

        # Step 2: Generate feedback using inferred traits
        feedback_input = {
            **extracted_data,
            "input_to_review": combined_input.strip()
        }

        return self.feedback_chain.run(feedback_input)