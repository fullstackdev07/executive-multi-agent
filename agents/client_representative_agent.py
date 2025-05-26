import os
import re
from typing import Optional, List
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import fitz  # PyMuPDF for PDF reading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClientRepresentativeAgent:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OpenAI API key not found in environment variables.")
            raise ValueError("OPENAI_API_KEY not found. Set environment variable.")

        try:
            self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7, max_tokens=3000)
        except Exception as e:
            logger.exception(f"Error initializing OpenAI LLM: {e}")
            raise

        self._create_extraction_prompt()
        self._create_feedback_prompt()
        self._create_chains()

    def _create_extraction_prompt(self):
        self.extraction_template = PromptTemplate(
            input_variables=["user_input"],
            template="""
From the following statement, infer the client's characteristics:

Statement:
{user_input}

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

    def run(self, user_input: str, files: Optional[List[str]] = None) -> str:
        logger.info(f"Agent running with input: {user_input[:100]}...")

        # 1. Input sanity checks
        if not user_input or len(user_input.strip()) < 10:
            warning = "⚠️ It seems your input is too short or unclear. Please provide a more meaningful client description."
            logger.warning(warning)
            return warning

        if not re.search(r"[a-zA-Z]{3,}", user_input):
            warning = "⚠️ I couldn't understand your input. Please describe the client using clear and meaningful language."
            logger.warning(warning)
            return warning

        # 2. Combine input with transcript files if provided
        combined_input = user_input.strip()
        if files:
            logger.info(f"Processing transcript files: {files}")
            for file_path in files:
                try:
                    ext = os.path.splitext(file_path)[-1].lower()
                    if ext == ".pdf":
                        with fitz.open(file_path) as doc:
                            pdf_text = "".join(page.get_text() for page in doc)
                        combined_input += "\n" + pdf_text
                        logger.info(f"Successfully extracted text from PDF: {file_path}")
                    elif ext == ".txt":
                        with open(file_path, "r", encoding="utf-8") as f:
                            txt_text = f.read()
                        combined_input += "\n" + txt_text
                        logger.info(f"Successfully read text from TXT: {file_path}")
                    else:
                        logger.warning(f"Skipping unsupported file type: {file_path}")
                        if self.verbose:
                            print(f"Skipping unsupported file type: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    if self.verbose:
                        print(f"Failed to read file {file_path}: {e}")

        logger.info(f"Combined input after file processing: {combined_input[:200]}...")

        # 3. Run extraction chain
        try:
            extracted_json_str = self.extraction_chain.run({"user_input": combined_input})
            logger.info(f"Extracted JSON string: {extracted_json_str}")
            if self.verbose:
                print("Extracted Traits:\n", extracted_json_str)
        except Exception as extraction_error:
            logger.error(f"Error during extraction chain run: {extraction_error}")
            raise ValueError(f"Error during extraction: {extraction_error}")

        # 4. Parse JSON
        try:
            extracted_data = json.loads(extracted_json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extracted client data (JSONDecodeError): {e}, raw string: {extracted_json_str}")
            raise ValueError(f"Failed to parse extracted client data: {e}. Raw string: {extracted_json_str}")
        except Exception as e:
            logger.error(f"Failed to parse extracted client data (General Exception): {e}")
            raise ValueError(f"Failed to parse extracted client data: {e}")

        # 5. Generate feedback
        feedback_input = {
            **extracted_data,
            "input_to_review": combined_input.strip()
        }

        try:
            feedback_response = self.feedback_chain.run(feedback_input)
            logger.info("Feedback generated successfully.")
            return feedback_response
        except Exception as feedback_error:
            logger.error(f"Error during feedback chain run: {feedback_error}")
            raise ValueError(f"Error during feedback generation: {feedback_error}")