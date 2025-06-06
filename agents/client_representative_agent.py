import os
from typing import List
from dotenv import load_dotenv
from fastapi import UploadFile
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import fitz  # PyMuPDF
import json
import logging
from utils.agent_output_formatter import format_agent_output

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def is_input_valid(text: str, min_length: int = 10) -> bool:
    return bool(text and len(text.strip()) >= min_length)


class ClientRepresentativeAgent:
    def __init__(self, verbose: bool = False):
        self.name = "Client Representative Agent"
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY not found.")
            raise ValueError("OPENAI_API_KEY not found.")
        
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0.7,
            model_name="gpt-4"
        )

        self._create_prompt()
        self._create_chain()

    def _create_prompt(self):
        template = """
You are an experienced executive search evaluator. Based on the candidate input and any additional documents, produce a structured evaluation. Do not assume a default persona or background.

Instructions:
1. Use only the information provided below.
2. Analyze any documents or candidate text for leadership capability, cultural fit, and strategic experience.
3. Write in a professional tone with clear sections like "Summary Evaluation," "Strengths," "Risks," and "Recommendation."

Candidate Information:
{candidate_information_text}

Additional Context (if any):
{ceo_job_description}
"""
        self.prompt_template = PromptTemplate(
            input_variables=["candidate_information_text", "ceo_job_description"],
            template=template,
        )

    def _create_chain(self):
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=self.verbose)

    def _extract_text_from_file(self, file: UploadFile) -> str:
        ext = os.path.splitext(file)[-1].lower()
        try:
            if ext == ".pdf":
                doc = fitz.open(stream=file.file.read(), filetype="pdf")
                return "".join([page.get_text() for page in doc])
            elif ext == ".txt":
                return file.file.read().decode("utf-8")
            elif ext == ".json":
                data = json.load(file.file)
                if isinstance(data, dict):
                    return json.dumps(data, indent=2)
                return "\n".join(str(item) for item in data)
            else:
                logger.warning(f"Unsupported file type: {file.filename}")
                return ""
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
            return ""

    def _extract_all_files_text(self, files: List[UploadFile]) -> str:
        texts = []
        for file in files:
            text = self._extract_text_from_file(file)
            if text:
                texts.append(f"--- File: {file.filename} ---\n{text}")
        return "\n\n".join(texts)

    def run(self, user_input: str, files: List[str]) -> dict:
        candidate_info_parts = []
        used_files = files if files else []

        if user_input and is_input_valid(user_input.strip(), min_length=10):
            candidate_info_parts.append(f"Manual Input:\n{user_input.strip()}")

        if files:
            file_text = self._extract_all_files_text(files)
            if file_text and is_input_valid(file_text, min_length=20):
                candidate_info_parts.append(f"Extracted from Files:\n{file_text}")

        if not candidate_info_parts:
            error_message = "Error: Candidate information is insufficient. Provide valid manual input or supported files."
            return format_agent_output(
                title="Client Feedback Error",
                sections=[],
                summary=error_message,
                next_steps=["Provide more detailed input or upload relevant files."],
                used_files=used_files,
                meta={"agent": "Client Rep Agent"}
            )

        final_text = "\n\n".join(candidate_info_parts)

        if self.verbose:
            logger.info(f"{self.name}: Combined input for evaluation: {final_text[:300]}...")

        try:
            input_data = {
                "candidate_information_text": final_text,
                "ceo_job_description": ""
            }

            if hasattr(self.chain, 'invoke'):
                res_dict = self.chain.invoke(input_data)
                response_raw = res_dict.get('text', str(res_dict))
            else:
                response_raw = self.chain.run(input_data)

            feedback = response_raw.strip()
        except Exception as e:
            feedback = f"Error: Evaluation failed due to: {str(e)}"
            return format_agent_output(
                title="Client Feedback Error",
                sections=[],
                summary=feedback,
                next_steps=["Try again with more detailed input or files."],
                used_files=used_files,
                meta={"agent": "Client Rep Agent"}
            )

        sections = [{"header": "Client Feedback", "content": feedback}]
        title = "Client Representative Feedback"
        summary = f"This feedback was generated using your input and the following files: {', '.join(used_files) if used_files else 'None'}."
        next_steps = [
            "Generate an interview report for a candidate",
            "Review another document with this client persona"
        ]
        meta = {"agent": "Client Rep Agent"}
        return format_agent_output(
            title=title,
            sections=sections,
            summary=summary,
            next_steps=next_steps,
            used_files=used_files,
            meta=meta
        )