import os , re
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import json
from utils.agent_prompt_utils import inject_guidance

load_dotenv()

class ClientRepresentativeCreatorAgent:
    def __init__(self, verbose: bool = False):
        self.name = "Client Representative Prompt Creator"
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7)
        self._create_prompt()
        self._create_chain()

    def _create_prompt(self) -> None:
        template = """You will be given a description of a specific client, including how they think, what they care about, and how they speak. Your task is to turn this into a natural-language prompt that can be used to configure another AI agent to behave exactly like this client when interacting with documents, people, and data.

Client Description:
{client_description}

Optional Tone and Communication Style (from transcripts):
{client_tone}

Write a single-paragraph prompt that:
- Embodies the client's voice, mindset, and values.
- Enables another agent to speak, think, and respond like the client.
- Sounds like a natural internal narrative, not a set of instructions.

Prompt:"""

        self.prompt_template = PromptTemplate(
            input_variables=["client_description", "client_tone"],
            template=inject_guidance(template)
        )

    def _create_chain(self) -> None:
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def _extract_text_from_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _extract_text_from_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _extract_text_from_json(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return "\n".join(f"{k}: {v}" for k, v in data.items())
        elif isinstance(data, list):
            return "\n".join(str(item) for item in data)
        else:
            return str(data)

    def _extract_text_from_file(self, file_path: str) -> str:
        if file_path.endswith(".pdf"):
            return self._extract_text_from_pdf(file_path)
        elif file_path.endswith(".txt"):
            return self._extract_text_from_txt(file_path)
        elif file_path.endswith(".json"):
            return self._extract_text_from_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _extract_all_files_text(self, file_paths: List[str]) -> str:
        all_texts = []
        for path in file_paths:
            try:
                all_texts.append(self._extract_text_from_file(path))
            except Exception as e:
                if self.verbose:
                    print(f"Failed to process {path}: {e}")
        return "\n\n".join(all_texts).strip()

    def run(self, client_description: str, transcript_file_paths: Optional[List[str]] = None) -> str:
        """
        Args:
            client_description (str): A free-form description of the client.
            transcript_file_paths (List[str], optional): Optional list of transcript files to enrich tone.

        Returns:
            str: A paragraph prompt emulating the client or a warning message for unclear input.
        """
        if not client_description or len(client_description.strip()) < 10:
            return "⚠️ It seems you didn't provide a meaningful client description. Please try again with more details."

        # Check for gibberish using a basic heuristic
        if not re.search(r'[a-zA-Z]{3,}', client_description):
            return "⚠️ I couldn't understand your input. Please describe the client using clear and meaningful sentences."

        client_tone = ""
        if transcript_file_paths:
            client_tone = self._extract_all_files_text(transcript_file_paths)

        input_data = {
            "client_description": client_description.strip(),
            "client_tone": client_tone.strip() or "Not specified"
        }

        return self.chain.run(input_data)