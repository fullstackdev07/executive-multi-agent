import os
import json
from typing import List, Union, Optional
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader

load_dotenv()

class ClientRepresentativeAgent:
    def __init__(self, verbose: bool = False):
        self.name = "Client Representative Prompt Creator"
        self.role = "Generating an agent prompt that reflects the client's mindset"
        self.goal = "To create a prompt that enables another agent to behave like the client"
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7)
        self._create_prompt()
        self._create_chain()

    def _create_prompt(self) -> None:
        template = """You are creating a prompt for an AI agent that must behave like a specific client when reviewing and interacting with documents and data.

Client Persona:
{client_persona}

Client Priorities:
{client_priorities}

Client Values:
{client_values}

Client Tone and Communication Style (based on transcripts):
{client_tone}

Generate a prompt that configures an AI agent to:
1. Emulate the client's tone and communication style.
2. Reflect the client's values and priorities.
3. Review documents or data critically and constructively from the client's perspective.
4. Respond in a way that aligns with how the client would speak and think.

The output should be a prompt that can be used to configure another LLM-based agent to behave like this client.

Generated Prompt:"""

        self.prompt_template = PromptTemplate(
            input_variables=[
                "client_persona",
                "client_priorities",
                "client_values",
                "client_tone",
            ],
            template=template,
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

    def _parse_manual_input(self, manual_text: str) -> dict:
        fields = {
            "client_persona": "Unknown",
            "client_priorities": "Not specified",
            "client_values": "Not specified"
        }

        for line in manual_text.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().lower()
                val = val.strip()
                if "persona" in key:
                    fields["client_persona"] = val
                elif "priorities" in key:
                    fields["client_priorities"] = val
                elif "values" in key:
                    fields["client_values"] = val

        return fields

    def run(self, manual_input_text: Optional[str] = None, transcript_file_paths: Optional[List[str]] = None) -> str:
        if not manual_input_text and not transcript_file_paths:
            raise ValueError("At least manual_input_text or transcript_file_paths must be provided.")

        # Extract fields from manual input
        manual_fields = {
            "client_persona": "Unknown",
            "client_priorities": "Not specified",
            "client_values": "Not specified"
        }
        if manual_input_text:
            manual_fields = self._parse_manual_input(manual_input_text)

        # Extract client tone from transcript files
        client_tone = ""
        if transcript_file_paths:
            client_tone = self._extract_all_files_text(transcript_file_paths)

        # Final input for the prompt template
        input_data = {
            "client_persona": manual_fields["client_persona"],
            "client_priorities": manual_fields["client_priorities"],
            "client_values": manual_fields["client_values"],
            "client_tone": client_tone.strip() or "Not specified"
        }

        return self.chain.run(input_data)