from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os, json, fitz
from typing import List
import tiktoken

load_dotenv()

class JobDescriptionWriterAgent:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found.")
        
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0.5,
            model_name="gpt-4",  # Use "gpt-4o" or "gpt-3.5-turbo" if needed
            max_tokens=3500
        )
        self.chain = LLMChain(llm=self.llm, prompt=self._create_prompt())

    def _create_prompt(self) -> PromptTemplate:
        template = """
You are an expert job description writer.

Below is a summary of inputs provided for the JD:

{manual_input}

Below is supporting information extracted from documents and transcripts:

{file_text}

Write a compelling job description using the following structure:

### 1. Current Situation (300–350 words)
- Describe the company, its history, context, and strategic goals.
- Explain why this role is relevant now.

### 2. The Position (300–350 words)
- Responsibilities, scope, expected contributions.
- Include explicit and inferred tasks.
- Mix prose and bullet points.

### 3. Candidate Profile (300–350 words)
- Required qualifications, experience, and traits.
- Help readers assess fit.

Use fluent, professional third-person tone throughout. Each section must be 300–350 words.

Begin the job description below:
"""
        return PromptTemplate(
            input_variables=["manual_input", "file_text"],
            template=template
        )

    def run(self, manual_input: str, file_paths: List[str]) -> str:
        file_text = self._load_files(file_paths)
        file_text = self._truncate_to_token_limit(file_text, max_tokens=3000)

        if self.verbose:
            print("Manual Input:", manual_input)
            print("File Text Preview:", file_text[:1000], "...")

        return self.chain.invoke({
            "manual_input": manual_input,
            "file_text": file_text
        })

        if self.verbose:
            print("\n=== Generated Job Description ===\n")
            print(result)

        return result

    def _load_files(self, file_paths: List[str]) -> str:
        text = ""
        for path in file_paths:
            try:
                label = f"\n--- File: {os.path.basename(path)} ---\n"
                if path.lower().endswith('.pdf'):
                    text += label + self._extract_text_from_pdf(path) + "\n"
                elif path.lower().endswith('.txt'):
                    with open(path, 'r', encoding='utf-8') as f:
                        text += label + f.read() + "\n"
                elif path.lower().endswith('.json'):
                    text += label + self._extract_transcript_from_json(path) + "\n"
                else:
                    if self.verbose:
                        print(f"Unsupported file type: {path}")
            except Exception as e:
                if self.verbose:
                    print(f"Error reading file {path}: {e}")
        return text.strip()

    def _extract_text_from_pdf(self, filepath: str) -> str:
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def _extract_transcript_from_json(self, filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return "\n".join(f"{d.get('speaker', 'Unknown')}: {d.get('text', '')}" for d in data)
        return json.dumps(data, indent=2)

    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        try:
            enc = tiktoken.encoding_for_model("gpt-4")
        except:
            enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) > max_tokens:
            if self.verbose:
                print(f"Truncating file_text from {len(tokens)} to {max_tokens} tokens.")
            tokens = tokens[:max_tokens]
        return enc.decode(tokens)