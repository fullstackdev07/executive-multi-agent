# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import os, json, fitz
# from typing import List
# import tiktoken

# load_dotenv()

# class JobDescriptionWriterAgent:
#     def __init__(self, verbose: bool = False):
#         self.verbose = verbose
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key:
#             raise ValueError("OpenAI API key not found.")
        
#         self.llm = ChatOpenAI(
#             openai_api_key=self.openai_api_key,
#             temperature=0.5,
#             model_name="gpt-4",  # Use "gpt-4o" or "gpt-3.5-turbo" if needed
#             max_tokens=3500
#         )
#         self.chain = LLMChain(llm=self.llm, prompt=self._create_prompt())

#     def _create_prompt(self) -> PromptTemplate:
#         template = """
# You are an expert job description writer.

# Below is a summary of inputs provided for the JD:

# {manual_input}

# Below is supporting information extracted from documents and transcripts:

# {file_text}

# Write a compelling job description using the following structure:

# ### 1. Current Situation (300–350 words)
# - Describe the company, its history, context, and strategic goals.
# - Explain why this role is relevant now.

# ### 2. The Position (300–350 words)
# - Responsibilities, scope, expected contributions.
# - Include explicit and inferred tasks.
# - Mix prose and bullet points.

# ### 3. Candidate Profile (300–350 words)
# - Required qualifications, experience, and traits.
# - Help readers assess fit.

# Use fluent, professional third-person tone throughout. Each section must be 300–350 words.

# Begin the job description below:
# """
#         return PromptTemplate(
#             input_variables=["manual_input", "file_text"],
#             template=template
#         )

#     def run(self, manual_input: str, file_paths: List[str]) -> str:
#         file_text = self._load_files(file_paths)
#         file_text = self._truncate_to_token_limit(file_text, max_tokens=3000)

#         if self.verbose:
#             print("Manual Input:", manual_input)
#             print("File Text Preview:", file_text[:1000], "...")

#         return self.chain.invoke({
#             "manual_input": manual_input,
#             "file_text": file_text
#         })

#         if self.verbose:
#             print("\n=== Generated Job Description ===\n")
#             print(result)

#         return result

#     def _load_files(self, file_paths: List[str]) -> str:
#         text = ""
#         for path in file_paths:
#             try:
#                 label = f"\n--- File: {os.path.basename(path)} ---\n"
#                 if path.lower().endswith('.pdf'):
#                     text += label + self._extract_text_from_pdf(path) + "\n"
#                 elif path.lower().endswith('.txt'):
#                     with open(path, 'r', encoding='utf-8') as f:
#                         text += label + f.read() + "\n"
#                 elif path.lower().endswith('.json'):
#                     text += label + self._extract_transcript_from_json(path) + "\n"
#                 else:
#                     if self.verbose:
#                         print(f"Unsupported file type: {path}")
#             except Exception as e:
#                 if self.verbose:
#                     print(f"Error reading file {path}: {e}")
#         return text.strip()

#     def _extract_text_from_pdf(self, filepath: str) -> str:
#         text = ""
#         with fitz.open(filepath) as doc:
#             for page in doc:
#                 text += page.get_text()
#         return text

#     def _extract_transcript_from_json(self, filepath: str) -> str:
#         with open(filepath, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         if isinstance(data, list):
#             return "\n".join(f"{d.get('speaker', 'Unknown')}: {d.get('text', '')}" for d in data)
#         return json.dumps(data, indent=2)

#     def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
#         try:
#             enc = tiktoken.encoding_for_model("gpt-4")
#         except:
#             enc = tiktoken.get_encoding("cl100k_base")
#         tokens = enc.encode(text)
#         if len(tokens) > max_tokens:
#             if self.verbose:
#                 print(f"Truncating file_text from {len(tokens)} to {max_tokens} tokens.")
#             tokens = tokens[:max_tokens]
#         return enc.decode(tokens)

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os, json, fitz
from typing import List
import tiktoken
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class JobDescriptionWriterAgent:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OpenAI API key not found for JobDescriptionWriterAgent.")
            raise ValueError("OpenAI API key not found.")
        
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0.5,
            model_name="gpt-4",  # Use "gpt-4o" or "gpt-3.5-turbo" if needed
            max_tokens=3500 # Max tokens for the *output*, context window is separate
        )
        self.chain = LLMChain(llm=self.llm, prompt=self._create_prompt(), verbose=self.verbose)
        if self.verbose:
            logger.info("JobDescriptionWriterAgent initialized.")

    def _create_prompt(self) -> PromptTemplate:
        template = """
You are an expert job description writer. Your task is to synthesize the provided information into a compelling job description.

Below is a summary of inputs provided for the JD (this could be specific instructions, desired tone, key points from a client, etc.):
--- MANUAL INPUT ---
{manual_input}
--- END MANUAL INPUT ---

Below is supporting information extracted from documents (like company brochures, existing JDs, market reports, interview transcripts, etc.):
--- FILE TEXT ---
{file_text}
--- END FILE TEXT ---

Write a compelling job description using the following structure. Ensure each section is well-developed and targets the specified word count.

### 1. Current Situation (Approximately 300–350 words)
- Describe the company: its history, current context, market position, and strategic goals.
- Explain why this role is being created or is currently open and its relevance to the company's objectives.

### 2. The Position (Approximately 300–350 words)
- Detail the core responsibilities and scope of the role.
- Outline the expected contributions and key performance indicators.
- Include both explicit tasks mentioned in inputs and any reasonably inferred tasks critical for the role.
- Use a mix of prose for overall description and bullet points for specific duties.

### 3. Candidate Profile (Approximately 300–350 words)
- Specify the required qualifications: education, years and type of experience.
- Describe essential skills: technical, soft, and leadership.
- Outline desired personality traits and cultural fit.
- This section should help potential candidates realistically assess their suitability for the role.

Maintain a fluent, professional, and engaging third-person tone throughout the entire job description.
Adhere strictly to the word count guidelines for each section to ensure a balanced and comprehensive document.

Begin the job description below:
"""
        return PromptTemplate(
            input_variables=["manual_input", "file_text"],
            template=template
        )

    def run(self, manual_input: str, file_paths: List[str]) -> str:
        if self.verbose:
            logger.info(f"JobDescriptionWriterAgent: Running with manual_input and {len(file_paths)} files.")
        
        file_text_content = self._load_files(file_paths)
        # Consider a token limit for file_text_content that leaves room for manual_input and prompt itself
        # Model gpt-4 has a large context window (e.g., 8k or 32k tokens).
        # max_tokens in LLM init is for *output*.
        # A 4000 token limit for input file_text seems reasonable to avoid hitting overall context limits with gpt-4.
        truncated_file_text = self._truncate_to_token_limit(file_text_content, max_tokens=7000) # Increased for gpt-4

        if self.verbose:
            logger.info(f"Manual Input for JD: {manual_input[:200]}...")
            logger.info(f"File Text Preview for JD (truncated): {truncated_file_text[:500]}...")

        try:
            response_dict = self.chain.invoke({
                "manual_input": manual_input if manual_input else "No specific manual input provided. Rely on file text.",
                "file_text": truncated_file_text if truncated_file_text else "No file content provided."
            })
            # The output key for LLMChain with a ChatModel is typically 'text'
            result_text = response_dict.get('text', '')
            
            if not result_text:
                logger.warning("JobDescriptionWriterAgent: LLM returned an empty response.")
                return "Error: Could not generate job description."

            if self.verbose:
                logger.info("\n=== Generated Job Description ===\n")
                logger.info(result_text[:500] + "...") # Log beginning of JD
            
            return result_text

        except Exception as e:
            logger.exception("JobDescriptionWriterAgent: Error during LLM chain invocation.")
            raise  # Re-raise the exception to be caught by the caller

    def _load_files(self, file_paths: List[str]) -> str:
        texts = []
        if not file_paths:
            return ""
        for path in file_paths:
            try:
                label = f"\n\n--- Content from File: {os.path.basename(path)} ---\n"
                content = ""
                if path.lower().endswith('.pdf'):
                    content = self._extract_text_from_pdf(path)
                elif path.lower().endswith('.txt'):
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                elif path.lower().endswith('.json'):
                    content = self._extract_transcript_from_json(path) # Or generic JSON to text
                else:
                    if self.verbose:
                        logger.warning(f"JobDescriptionWriterAgent: Unsupported file type: {path}, skipping.")
                    continue
                
                if content:
                    texts.append(label + content)
            except Exception as e:
                if self.verbose:
                    logger.error(f"JobDescriptionWriterAgent: Error reading file {path}: {e}")
        return "\n".join(texts).strip()

    def _extract_text_from_pdf(self, filepath: str) -> str:
        text = ""
        try:
            with fitz.open(filepath) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            logger.error(f"JobDescriptionWriterAgent: Failed to extract text from PDF {filepath}: {e}")
            return ""
        return text

    def _extract_transcript_from_json(self, filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Attempt to parse structured transcript if possible, otherwise dump JSON
            if isinstance(data, list) and data and isinstance(data[0], dict) and ('speaker' in data[0] or 'text' in data[0] or 'transcript' in data[0]):
                # Heuristic for transcript-like JSON
                transcript_parts = []
                for item in data:
                    speaker = item.get('speaker', item.get('Speaker', 'Unknown Speaker'))
                    text = item.get('text', item.get('transcript', item.get('line', '')))
                    if text: # Only add if there's text
                        transcript_parts.append(f"{speaker}: {text}")
                return "\n".join(transcript_parts)
            else: # Fallback to pretty-printed JSON
                return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"JobDescriptionWriterAgent: Failed to extract text from JSON {filepath}: {e}")
            return ""

    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        if not text:
            return ""
        try:
            # Using encoding for gpt-4, but cl100k_base is common for newer models.
            enc = tiktoken.encoding_for_model("gpt-4") 
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        
        tokens = enc.encode(text)
        if len(tokens) > max_tokens:
            if self.verbose:
                logger.info(f"JobDescriptionWriterAgent: Truncating input text from {len(tokens)} to {max_tokens} tokens.")
            truncated_tokens = tokens[:max_tokens]
            return enc.decode(truncated_tokens)
        return text