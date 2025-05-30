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

def is_input_valid(text: str, min_length: int = 10, short_text_threshold: int = 20, min_unique_chars_for_short_text: int = 3, allow_short_natural_lang: bool = False) -> bool:
    if not text or not text.strip(): return False
    stripped_text = text.strip()
    if allow_short_natural_lang and len(stripped_text) < min_length:
        # Check if it contains at least one alphanumeric character, to allow short meaningful phrases
        return bool(len(stripped_text) >= 1 and any(char.isalnum() for char in stripped_text))
    if len(stripped_text) < min_length: return False
    if len(stripped_text) <= short_text_threshold:
        alnum = ''.join(filter(str.isalnum, stripped_text))
        if not alnum and len(stripped_text) > 0: return False # Pure symbols
        elif alnum and len(set(alnum.lower())) < min_unique_chars_for_short_text: return False # Repetitive
    return True

class JobDescriptionWriterAgent:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("JD Writer: OpenAI API key not found.")
            raise ValueError("OpenAI API key not found.")
        
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key, temperature=0.5,
            model_name="gpt-4", max_tokens=3500 
        )
        self.chain = LLMChain(llm=self.llm, prompt=self._create_prompt(), verbose=self.verbose)
        if self.verbose: logger.info("JobDescriptionWriterAgent initialized.")

    def _create_prompt(self) -> PromptTemplate:
        template = """
You are an expert job description writer. Your task is to synthesize the provided information into a compelling job description.
The "MANUAL INPUT" might be a detailed client brief, a previously generated client persona, or a very short natural language request like "AI developer" or "JD for a marketing manager".
If the manual input is very brief, interpret it as the core role or requirement and use your extensive knowledge to generate a comprehensive job description, leveraging any supporting "FILE TEXT" for company-specific context if available.

--- MANUAL INPUT (Role/Client Brief) ---
{manual_input}
--- END MANUAL INPUT ---

--- SUPPORTING FILE TEXT (Company Info, Market Reports, etc.) ---
{file_text}
--- END SUPPORTING FILE TEXT ---

Write a compelling job description using the following structure. Ensure each section is well-developed and targets the specified word count.

### 1. Current Situation (Approximately 300–350 words)
- Describe the company: its history, current context, market position, and strategic goals. (Use FILE TEXT primarily for this)
- Explain why this role is being created or is currently open and its relevance to the company's objectives.

### 2. The Position (Approximately 300–350 words)
- Detail the core responsibilities and scope of the role identified from MANUAL INPUT.
- Outline the expected contributions and key performance indicators.
- Include both explicit tasks and any reasonably inferred tasks critical for such a role.
- Use a mix of prose for overall description and bullet points for specific duties.

### 3. Candidate Profile (Approximately 300–350 words)
- Specify the required qualifications: education, years and type of experience for the role.
- Describe essential skills: technical, soft, and leadership.
- Outline desired personality traits and cultural fit.
- This section should help potential candidates realistically assess their suitability.

Maintain a fluent, professional, and engaging third-person tone.
Adhere strictly to the word count guidelines for each section.

Begin the job description below:
"""
        return PromptTemplate(input_variables=["manual_input", "file_text"], template=template)

    def run(self, manual_input: str, file_paths: List[str]) -> str:
        if self.verbose: logger.info(f"JD Writer: Manual: '{manual_input[:100]}...', Files: {len(file_paths)}")
        
        final_manual_input = manual_input.strip() if manual_input else ""
        file_text_content = self._load_files(file_paths) 
        
        # Allow very short manual_input like "AI developer" (min_length=3, allow_short_natural_lang=True)
        is_manual_valid = final_manual_input and is_input_valid(final_manual_input, min_length=3, allow_short_natural_lang=True)
        is_files_valid = file_text_content and is_input_valid(file_text_content, min_length=50) # Files content should be substantial

        if not is_manual_valid and not is_files_valid: # If manual input is bad AND no good files
            logger.error("JD Writer: Manual input and file content both insufficient/non-meaningful.")
            return "Error: Both manual input and supporting file content are insufficient or non-meaningful for JD generation."

        llm_manual_input = final_manual_input if is_manual_valid else "No specific valid manual input provided; role needs to be inferred or is very generic. Rely on file text if available for company context."
        # The prompt tells LLM to interpret short manual input as role.
        
        truncated_file_text = self._truncate_to_token_limit(file_text_content, 7000) if is_files_valid else ""
        
        if not is_manual_valid and not truncated_file_text: # Double check after processing
             logger.error("JD Writer: No meaningful input for LLM after validation.")
             return "Error: No meaningful input available to generate job description after validation."

        if self.verbose:
            logger.info(f"JD Writer LLM Input: Manual='{llm_manual_input[:200]}...', Files Preview='{truncated_file_text[:200]}...'")

        try:
            payload = {
                "manual_input": llm_manual_input,
                "file_text": truncated_file_text if truncated_file_text else "No meaningful file content provided."
            }
            if hasattr(self.chain, 'invoke'):
                res_dict = self.chain.invoke(payload); result_text = res_dict.get('text', str(res_dict))
            else: result_text = self.chain.run(payload)

            if not result_text or not result_text.strip():
                logger.warning("JD Writer: LLM returned empty response.")
                return "Error: Could not generate job description (empty LLM response)."
            return result_text
        except Exception as e:
            logger.exception("JD Writer: LLM chain error.")
            return f"Error: Could not generate JD due to an exception: {e}"

    def _load_files(self, file_paths: List[str]) -> str:
        # ... (file loading logic remains the same, ensuring content.strip() before append)
        texts = []
        if not file_paths: return ""
        for path in file_paths:
            if not os.path.exists(path): logger.warning(f"JD Writer: File not found {path}"); continue
            label = f"\n\n--- Content from File: {os.path.basename(path)} ---\n"
            content = ""; ext = os.path.splitext(path)[-1].lower()
            try:
                if ext == '.pdf': content = self._extract_text_from_pdf(path)
                elif ext == '.txt':
                    with open(path, 'r', encoding='utf-8') as f: content = f.read()
                elif ext == '.json': content = self._extract_transcript_from_json(path)
                else: logger.warning(f"JD Writer: Unsupported file {path}"); continue
                if content and content.strip(): texts.append(label + content.strip())
            except Exception as e: logger.error(f"JD Writer: Error reading {path}: {e}")
        return "\n".join(texts).strip()

    def _extract_text_from_pdf(self, filepath: str) -> str:
        # ... (same)
        text = ""
        try:
            with fitz.open(filepath) as doc: text = "".join(page.get_text() for page in doc)
        except Exception as e: logger.error(f"JD Writer: PDF Error {filepath}: {e}"); return ""
        return text

    def _extract_transcript_from_json(self, filepath: str) -> str:
        # ... (same, ensure it handles various JSONs and returns string)
        try:
            with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
            if isinstance(data, list) and data and isinstance(data[0], dict) and \
               any(k in data[0] for k in ['speaker', 'text', 'transcript']):
                parts = [f"{item.get('speaker', 'N/A')}: {item.get('text', item.get('transcript',''))}" for item in data if isinstance(item, dict)]
                return "\n".join(parts)
            return json.dumps(data, indent=2)
        except Exception as e: logger.error(f"JD Writer: JSON Error {filepath}: {e}"); return ""

    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        # ... (same)
        if not text: return ""
        try: enc = tiktoken.encoding_for_model("gpt-4")
        except KeyError: enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) > max_tokens:
            logger.info(f"JD Writer: Truncating from {len(tokens)} to {max_tokens} tokens.")
            return enc.decode(tokens[:max_tokens])
        return text