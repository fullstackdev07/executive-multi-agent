# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import os, json, fitz
# from typing import List
# import tiktoken
# import logging

# load_dotenv()
# logger = logging.getLogger(__name__)

# def is_input_valid(text: str, min_length: int = 10, short_text_threshold: int = 20, min_unique_chars_for_short_text: int = 3, allow_short_natural_lang: bool = False) -> bool:
#     if not text or not text.strip(): return False
#     stripped_text = text.strip()
#     if allow_short_natural_lang and len(stripped_text) < min_length:
#         # Check if it contains at least one alphanumeric character, to allow short meaningful phrases
#         return bool(len(stripped_text) >= 1 and any(char.isalnum() for char in stripped_text))
#     if len(stripped_text) < min_length: return False
#     if len(stripped_text) <= short_text_threshold:
#         alnum = ''.join(filter(str.isalnum, stripped_text))
#         if not alnum and len(stripped_text) > 0: return False # Pure symbols
#         elif alnum and len(set(alnum.lower())) < min_unique_chars_for_short_text: return False # Repetitive
#     return True

# class JobDescriptionWriterAgent:
#     def __init__(self, verbose: bool = False):
#         self.verbose = verbose
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key:
#             logger.error("JD Writer: OpenAI API key not found.")
#             raise ValueError("OpenAI API key not found.")
        
#         self.llm = ChatOpenAI(
#             openai_api_key=self.openai_api_key, temperature=0.5,
#             model_name="gpt-4", max_tokens=3500 
#         )
#         self.chain = LLMChain(llm=self.llm, prompt=self._create_prompt(), verbose=self.verbose)
#         if self.verbose: logger.info("JobDescriptionWriterAgent initialized.")

#     def _create_prompt(self) -> PromptTemplate:
#         template = """
# You are an expert job description writer. Your task is to synthesize the provided information into a compelling job description.
# The "MANUAL INPUT" might be a detailed client brief, a previously generated client persona, or a very short natural language request like "AI developer" or "JD for a marketing manager".
# If the manual input is very brief, interpret it as the core role or requirement and use your extensive knowledge to generate a comprehensive job description, leveraging any supporting "FILE TEXT" for company-specific context if available.

# --- MANUAL INPUT (Role/Client Brief) ---
# {manual_input}
# --- END MANUAL INPUT ---

# --- SUPPORTING FILE TEXT (Company Info, Market Reports, etc.) ---
# {file_text}
# --- END SUPPORTING FILE TEXT ---

# Write a compelling job description using the following structure. Ensure each section is well-developed and targets the specified word count.

# 1. Current Situation
# - Describe the company: its history, current context, market position, and strategic goals. (Use FILE TEXT primarily for this)
# - Explain why this role is being created or is currently open and its relevance to the company's objectives.

# 2. The Position
# - Detail the core responsibilities and scope of the role identified from MANUAL INPUT.
# - Outline the expected contributions and key performance indicators.
# - Include both explicit tasks and any reasonably inferred tasks critical for such a role.
# - Use a mix of prose for overall description and bullet points for specific duties.

# 3. Candidate Profile 
# - Specify the required qualifications: education, years and type of experience for the role.
# - Describe essential skills: technical, soft, and leadership.
# - Outline desired personality traits and cultural fit.
# - This section should help potential candidates realistically assess their suitability.

# Maintain a fluent, professional, and engaging third-person tone.
# Adhere strictly to the word count guidelines for each section.

# Begin the job description below:
# """
#         return PromptTemplate(input_variables=["manual_input", "file_text"], template=template)

#     def run(self, manual_input: str, file_paths: List[str]) -> str:
#         if self.verbose: logger.info(f"JD Writer: Manual: '{manual_input[:100]}...', Files: {len(file_paths)}")
        
#         final_manual_input = manual_input.strip() if manual_input else ""
#         file_text_content = self._load_files(file_paths) 
        
#         # Allow very short manual_input like "AI developer" (min_length=3, allow_short_natural_lang=True)
#         is_manual_valid = final_manual_input and is_input_valid(final_manual_input, min_length=3, allow_short_natural_lang=True)
#         is_files_valid = file_text_content and is_input_valid(file_text_content, min_length=50) # Files content should be substantial

#         if not is_manual_valid and not is_files_valid: # If manual input is bad AND no good files
#             logger.error("JD Writer: Manual input and file content both insufficient/non-meaningful.")
#             return "Error: Both manual input and supporting file content are insufficient or non-meaningful for JD generation."

#         llm_manual_input = final_manual_input if is_manual_valid else "No specific valid manual input provided; role needs to be inferred or is very generic. Rely on file text if available for company context."
#         # The prompt tells LLM to interpret short manual input as role.
        
#         truncated_file_text = self._truncate_to_token_limit(file_text_content, 7000) if is_files_valid else ""
        
#         if not is_manual_valid and not truncated_file_text: # Double check after processing
#              logger.error("JD Writer: No meaningful input for LLM after validation.")
#              return "Error: No meaningful input available to generate job description after validation."

#         if self.verbose:
#             logger.info(f"JD Writer LLM Input: Manual='{llm_manual_input[:200]}...', Files Preview='{truncated_file_text[:200]}...'")

#         try:
#             payload = {
#                 "manual_input": llm_manual_input,
#                 "file_text": truncated_file_text if truncated_file_text else "No meaningful file content provided."
#             }
#             if hasattr(self.chain, 'invoke'):
#                 res_dict = self.chain.invoke(payload); result_text = res_dict.get('text', str(res_dict))
#             else: result_text = self.chain.run(payload)

#             if not result_text or not result_text.strip():
#                 logger.warning("JD Writer: LLM returned empty response.")
#                 return "Error: Could not generate job description (empty LLM response)."
#             return result_text
#         except Exception as e:
#             logger.exception("JD Writer: LLM chain error.")
#             return f"Error: Could not generate JD due to an exception: {e}"

#     def _load_files(self, file_paths: List[str]) -> str:
#         # ... (file loading logic remains the same, ensuring content.strip() before append)
#         texts = []
#         if not file_paths: return ""
#         for path in file_paths:
#             if not os.path.exists(path): logger.warning(f"JD Writer: File not found {path}"); continue
#             label = f"\n\n--- Content from File: {os.path.basename(path)} ---\n"
#             content = ""; ext = os.path.splitext(path)[-1].lower()
#             try:
#                 if ext == '.pdf': content = self._extract_text_from_pdf(path)
#                 elif ext == '.txt':
#                     with open(path, 'r', encoding='utf-8') as f: content = f.read()
#                 elif ext == '.json': content = self._extract_transcript_from_json(path)
#                 else: logger.warning(f"JD Writer: Unsupported file {path}"); continue
#                 if content and content.strip(): texts.append(label + content.strip())
#             except Exception as e: logger.error(f"JD Writer: Error reading {path}: {e}")
#         return "\n".join(texts).strip()

#     def _extract_text_from_pdf(self, filepath: str) -> str:
#         # ... (same)
#         text = ""
#         try:
#             with fitz.open(filepath) as doc: text = "".join(page.get_text() for page in doc)
#         except Exception as e: logger.error(f"JD Writer: PDF Error {filepath}: {e}"); return ""
#         return text

#     def _extract_transcript_from_json(self, filepath: str) -> str:
#         # ... (same, ensure it handles various JSONs and returns string)
#         try:
#             with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
#             if isinstance(data, list) and data and isinstance(data[0], dict) and \
#                any(k in data[0] for k in ['speaker', 'text', 'transcript']):
#                 parts = [f"{item.get('speaker', 'N/A')}: {item.get('text', item.get('transcript',''))}" for item in data if isinstance(item, dict)]
#                 return "\n".join(parts)
#             return json.dumps(data, indent=2)
#         except Exception as e: logger.error(f"JD Writer: JSON Error {filepath}: {e}"); return ""

#     def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
#         # ... (same)
#         if not text: return ""
#         try: enc = tiktoken.encoding_for_model("gpt-4")
#         except KeyError: enc = tiktoken.get_encoding("cl100k_base")
#         tokens = enc.encode(text)
#         if len(tokens) > max_tokens:
#             logger.info(f"JD Writer: Truncating from {len(tokens)} to {max_tokens} tokens.")
#             return enc.decode(tokens[:max_tokens])
#         return text
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os, json, fitz
from typing import List, Dict
import tiktoken
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import tempfile
import shutil

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI()  # Create FastAPI instance

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

def is_api_input_valid(text: str, field_name: str, min_length: int) -> bool:
    if not text or not text.strip(): return False
    if len(text.strip()) < min_length:
        logger.warning(f"API: {field_name} is too short (min {min_length} chars required).")
        return False
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
            model_name="gpt-4", max_tokens=4000  # Adjusted max_tokens
        )
        # No chain in init, created dynamically during run
        if self.verbose:
            logger.info("JobDescriptionWriterAgent initialized.")

    def _create_prompt(self, section: str, language: str, role_title: str, company_name: str) -> PromptTemplate:
        """Creates a prompt tailored for each JD section."""
        if language == "German":
            template_prefix = "Sie sind ein erfahrener Personalberater für Führungskräfte und erstellen eine Stellenbeschreibung für eine gehobene Position in Deutsch.  Halten Sie die folgenden Richtlinien ein."
            word_count_message = "Die Länge der folgenden Text sollte zwischen 200 und 300 Wörtern liegen."
        else: #English
            template_prefix = "You are an experienced executive search consultant, writing a job description for a senior-level position at {company_name} for the role: {role_title}.  Adhere to the following guidelines."
            word_count_message = "The following text should be between 200 and 300 words in length."
            template_prefix = template_prefix.format(company_name=company_name, role_title=role_title)


        if section == "Current Situation":
            instructions = f"""{template_prefix}
            Create a compelling 'Current Situation' section for the job description. {word_count_message}
            - Briefly describe the company's background and strategic context.
            - Explain the reason this role is being created now.
            - Focus on the company's market position and strategic goals.
            """
        elif section == "The Position":
            instructions = f"""{template_prefix}
            Create a detailed 'The Position' section for the job description. {word_count_message}
            - Clearly outline the responsibilities and expectations of the role.
            - Use a combination of prose and bullet points for clarity.
            - Include both explicit and inferred tasks.
            - Focus on the impact and challenges of the role.
            """
        elif section == "Candidate Profile":
            instructions = f"""{template_prefix}
            Create a well-defined 'Candidate Profile' section for the job description. {word_count_message}
            - Describe the experience, qualifications, and traits required for success.
            - Write in a tone that allows candidates to self-assess their suitability.
            - Focus on the 'must-have' qualities and the ideal candidate's background.
            """
        else:
            raise ValueError(f"Invalid section: {section}")

        template = f"""{instructions}

        Given Context:
        {{input_text}}

        Additional Files Content:
        {{files_content}}

        Job Description Section ({section}):"""

        return PromptTemplate(
            input_variables=["input_text", "files_content"],
            template=template
        )

    def run(self, manual_input: str, file_paths: List[str]) -> str:  # Arguments as before
        """Generates the job description in a non-interactive way."""

        # Extract initial info from manual input (if present)
        role_title = self._extract_value(manual_input, "Role Title:")
        company_name = self._extract_value(manual_input, "Company Name:")
        language = self._extract_value(manual_input, "Language:").capitalize()
        language = language if language in ["English", "German"] else "English" #Default

        all_files_content = ""
        if file_paths:
            all_files_content = self._load_files(file_paths)

        # Generate Each Section
        jd_sections = {}
        for section_name in ["Current Situation", "The Position", "Candidate Profile"]:
            prompt = self._create_prompt(section_name, language, role_title, company_name)
            chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
            try:
                jd_sections[section_name] = chain.run({
                    "input_text": manual_input.strip(),
                    "files_content": all_files_content.strip()
                }).strip()
            except Exception as e:
                logger.error(f"JD Writer: Error during section {section_name} creation: {e}")
                return f"Error: Failed to create {section_name} section. Details: {e}"

        # Assemble the Final Job Description
        final_jd = f"""
        **Current Situation**
        {jd_sections['Current Situation']}

        **The Position**
        {jd_sections['The Position']}

        **Candidate Profile**
        {jd_sections['Candidate Profile']}
        """

        return final_jd.strip()

    def _extract_value(self, text: str, label: str) -> str:
        """Helper to extract values from manual input string."""
        try:
            start_index = text.index(label) + len(label)
            end_index = text.index("\n", start_index) if "\n" in text[start_index:] else len(text)
            return text[start_index:end_index].strip()
        except ValueError:
            return ""

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
