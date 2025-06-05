# import os
# from typing import List, Optional
# from dotenv import load_dotenv
# from langchain_community.llms import OpenAI 
# from langchain_community.chat_models import ChatOpenAI 
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# import fitz 
# import json
# import logging
# import re

# load_dotenv()
# logger = logging.getLogger(__name__)

# def is_input_valid(text: str, min_length: int = 10, short_text_threshold: int = 20, min_unique_chars_for_short_text: int = 3, allow_short_natural_lang: bool = False) -> bool:
#     if not text or not text.strip(): return False
#     stripped_text = text.strip()
#     if allow_short_natural_lang and len(stripped_text) < min_length:
#         return bool(len(stripped_text) >=1 and any(c.isalnum() for c in stripped_text))
#     if len(stripped_text) < min_length: return False
#     if len(stripped_text) <= short_text_threshold:
#         alnum = ''.join(filter(str.isalnum, stripped_text))
#         if not alnum and len(stripped_text) > 0: return False
#         elif alnum and len(set(alnum.lower())) < min_unique_chars_for_short_text: return False
#     return True

# class ClientRepresentativeCreatorAgent:
#     def __init__(self, verbose: bool = False):
#         self.name = "Client Representative Prompt Creator"
#         self.verbose = verbose
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key:
#             logger.error(f"{self.name}: OpenAI API key not found.")
#             raise ValueError("OPENAI_API_KEY not found.")
        
#         try:
#             self.llm = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=0.7, model_name="gpt-3.5-turbo")
#         except Exception as e: logger.exception(f"{self.name}: Error initializing LLM: {e}"); raise

#         self._create_prompt()
#         self._create_chain()
#         if self.verbose: logger.info(f"{self.name} initialized.")

#     def _create_prompt(self) -> None:
#         template = """You are an expert at crafting AI agent personas.
# Your task is to transform a description of a client into a concise, natural-language prompt. This prompt will configure another AI agent to behave, think, and communicate exactly like that client.
# The client description might be detailed, or it could be a very short phrase (e.g., "tech startup founder," "conservative banker," "make a persona for a friendly HR manager"). If the description is brief, use your general knowledge to develop a plausible and rich persona based on that core idea.

# Here is the information about the client:
# --- CLIENT DESCRIPTION START ---
# {client_description}
# --- CLIENT DESCRIPTION END ---

# Here is supplemental information about the client's communication style, potentially extracted from transcripts or other documents (if available):
# --- CLIENT TONE AND COMMUNICATION STYLE (from transcripts/documents) ---
# {client_tone_from_files}
# --- CLIENT TONE AND COMMUNICATION STYLE END ---

# Based on ALL the provided information, write a single, compelling paragraph that serves as an AI agent's system prompt or internal monologue. This paragraph should:
# - Embody the client's core persona: their mindset, primary concerns, values, and decision-making style.
# - Capture the client's typical communication style: tone, formality, and common expressions (if discernible).
# - Be written as if it's the client speaking or thinking to themselves, guiding their actions. It should be a natural internal narrative, NOT a list of instructions for the AI.
# - Enable another AI agent, when given this paragraph as its persona, to interact with documents, data, and people as if it *were* this specific client.
# - Be rich and descriptive, yet concise.

# Generated Client Persona Prompt:
# """
#         self.prompt_template = PromptTemplate(
#             input_variables=["client_description", "client_tone_from_files"], template=template,
#         )

#     def _create_chain(self) -> None:
#         self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=self.verbose)

#     def _extract_text_from_file(self, file_path: str) -> str:
#         # ... (file extraction logic remains the same)
#         if not file_path or not os.path.exists(file_path): return ""
#         content = ""; ext = os.path.splitext(file_path)[-1].lower()
#         try:
#             if ext == ".pdf":
#                 with fitz.open(file_path) as doc: content = "".join(p.get_text() for p in doc)
#             elif ext == ".txt":
#                 with open(file_path, "r", encoding="utf-8") as f: content = f.read()
#             elif ext == ".json":
#                 with open(file_path, "r", encoding="utf-8") as f: data = json.load(f)
#                 # ... (json to text heuristic as before)
#                 text_parts = []
#                 if isinstance(data, list): 
#                     for item in data:
#                         if isinstance(item, dict):
#                             for k_item in ['text', 'transcript', 'line', 'message']:
#                                 if k_item in item: text_parts.append(str(item[k_item])); break
#                             else: text_parts.extend(map(str, item.values()))
#                         else: text_parts.append(str(item))
#                 elif isinstance(data, dict): text_parts.extend(f"{k}: {v}" for k,v in data.items())
#                 else: text_parts.append(str(data))
#                 content = "\n".join(filter(None,text_parts)) or json.dumps(data, indent=2)
#             else: logger.warning(f"{self.name}: Unsupported file: {file_path}"); return ""
#             return content.strip()
#         except Exception as e: logger.error(f"{self.name}: Error reading {file_path}: {e}"); return ""


#     def _extract_all_files_text(self, file_paths: List[str]) -> str:
#         all_texts = [f"--- Content from file: {os.path.basename(p)} ---\n{c}" 
#                      for p in file_paths if (c := self._extract_text_from_file(p))]
#         return "\n\n".join(all_texts).strip()

#     def run(self, client_description: str, transcript_file_paths: Optional[List[str]] = None) -> str:
#         final_client_desc = client_description.strip() if client_description else ""
#         client_tone_files = self._extract_all_files_text(transcript_file_paths or [])

#         # Allow short natural language for client_description. min_length 3 for "CEO" etc.
#         is_desc_valid = final_client_desc and is_input_valid(final_client_desc, min_length=3, allow_short_natural_lang=True)
#         is_files_valid = client_tone_files and is_input_valid(client_tone_files, min_length=20) # Files should have more substance

#         if not is_desc_valid and not is_files_valid:
#             logger.error(f"{self.name}: Client desc and file content insufficient.")
#             return "Error: Client description and transcript file content are both insufficient or non-meaningful."
        
#         desc_for_llm = final_client_desc if is_desc_valid else "No explicit client description or it was too brief/non-meaningful. Infer from file content if available or use general knowledge for common roles."
        
#         if self.verbose:
#             logger.info(f"{self.name}: Desc for LLM: {desc_for_llm[:100]}..., Files text preview: {client_tone_files[:100]}...")

#         input_data = {
#             "client_description": desc_for_llm,
#             "client_tone_from_files": client_tone_files if is_files_valid else "No meaningful transcript/document content provided."
#         }
        
#         try:
#             if hasattr(self.chain, 'invoke'):
#                 res_dict = self.chain.invoke(input_data); response = res_dict.get('text', str(res_dict))
#             else: response = self.chain.run(input_data)
#             logger.info(f"{self.name}: Generated client prompt.")
#             return response.strip() 
#         except Exception as e:
#             logger.exception(f"{self.name}: LLM chain error: {e}")
#             return f"Error: Could not generate client prompt. Details: {e}"

import os
from typing import List, Optional, Union
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import fitz
import json
import logging
import re
from fastapi import UploadFile
from io import BytesIO

load_dotenv()
logger = logging.getLogger(__name__)

# Configure logging basic settings if not already configured
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def is_input_valid(text: str, min_length: int = 5, short_text_threshold: int = 20,
                   min_unique_chars_for_short_text: int = 3, allow_short_natural_lang: bool = True) -> bool:
    """
    Validates if the input text meets minimum length and character complexity criteria.
    Allows short natural language inputs if specified.
    """
    if not text or not text.strip():
        return False
    stripped_text = text.strip()

    # Allow short natural language like "CEO" if specified
    if allow_short_natural_lang and len(stripped_text) < min_length:
        # Check if it contains at least one alphanumeric character
        return bool(len(stripped_text) >= 1 and any(c.isalnum() for c in stripped_text))

    # For longer text or if short natural language is not allowed
    if len(stripped_text) < min_length:
        return False

    # Additional check for very short texts (up to short_text_threshold) to prevent repetitive garbage
    if len(stripped_text) <= short_text_threshold:
        alnum = ''.join(filter(str.isalnum, stripped_text))
        if not alnum and len(stripped_text) > 0: # If text has chars but none are alnum (e.g., "---")
            return False
        elif alnum and len(set(alnum.lower())) < min_unique_chars_for_short_text: # Check unique alnum chars
            return False

    return True

class ClientRepresentativeCreatorAgent:
    """
    An agent that generates a prompt template for creating an evaluator persona
    based on manual text input and/or file content (transcripts, documents).
    """
    def __init__(self, verbose: bool = False):
        self.name = "Evaluator Persona Prompt Creator"
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error(f"{self.name}: OpenAI API key not found.")
            raise ValueError("OPENAI_API_KEY not found.")

        try:
            # Using the recommended way to import ChatOpenAI
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=0.7, model_name="gpt-3.5-turbo")
        except ImportError:
             logger.warning("Could not import langchain_openai. Falling back to langchain_community.")
             try:
                 self.llm = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=0.7, model_name="gpt-3.5-turbo")
             except Exception as e:
                 logger.exception(f"{self.name}: Error initializing LLM with community: {e}");
                 raise
        except Exception as e:
            logger.exception(f"{self.name}: Error initializing LLM: {e}");
            raise

        self._create_prompt()
        self._create_chain()
        if self.verbose: logger.info(f"{self.name} initialized.")


    def _create_prompt(self) -> None:
        """
        Defines the prompt template used to instruct the LLM to generate the
        evaluator persona creation prompt.
        """
        template = """You are an expert in creating prompt templates for AI agents. Your task is to generate a prompt template that will guide an AI agent in creating an evaluator persona for a SaaS CEO candidate assessment.

        The prompt template should instruct the AI to:

        1.  Create a Persona Summary (1 paragraph):  This should cover the tone, strategic lens, and priorities of the evaluator persona. The persona represents in their role as.
        2.  Define Evaluation Heuristics:  Create a table of 5–8 evaluation rules that the persona would use, along with a rationale for each rule (e.g., “Prefers product-centric CEOs with GTM scaling experience because…”).
        3.  Identify Success Markers: List observable signs of high potential in candidates, according to the persona (e.g., demonstrated resilience, unusual P&L insight, etc.).
        4.  Establish Cultural Fit Filters: Define beliefs or attitudes that the persona would find acceptable or unacceptable in the organization's culture.

        The prompt template MUST include a placeholder for "Source Insights" where the extracted information about the stakeholder’s preferences will be inserted.

        Here's the prompt template you should generate, incorporating any provided input as context for the prompt itself:

        """
        # The template expects 'combined_input' which will contain text from manual input and/or files.
        self.prompt_template = PromptTemplate(
            input_variables=["combined_input"],
            template=template,
        )

    def _create_chain(self) -> None:
        """Initializes the LangChain LLMChain."""
        # Using the recommended way to create the chain
        from langchain_core.runnables import RunnablePassthrough
        self.chain = {"combined_input": RunnablePassthrough()} | self.prompt_template | self.llm


    def _extract_text_from_uploadfile(self, file: UploadFile) -> str:
        """Extracts text content from an UploadFile object (PDF, TXT, JSON)."""
        if not file: return ""

        try:
            # Read the file content. SpooledTemporaryFile needs to be reset after reading.
            file.file.seek(0) # Ensure pointer is at the beginning
            contents = file.file.read()
            file.file.seek(0) # Reset pointer for potential future reads if needed

            ext = os.path.splitext(file.filename)[-1].lower()

            if ext == ".pdf":
                # Process PDF from bytes in memory
                with BytesIO(contents) as f:
                    with fitz.open(stream=f, filetype="pdf") as doc:
                        content = "".join(p.get_text() for p in doc)
            elif ext == ".txt":
                content = contents.decode("utf-8", errors='ignore') # Use errors='ignore' for robustness
            elif ext == ".json":
                content = self._extract_text_from_json(contents)
            else:
                logger.warning(f"{self.name}: Unsupported file type: {ext} for file {file.filename}")
                return ""

            return content.strip()

        except Exception as e:
            logger.error(f"{self.name}: Error reading {file.filename}: {e}")
            return ""
        # FastAPI handles closing the UploadFile's SpooledTemporaryFile automatically.
        # No need for file.file.close() in a finally block here with typical FastAPI usage.


    def _extract_text_from_json(self, contents: bytes) -> str:
        """Tries to extract text content from JSON bytes."""
        try:
            data = json.loads(contents.decode("utf-8"))
            text_parts = []
            # Basic heuristic to get text from common key names or values
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Prioritize common text keys
                        for k_item in ['text', 'transcript', 'line', 'message', 'content', 'description', 'value']:
                            if k_item in item and isinstance(item[k_item], str):
                                text_parts.append(str(item[k_item]))
                                break # Take the first match in this item
                        else: # If no common key found, append all string values
                             text_parts.extend([str(v) for v in item.values() if isinstance(v, (str, int, float))]) # Include numbers too
                    else: # Handle list of non-dict items
                        text_parts.append(str(item))
            elif isinstance(data, dict):
                 # Prioritize common text keys at the top level
                 for k_item in ['text', 'transcript', 'line', 'message', 'content', 'description', 'summary', 'details', 'value']:
                      if k_item in data and isinstance(data[k_item], str):
                           text_parts.append(f"{k_item}: {data[k_item]}")
                           break # Take the first match
                 else: # If no common key found, append all string values
                      text_parts.extend([f"{k}: {v}" for k, v in data.items() if isinstance(v, (str, int, float))])
            else: # Handle simple JSON types (string, number, boolean, null)
                text_parts.append(str(data))

            # Join extracted parts, filter out empty strings, and return original JSON if nothing found
            return "\n".join(filter(None, text_parts)).strip() or json.dumps(data, indent=2)
        except json.JSONDecodeError as e:
            logger.error(f"{self.name}: Error decoding JSON: {e}")
            return ""
        except Exception as e:
             logger.error(f"{self.name}: Error processing JSON content: {e}")
             return ""


    def _extract_all_files_text(self, files: Optional[List[UploadFile]]) -> str:
        """Extracts text from a list of UploadFile objects."""
        if not files:
            return ""

        all_texts = []
        for file in files:
            content = self._extract_text_from_uploadfile(file)
            if content:
                all_texts.append(f"--- Content from file: {file.filename} ---\n{content}")

        return "\n\n".join(all_texts).strip()


    def run(self, client_description: Optional[str] = None, transcript_files: Optional[List[UploadFile]] = None) -> str:
        """
        Generates the evaluator persona prompt template based on provided client_description
        and/or transcript files.

        Args:
            client_description (Optional[str]): Manual text client_description/description.
            transcript_files (Optional[List[UploadFile]]): List of uploaded files
                                                         (e.g., interview transcripts).

        Returns:
            str: The generated prompt template, or an error message.
        """
        instruction_text = client_description.strip() if client_description else ""
        files_text = self._extract_all_files_text(transcript_files or [])

        # Validate input sources - files need more substance than manual client_description
        is_instruction_valid = instruction_text and is_input_valid(instruction_text, min_length=5, allow_short_natural_lang=True)
        is_files_valid = files_text and is_input_valid(files_text, min_length=50) # Require more content from files

        combined_input = ""

        if is_instruction_valid:
            combined_input += f"--- Manual client_description: ---\n{instruction_text}\n\n" # Added newline
        if is_files_valid:
            combined_input += f"--- File Content: ---\n{files_text}\n"

        if not combined_input.strip(): # Check if combined input is just whitespace
            logger.error(f"{self.name}: No valid input provided.")
            return "Insufficient input (neither meaningful client_description nor file content) to generate a meaningful prompt."

        if self.verbose:
            logger.info(f"{self.name}: Combined input for LLM preview:\n{combined_input[:500]}...")

        # Pass the combined input to the LLM chain
        input_data = {"combined_input": combined_input}

        try:
            # Use invoke for newer LangChain versions
            if hasattr(self.chain, 'invoke'):
                 response = self.chain.invoke(input_data)
                 # If the chain is a sequence ending in an LLM, invoke might return a Message object
                 # If it's a simple LLMChain, it might return a dict.
                 # Handle both cases or just stringify as a fallback
                 if isinstance(response, dict) and 'text' in response:
                     response_text = response['text']
                 elif hasattr(response, 'content'): # Handle Message objects
                     response_text = response.content
                 else:
                     response_text = str(response) # Fallback
            else: # Use run for older LangChain versions
                response_text = self.chain.run(input_data)

            logger.info(f"{self.name}: Generated evaluator persona prompt successfully.")
            return response_text.strip()

        except Exception as e:
            logger.exception(f"{self.name}: LLM chain execution error: {e}")
            return f"Error: Could not generate evaluator persona prompt. Details: {e}"