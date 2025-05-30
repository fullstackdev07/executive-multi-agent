import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.llms import OpenAI 
from langchain_community.chat_models import ChatOpenAI 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import fitz 
import json
import logging
import re

load_dotenv()
logger = logging.getLogger(__name__)

def is_input_valid(text: str, min_length: int = 10, short_text_threshold: int = 20, min_unique_chars_for_short_text: int = 3, allow_short_natural_lang: bool = False) -> bool:
    if not text or not text.strip(): return False
    stripped_text = text.strip()
    if allow_short_natural_lang and len(stripped_text) < min_length:
        return bool(len(stripped_text) >=1 and any(c.isalnum() for c in stripped_text))
    if len(stripped_text) < min_length: return False
    if len(stripped_text) <= short_text_threshold:
        alnum = ''.join(filter(str.isalnum, stripped_text))
        if not alnum and len(stripped_text) > 0: return False
        elif alnum and len(set(alnum.lower())) < min_unique_chars_for_short_text: return False
    return True

class ClientRepresentativeCreatorAgent:
    def __init__(self, verbose: bool = False):
        self.name = "Client Representative Prompt Creator"
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error(f"{self.name}: OpenAI API key not found.")
            raise ValueError("OPENAI_API_KEY not found.")
        
        try:
            self.llm = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=0.7, model_name="gpt-3.5-turbo")
        except Exception as e: logger.exception(f"{self.name}: Error initializing LLM: {e}"); raise

        self._create_prompt()
        self._create_chain()
        if self.verbose: logger.info(f"{self.name} initialized.")

    def _create_prompt(self) -> None:
        template = """You are an expert at crafting AI agent personas.
Your task is to transform a description of a client into a concise, natural-language prompt. This prompt will configure another AI agent to behave, think, and communicate exactly like that client.
The client description might be detailed, or it could be a very short phrase (e.g., "tech startup founder," "conservative banker," "make a persona for a friendly HR manager"). If the description is brief, use your general knowledge to develop a plausible and rich persona based on that core idea.

Here is the information about the client:
--- CLIENT DESCRIPTION START ---
{client_description}
--- CLIENT DESCRIPTION END ---

Here is supplemental information about the client's communication style, potentially extracted from transcripts or other documents (if available):
--- CLIENT TONE AND COMMUNICATION STYLE (from transcripts/documents) ---
{client_tone_from_files}
--- CLIENT TONE AND COMMUNICATION STYLE END ---

Based on ALL the provided information, write a single, compelling paragraph that serves as an AI agent's system prompt or internal monologue. This paragraph should:
- Embody the client's core persona: their mindset, primary concerns, values, and decision-making style.
- Capture the client's typical communication style: tone, formality, and common expressions (if discernible).
- Be written as if it's the client speaking or thinking to themselves, guiding their actions. It should be a natural internal narrative, NOT a list of instructions for the AI.
- Enable another AI agent, when given this paragraph as its persona, to interact with documents, data, and people as if it *were* this specific client.
- Be rich and descriptive, yet concise.

Generated Client Persona Prompt:
"""
        self.prompt_template = PromptTemplate(
            input_variables=["client_description", "client_tone_from_files"], template=template,
        )

    def _create_chain(self) -> None:
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=self.verbose)

    def _extract_text_from_file(self, file_path: str) -> str:
        # ... (file extraction logic remains the same)
        if not file_path or not os.path.exists(file_path): return ""
        content = ""; ext = os.path.splitext(file_path)[-1].lower()
        try:
            if ext == ".pdf":
                with fitz.open(file_path) as doc: content = "".join(p.get_text() for p in doc)
            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f: content = f.read()
            elif ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f: data = json.load(f)
                # ... (json to text heuristic as before)
                text_parts = []
                if isinstance(data, list): 
                    for item in data:
                        if isinstance(item, dict):
                            for k_item in ['text', 'transcript', 'line', 'message']:
                                if k_item in item: text_parts.append(str(item[k_item])); break
                            else: text_parts.extend(map(str, item.values()))
                        else: text_parts.append(str(item))
                elif isinstance(data, dict): text_parts.extend(f"{k}: {v}" for k,v in data.items())
                else: text_parts.append(str(data))
                content = "\n".join(filter(None,text_parts)) or json.dumps(data, indent=2)
            else: logger.warning(f"{self.name}: Unsupported file: {file_path}"); return ""
            return content.strip()
        except Exception as e: logger.error(f"{self.name}: Error reading {file_path}: {e}"); return ""


    def _extract_all_files_text(self, file_paths: List[str]) -> str:
        all_texts = [f"--- Content from file: {os.path.basename(p)} ---\n{c}" 
                     for p in file_paths if (c := self._extract_text_from_file(p))]
        return "\n\n".join(all_texts).strip()

    def run(self, client_description: str, transcript_file_paths: Optional[List[str]] = None) -> str:
        final_client_desc = client_description.strip() if client_description else ""
        client_tone_files = self._extract_all_files_text(transcript_file_paths or [])

        # Allow short natural language for client_description. min_length 3 for "CEO" etc.
        is_desc_valid = final_client_desc and is_input_valid(final_client_desc, min_length=3, allow_short_natural_lang=True)
        is_files_valid = client_tone_files and is_input_valid(client_tone_files, min_length=20) # Files should have more substance

        if not is_desc_valid and not is_files_valid:
            logger.error(f"{self.name}: Client desc and file content insufficient.")
            return "Error: Client description and transcript file content are both insufficient or non-meaningful."
        
        desc_for_llm = final_client_desc if is_desc_valid else "No explicit client description or it was too brief/non-meaningful. Infer from file content if available or use general knowledge for common roles."
        
        if self.verbose:
            logger.info(f"{self.name}: Desc for LLM: {desc_for_llm[:100]}..., Files text preview: {client_tone_files[:100]}...")

        input_data = {
            "client_description": desc_for_llm,
            "client_tone_from_files": client_tone_files if is_files_valid else "No meaningful transcript/document content provided."
        }
        
        try:
            if hasattr(self.chain, 'invoke'):
                res_dict = self.chain.invoke(input_data); response = res_dict.get('text', str(res_dict))
            else: response = self.chain.run(input_data)
            logger.info(f"{self.name}: Generated client prompt.")
            return response.strip() 
        except Exception as e:
            logger.exception(f"{self.name}: LLM chain error: {e}")
            return f"Error: Could not generate client prompt. Details: {e}"