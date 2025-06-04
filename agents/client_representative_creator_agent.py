
# import os
# from typing import List, Optional
# from dotenv import load_dotenv
# # Assuming OpenAI and ChatOpenAI are correctly imported from langchain_openai
# # For older Langchain versions, it might be langchain_community.llms and langchain_community.chat_models
# try:
#     from langchain_openai import OpenAI, ChatOpenAI
# except ImportError:
#     from langchain_community.llms import OpenAI
#     from langchain_community.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# import fitz  # PyMuPDF
# import json
# import logging
# import re

# load_dotenv()
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # Basic logging configuration for visibility

# def is_input_valid(text: str, min_length: int = 10, short_text_threshold: int = 20, min_unique_chars_for_short_text: int = 3, allow_short_natural_lang: bool = False) -> bool:
#     if not text or not text.strip(): return False
#     stripped_text = text.strip()
#     if allow_short_natural_lang and len(stripped_text) < min_length:
#         # For short natural language, ensure it's not just punctuation or a single repeating char
#         alnum_present = any(c.isalnum() for c in stripped_text)
#         if not alnum_present and len(stripped_text) > 0: return False # e.g. "---"
#         if alnum_present and len(set(''.join(filter(str.isalnum, stripped_text)).lower())) < min_unique_chars_for_short_text and len(stripped_text) < short_text_threshold :
#              # e.g. "aaa" with min_unique_chars_for_short_text = 3 and length < threshold
#              if len(set(''.join(filter(str.isalnum, stripped_text)).lower())) < 1 : return False # ensure at least one unique alnum if alnum_present
#         return bool(len(stripped_text) >=1 and alnum_present) # Must have at least one char and some alnum
#     if len(stripped_text) < min_length: return False
#     if len(stripped_text) <= short_text_threshold:
#         alnum = ''.join(filter(str.isalnum, stripped_text))
#         if not alnum and len(stripped_text) > 0: return False
#         elif alnum and len(set(alnum.lower())) < min_unique_chars_for_short_text: return False
#     return True

# class ClientRepresentativeCreatorAgent:
#     def __init__(self, verbose: bool = False):
#         self.name = "AI Persona Model Creator" # Updated name
#         self.verbose = verbose
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key:
#             logger.error(f"{self.name}: OpenAI API key not found.")
#             raise ValueError("OPENAI_API_KEY not found.")
        
#         try:
#             self.llm = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=0.7, model_name="gpt-3.5-turbo")
#         except Exception as e:
#             logger.exception(f"{self.name}: Error initializing LLM: {e}")
#             raise

#         self._create_prompt()
#         self._create_chain()
#         if self.verbose:
#             logger.info(f"{self.name} initialized.")

#     def _create_prompt(self) -> None:
#         # This template is now designed to generate the full Christian Roth persona model
#         # as per your "Part 2: Persona Construction Template"
#         template = """You are an expert AI assistant tasked with generating a detailed AI persona agent model.
# Your goal is to construct the complete persona for an AI agent that will simulate **Christian Roth, Senior Partner at LEA Partners**, in his role evaluating CEO candidates for a midsize SaaS company.
# The persona model you generate MUST strictly follow the structure and include all predefined instructional text as specified below.
# The content for "Persona Summary", "Evaluation Heuristics", "Success Markers", and "Cultural Fit Filters" sections should be derived and synthesized from the "Source Insights" provided at the end.

# --- START OF AI AGENT PERSONA MODEL FOR CHRISTIAN ROTH ---

# You are Christian Roth, senior Partner at LEA Partners. Before you start any new task, always first read the document "LEA Partner and Christian Roth Summary Information" to familiarise yourself with your base persona. Please act according to the Persona.

# Before evaluating any CEO candidate:
# 1. Always first read the document "ACME CEO Job Description"
# 2. Always first ask if there is additional context you should consider before giving an opinion.
# 3. Always provide your answer in English, even if the candidate CV is in German.

# When evaluating potential ACME CEO candidates you will use the following rules to do so. In addition to these rules you will use your persona to act in accordance to your experience.

# 👤 Persona Summary:
# [Your task: Generate a 1-paragraph summary here based on the Source Insights. This summary should capture Christian Roth's tone, strategic lens, and priorities as a PE Senior Partner evaluating SaaS CEOs.
# Example of expected style: "Christian Roth evaluates CEO candidates through a high-performance private equity lens, with a clear focus on execution, speed, and strategic clarity. His tone is pragmatic and incisive, with a bias for data-driven decision makers who can thrive in messy, ambiguous environments. He prioritizes leaders who are hands-on, capable of post-founder transformation, and emotionally intelligent enough to maintain momentum without alienating legacy teams. His strategic lens is deeply shaped by growth mandates and exit-readiness, and he brings a sharp sensitivity to contextual fit—especially in culturally unstructured, resource-lean SaaS companies."]

# 📊 Evaluation Heuristics
# [Your task: Generate a table of 5–8 evaluation rules with rationales, derived from the Source Insights. The rules should reflect Christian Roth's evaluation criteria for SaaS CEOs.
# Follow this Markdown table format:
# | Rule | Evaluation Principle | Rationale |
# |---|---|---|
# | 1. | [Principle 1 based on Source Insights] | [Rationale 1 based on Source Insights] |
# | 2. | [Principle 2 based on Source Insights] | [Rationale 2 based on Source Insights] |
# ... (up to 8 rules)
# Example of expected style for a rule:
# | 1. | SaaS-native, metric-fluent | Prioritize candidates who speak in cohorts, CAC, retention, and expansion metrics. Demonstrates deep operating fluency; must lead via KPIs in a PE-backed context. |
# ]

# 🏆 Success Markers
# [Your task: Generate a bulleted list of 3-7 observable signs of high potential in CEO candidates, derived from the Source Insights. These should align with Christian Roth's perspective.
# Example of expected style:
# - Deep command of SaaS KPIs without needing prompts
# - Fast-cycle decision making with strong rationale
# - Track record of post-founder growth leadership
# - ...
# ]

# 🌍 Cultural Fit Filters
# [Your task: Generate bulleted lists for "Will work" and "Won't work" beliefs or attitudes in the organization's culture, derived from the Source Insights. These should reflect Christian Roth's view on cultural compatibility.
# Example of expected style:
# **Will work:**
# - [Belief/Attitude 1 from Source Insights]
# - [Belief/Attitude 2 from Source Insights]
# **Won't work:**
# - [Belief/Attitude 3 from Source Insights]
# - [Belief/Attitude 4 from Source Insights]
# ]

# --- END OF AI AGENT PERSONA MODEL FOR CHRISTIAN ROTH ---

# Source Insights (Information about Christian Roth, LEA Partners, ACME's needs, and any other relevant context for building his persona):
# --- SOURCE INSIGHTS START ---
# {source_insights_text}
# --- SOURCE INSIGHTS END ---

# Now, generate the complete "AI AGENT PERSONA MODEL FOR CHRISTIAN ROTH" by filling in the bracketed sections based *only* on the "Source Insights" provided above. Adhere strictly to the specified structure and formatting, including the predefined instructional text.
# """
#         self.prompt_template = PromptTemplate(
#             input_variables=["source_insights_text"], template=template,
#         )

#     def _create_chain(self) -> None:
#         self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=self.verbose)

#     def _extract_text_from_file(self, file_path: str) -> str:
#         if not file_path or not os.path.exists(file_path): return ""
#         content = ""; ext = os.path.splitext(file_path)[-1].lower()
#         try:
#             if ext == ".pdf":
#                 with fitz.open(file_path) as doc: content = "".join(p.get_text() for p in doc)
#             elif ext == ".txt":
#                 with open(file_path, "r", encoding="utf-8") as f: content = f.read()
#             elif ext == ".json":
#                 with open(file_path, "r", encoding="utf-8") as f: data = json.load(f)
#                 text_parts = []
#                 if isinstance(data, list): 
#                     for item in data:
#                         if isinstance(item, dict):
#                             # Prioritize common keys for textual content
#                             for k_item in ['text', 'transcript', 'line', 'message', 'content', 'summary']:
#                                 if k_item in item and isinstance(item[k_item], str): text_parts.append(item[k_item]); break
#                             else: # Fallback to joining all string values if specific keys not found
#                                 text_parts.extend(str(v) for v in item.values() if isinstance(v, str))
#                         elif isinstance(item, str): text_parts.append(item)
#                         # else: text_parts.append(str(item)) # Avoid overly verbose stringification of complex objects
#                 elif isinstance(data, dict): 
#                     for k_item in ['text', 'transcript', 'full_text', 'summary', 'content', 'description']:
#                         if k_item in data and isinstance(data[k_item], str): text_parts.append(data[k_item]); break
#                     else: # Fallback to key-value pairs for simpler dicts or specific relevant keys
#                         text_parts.extend(f"{k}: {v}" for k,v in data.items() if isinstance(v, (str, int, float, bool)))
#                 # else: text_parts.append(str(data)) # Avoid general stringification
#                 content = "\n".join(filter(None,text_parts)) if text_parts else json.dumps(data, indent=2) # Fallback to pretty JSON if no text extracted
#             else:
#                 logger.warning(f"{self.name}: Unsupported file type: {file_path} for direct text extraction.")
#                 return ""
#             return content.strip()
#         except Exception as e:
#             logger.error(f"{self.name}: Error reading or processing file {file_path}: {e}")
#             return ""

#     def _extract_all_files_text(self, file_paths: List[str]) -> str:
#         all_texts = []
#         for p in file_paths:
#             if self.verbose: logger.info(f"Attempting to extract text from: {p}")
#             content = self.extract_text_from_file(p) # Corrected: call instance method
#             if content:
#                 all_texts.append(f"--- Content from file: {os.path.basename(p)} ---\n{content}")
#             elif self.verbose:
#                 logger.info(f"No content extracted or file unsupported: {p}")

#         return "\n\n".join(all_texts).strip()

#     # Make extract_text_from_file an instance method to be callable by _extract_all_files_text
#     def extract_text_from_file(self, file_path: str) -> str:
#         return self._extract_text_from_file(file_path)


#     def run(self, client_description: str, transcript_files: Optional[List[str]] = None) -> str:
#         # client_description is now the primary "Source Insights"
#         # transcript_file_paths provide supplemental "Source Insights"

#         primary_insights = client_description.strip() if client_description else ""
#         supplemental_insights_from_files = self._extract_all_files_text(transcript_files or [])

#         # Validation for primary insights (min_length=3 allows for brief core concepts like "PE Investor")
#         is_primary_valid = primary_insights and is_input_valid(primary_insights, min_length=3, allow_short_natural_lang=True)
#         # Validation for supplemental insights (min_length=20 as it's expected to be more substantial if provided)
#         is_supplemental_valid = supplemental_insights_from_files and is_input_valid(supplemental_insights_from_files, min_length=20)

#         source_insights_parts = []
#         if is_primary_valid:
#             source_insights_parts.append(f"Core Description/Instructions for Persona Generation:\n{primary_insights}")
#         if is_supplemental_valid:
#             source_insights_parts.append(f"Supplemental Information from Documents for Persona Generation:\n{supplemental_insights_from_files}")

#         if not source_insights_parts:
#             logger.error(f"{self.name}: No valid source insights provided from description or files.")
#             return "Error: Source insights (from description and/or files) are insufficient or non-meaningful to generate the persona."

#         final_source_insights_text = "\n\n".join(source_insights_parts)
        
#         if self.verbose:
#             logger.info(f"{self.name}: Final Source Insights for LLM (preview): {final_source_insights_text[:300]}...")

#         input_data = {
#             "source_insights_text": final_source_insights_text
#         }
        
#         try:
#             response_raw: str
#             if hasattr(self.chain, 'invoke'): # For newer LangChain versions
#                 res_dict = self.chain.invoke(input_data)
#                 response_raw = res_dict.get('text', str(res_dict))
#             else: # Fallback for older LangChain versions
#                 response_raw = self.chain.run(input_data) # type: ignore
            
#             # Extract only the persona model part, as LLMs might add preamble/postamble
#             start_marker = "--- START OF AI AGENT PERSONA MODEL FOR CHRISTIAN ROTH ---"
#             end_marker = "--- END OF AI AGENT PERSONA MODEL FOR CHRISTIAN ROTH ---"
            
#             start_index = response_raw.find(start_marker)
#             end_index = response_raw.rfind(end_marker) # Use rfind for end marker in case it appears in source text

#             if start_index != -1 and end_index != -1 and start_index < end_index:
#                 extracted_response = response_raw[start_index : end_index + len(end_marker)]
#                 logger.info(f"{self.name}: Successfully generated and extracted Christian Roth persona model.")
#                 return extracted_response.strip()
#             elif start_index != -1:
#                 logger.warning(f"{self.name}: Found start marker but not end marker in LLM response. Returning from start marker onwards.")
#                 return response_raw[start_index:].strip()
#             else:
#                 logger.warning(f"{self.name}: Persona model markers not found in LLM response. Returning raw response. This might need review or prompt adjustment.")
#                 if self.verbose: logger.debug(f"Raw LLM response for persona generation: {response_raw}")
#                 return response_raw.strip()

#         except Exception as e:
#             logger.exception(f"{self.name}: LLM chain error during persona generation: {e}")
#             return f"Error: Could not generate Christian Roth persona model. Details: {e}"
import os
from typing import List, Optional
from dotenv import load_dotenv
try:
    from langchain_openai import OpenAI, ChatOpenAI
except ImportError:
    from langchain_community.llms import OpenAI
    from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import fitz  # PyMuPDF
import json
import logging
import re

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def is_input_valid(text: str, min_length: int = 10, short_text_threshold: int = 20, min_unique_chars_for_short_text: int = 3, allow_short_natural_lang: bool = False) -> bool:
    if not text or not text.strip(): return False
    stripped_text = text.strip()
    if allow_short_natural_lang and len(stripped_text) < min_length:
        # For short natural language, ensure it's not just punctuation or a single repeating char
        alnum_present = any(c.isalnum() for c in stripped_text)
        if not alnum_present and len(stripped_text) > 0: return False
        if alnum_present and len(set(''.join(filter(str.isalnum, stripped_text)).lower())) < min_unique_chars_for_short_text and len(stripped_text) < short_text_threshold :
             # e.g. "aaa" with min_unique_chars_for_short_text = 3 and length < threshold
             if len(set(''.join(filter(str.isalnum, stripped_text)).lower())) < 1 : return False # ensure at least one unique alnum if alnum_present
        return bool(len(stripped_text) >=1 and alnum_present) # Must have at least one char and some alnum
    if len(stripped_text) < min_length: return False
    if len(stripped_text) <= short_text_threshold:
        alnum = ''.join(filter(str.isalnum, stripped_text))
        if not alnum and len(stripped_text) > 0: return False
        elif alnum and len(set(alnum.lower())) < min_unique_chars_for_short_text: return False
    return True


class ClientRepresentativeCreatorAgent:
    def __init__(self, verbose: bool = False):
        self.name = "AI Persona Model Creator"
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error(f"{self.name}: OpenAI API key not found.")
            raise ValueError("OPENAI_API_KEY not found.")

        try:
            self.llm = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=0.7, model_name="gpt-3.5-turbo")
        except Exception as e:
            logger.exception(f"{self.name}: Error initializing LLM: {e}")
            raise

        self._create_prompt()
        self._create_chain()
        if self.verbose:
            logger.info(f"{self.name} initialized")

    def _create_prompt(self) -> None:
        template = """You are an expert AI assistant tasked with generating a detailed AI persona agent model.
Your goal is to construct the complete persona for an AI agent that reflects the characteristics and priorities described in the Source Insights.
The persona model you generate MUST strictly follow the structure and include all predefined instructional text as specified below.
The content for "Persona Summary", "Evaluation Heuristics", "Success Markers", and "Cultural Fit Filters" sections should be derived and synthesized from the "Source Insights" provided at the end.

--- START OF AI AGENT PERSONA MODEL ---

You are an evaluator of CEO candidates. Please act according to the Persona described in the Source Insights.

Before evaluating any CEO candidate:
1. Always first ask if there is additional context you should consider before giving an opinion.
2. Always provide your answer in English.

When evaluating potential CEO candidates you will use the following rules to do so. In addition to these rules you will use your persona to act in accordance to your experience.

👤 Persona Summary:
[Your task: Generate a 1-paragraph summary here based on the Source Insights. This summary should capture the evaluator's tone, strategic lens, and priorities.
Example of expected style: "This evaluator assesses CEO candidates through a lens of long-term sustainability and ethical leadership. Their tone is thoughtful and deliberate, with a preference for leaders who prioritize employee well-being and community impact. They value leaders who are visionary, collaborative, and adaptable to changing market conditions."]

📊 Evaluation Heuristics
[Your task: Generate a table of 5–8 evaluation rules with rationales, derived from the Source Insights. The rules should reflect the evaluator's criteria.
Follow this Markdown table format:
| Rule | Evaluation Principle | Rationale |
|---|---|---|
| 1. | [Principle 1 based on Source Insights] | [Rationale 1 based on Source Insights] |
| 2. | [Principle 2 based on Source Insights] | [Rationale 2 based on Source Insights] |
... (up to 8 rules)
Example of expected style for a rule:
| 1. | Ethical leadership | Prioritize candidates who demonstrate a strong commitment to ethical business practices and social responsibility. |
]

🏆 Success Markers
[Your task: Generate a bulleted list of 3-7 observable signs of high potential in CEO candidates, derived from the Source Insights. These should align with the evaluator's perspective.
Example of expected style:
- Articulates a clear vision for the company's future
- Demonstrates a commitment to ethical leadership
- Prioritizes employee well-being and community impact
- ...
]

🌍 Cultural Fit Filters
[Your task: Generate bulleted lists for "Will work" and "Won't work" beliefs or attitudes in the organization's culture, derived from the Source Insights. These should reflect the evaluator's view on cultural compatibility.
Example of expected style:
**Will work:**
- Commitment to ethical behavior
- Collaborative leadership style
**Won't work:**
- Prioritization of short-term profits over long-term sustainability
- Authoritarian management style
]

--- END OF AI AGENT PERSONA MODEL ---

Source Insights:
--- SOURCE INSIGHTS START ---
{source_insights_text}
--- SOURCE INSIGHTS END ---

Now, generate the complete "AI AGENT PERSONA MODEL" by filling in the bracketed sections based *only* on the "Source Insights" provided above. Adhere strictly to the specified structure and formatting, including the predefined instructional text.
"""
        self.prompt_template = PromptTemplate(
            input_variables=["source_insights_text"], template=template,
        )

    def _create_chain(self) -> None:
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=self.verbose)

    def _extract_text_from_file(self, file_path: str) -> str:
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"{self.name}: File not found: {file_path}")
            return ""
        content = ""; ext = os.path.splitext(file_path)[-1].lower()
        try:
            if ext == ".pdf":
                with fitz.open(file_path) as doc: content = "".join(p.get_text() for p in doc)
            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f: content = f.read()
            elif ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f: data = json.load(f)
                text_parts = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            # Prioritize common keys for textual content
                            for k_item in ['text', 'transcript', 'line', 'message', 'content', 'summary']:
                                if k_item in item and isinstance(item[k_item], str): text_parts.append(item[k_item]); break
                            else:  # Fallback to joining all string values if specific keys not found
                                text_parts.extend(str(v) for v in item.values() if isinstance(v, str))
                        elif isinstance(item, str): text_parts.append(item)
                        # else: text_parts.append(str(item)) # Avoid overly verbose stringification of complex objects
                elif isinstance(data, dict):
                    for k_item in ['text', 'transcript', 'full_text', 'summary', 'content', 'description']:
                        if k_item in data and isinstance(data[k_item], str): text_parts.append(data[k_item]); break
                    else:  # Fallback to joining all string values if specific relevant keys
                        text_parts.extend(f"{k}: {v}" for k, v in data.items() if isinstance(v, (str, int, float, bool)))
                # else: text_parts.append(str(data)) # Avoid general stringification
                content = "\n".join(filter(None, text_parts)) if text_parts else json.dumps(data, indent=2)  # Fallback to pretty JSON if no text extracted
            else:
                logger.warning(f"{self.name}: Unsupported file type: {file_path} for direct text extraction.")
                return ""
            return content.strip()
        except Exception as e:
            logger.error(f"{self.name}: Error reading or processing file {file_path}: {e}")
            return ""

    def _extract_all_files_text(self, file_paths: List[str]) -> str:
        all_texts = []
        for p in file_paths:
            if self.verbose: logger.info(f"Attempting to extract text from: {p}")
            content = self.extract_text_from_file(p)  # Corrected: call instance method
            if content:
                all_texts.append(f"--- Content from file: {os.path.basename(p)} ---\n{content}")
            elif self.verbose:
                logger.info(f"No content extracted or file unsupported: {p}")

        return "\n\n".join(all_texts).strip()

    def extract_text_from_file(self, file_path: str) -> str:
        return self._extract_text_from_file(file_path)

    def run(self, client_description: str = "", transcript_file_paths: Optional[List[str]] = None) -> str:
        primary_insights = client_description.strip() if client_description else ""
        supplemental_insights_from_files = self._extract_all_files_text(transcript_file_paths or [])

        is_primary_valid = primary_insights and is_input_valid(primary_insights, min_length=3, allow_short_natural_lang=True)
        is_supplemental_valid = supplemental_insights_from_files and is_input_valid(supplemental_insights_from_files, min_length=20)

        source_insights_parts = []
        if is_primary_valid:
            source_insights_parts.append(f"Core Description/Instructions for Persona Generation:\n{primary_insights}")
        if is_supplemental_valid:
            source_insights_parts.append(f"Supplemental Information from Documents for Persona Generation:\n{supplemental_insights_from_files}")

        if not source_insights_parts:
            error_message = "Error: Source insights (from description and/or files) are insufficient or non-meaningful to generate the persona. Please provide more information."
            logger.error(f"{self.name}: {error_message}")
            return error_message  # Return error string as the API expects.

        final_source_insights_text = "\n\n".join(source_insights_parts)

        if self.verbose:
            logger.info(f"{self.name}: Final Source Insights for LLM (preview): {final_source_insights_text[:300]}...")

        input_data = {
            "source_insights_text": final_source_insights_text,
        }

        try:
            response_raw: str
            if hasattr(self.chain, 'invoke'):
                res_dict = self.chain.invoke(input_data)
                response_raw = res_dict.get('text', str(res_dict))
            else:
                response_raw = self.chain.run(input_data)  # type: ignore

            def extract_persona(text: str) -> str:
                pattern = re.compile(
                    r"--- START OF AI AGENT PERSONA MODEL ---\s*(.*?)\s*--- END OF AI AGENT PERSONA MODEL ---",
                    re.DOTALL
                )
                match = pattern.search(text)
                return match.group(1).strip() if match else text.strip()

            extracted_response = extract_persona(response_raw)

            logger.info(f"{self.name}: Successfully generated persona model.")
            return extracted_response.strip()

        except Exception as e:
            error_message = f"Error: Could not generate persona model. Details: {e}"
            logger.exception(f"{self.name}: LLM chain error during persona generation: {e}")
            return error_message  # Return error string as the API expects