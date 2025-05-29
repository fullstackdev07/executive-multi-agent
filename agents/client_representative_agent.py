# import os
# from typing import Optional, List
# from dotenv import load_dotenv
# from langchain_community.llms import OpenAI # Keep if you prefer, or switch to ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI # Recommended for newer features/models
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# import json
# import fitz  # PyMuPDF for PDF reading
# import logging
# import re

# # Configure logging (if not already configured at app level, this is fine)
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__) # Get logger for this module

# class ClientRepresentativeAgent:
#     def __init__(self, verbose: bool = False):
#         self.verbose = verbose
#         load_dotenv() # Load environment variables
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key:
#             logger.error("ClientRepresentativeAgent: OpenAI API key not found in environment variables.")
#             raise ValueError("OPENAI_API_KEY not found. Set environment variable.")

#         try:
#             # Using ChatOpenAI is generally recommended over the base OpenAI LLM
#             self.llm = ChatOpenAI(
#                 openai_api_key=self.openai_api_key,
#                 temperature=0.7,
#                 max_tokens=3000, # Max output tokens
#                 model_name="gpt-3.5-turbo" # Or gpt-4 if preferred/available
#             )
#             # If you must use the older OpenAI class:
#             # self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7, max_tokens=3000)
#         except Exception as e:
#             logger.exception(f"ClientRepresentativeAgent: Error initializing OpenAI LLM: {e}")
#             raise

#         self._create_extraction_prompt()
#         self._create_feedback_prompt()
#         self._create_chains()
#         if self.verbose:
#             logger.info("ClientRepresentativeAgent initialized.")

#     def _create_extraction_prompt(self):
#         self.extraction_template = PromptTemplate(
#             input_variables=["input_statement"],
#             template="""
#     From the following statement (which may include context, documents, or explicit descriptions), infer the client's key characteristics.
#     Focus on identifying their professional persona, main priorities, core values, and typical communication tone.

#     Statement:
#     --- STATEMENT START ---
#     {input_statement}
#     --- STATEMENT END ---

#     Your goal is to distill this into a structured JSON object.
#     Return ONLY the JSON object with the following keys:
#     {{
#         "client_persona": "A brief description of the client's professional role and general demeanor (e.g., 'Results-oriented executive', 'Detail-focused technical lead', 'Collaborative team player').",
#         "client_priorities": "List 2-3 key priorities the client seems to emphasize (e.g., 'Meeting deadlines', 'Cost efficiency', 'Innovation', 'Quality of deliverables').",
#         "client_values": "Identify 1-2 core values that appear important to the client (e.g., 'Transparency', 'Thoroughness', 'Creativity', 'Practicality').",
#         "client_tone": "Describe the client's typical communication style and tone (e.g., 'Formal and direct', 'Friendly and encouraging', 'Skeptical and questioning', 'Concise and data-driven')."
#     }}
#     """
#         )

#     def _create_feedback_prompt(self):
#         self.feedback_template = PromptTemplate(
#             input_variables=[
#                 "client_persona",
#                 "client_priorities",
#                 "client_values",
#                 "client_tone",
#                 "input_to_review", # This will be the document/text the "client" is reviewing
#             ],
#             template="""
#     You are acting as a Client Representative. Your personality and responses should strictly adhere to the defined client characteristics.

#     Your Client Profile:
#     - Client Persona: {client_persona}
#     - Client Priorities: {client_priorities}
#     - Client Values: {client_values}
#     - Client Tone and Communication Style: {client_tone}

#     You have been provided with the following input to review:
#     --- INPUT TO REVIEW START ---
#     {input_to_review}
#     --- INPUT TO REVIEW END ---

#     Your Task:
#     Provide comprehensive feedback on the "Input to Review" as if you ARE this client.
#     - Adopt the specified client_tone in your language and style.
#     - Evaluate the input based on the client_priorities and client_values.
#     - Offer constructive feedback. Point out what aligns well and what needs improvement.
#     - If appropriate for the persona, ask clarifying questions or express specific concerns.
#     - Be specific in your comments, referencing parts of the input if possible.
#     - Aim for a thorough response of approximately 300-500 words.

#     Your feedback should include:
#     - An overall impression.
#     - Specific points of strength related to priorities/values.
#     - Specific areas for improvement or points of concern, also related to priorities/values.
#     - Any questions you (as the client) might have.
#     - A concluding remark that fits the client_tone.

#     Client Representative Feedback:
#     """
#         )

#     def _create_chains(self):
#         self.extraction_chain = LLMChain(llm=self.llm, prompt=self.extraction_template, verbose=self.verbose)
#         self.feedback_chain = LLMChain(llm=self.llm, prompt=self.feedback_template, verbose=self.verbose)

#     def _read_file_content(self, file_path: str) -> Optional[str]:
#         """Reads content from PDF, TXT, or JSON files."""
#         try:
#             ext = os.path.splitext(file_path)[-1].lower()
#             content = f"\n\n--- Content from file: {os.path.basename(file_path)} ---\n"
#             if ext == ".pdf":
#                 with fitz.open(file_path) as doc:
#                     text = "".join(page.get_text() for page in doc)
#                 content += text
#                 logger.info(f"ClientRepAgent: Successfully extracted text from PDF: {file_path}")
#             elif ext == ".txt":
#                 with open(file_path, "r", encoding="utf-8") as f:
#                     text = f.read()
#                 content += text
#                 logger.info(f"ClientRepAgent: Successfully read text from TXT: {file_path}")
#             elif ext == ".json":
#                 with open(file_path, "r", encoding="utf-8") as f:
#                     json_data = json.load(f)
#                 # Convert JSON to a readable string format for the LLM
#                 content += json.dumps(json_data, indent=2)
#                 logger.info(f"ClientRepAgent: Successfully read and formatted JSON from: {file_path}")
#             else:
#                 logger.warning(f"ClientRepAgent: Skipping unsupported file type: {file_path}")
#                 return None # Indicate unsupported type
#             return content
#         except FileNotFoundError:
#             logger.error(f"ClientRepAgent: File not found: {file_path}")
#         except Exception as e:
#             logger.error(f"ClientRepAgent: Failed to read or process file {file_path}: {e}")
#         return "" # Return empty string on error to avoid None issues later, or None if strict handling preferred

#     def run(self, input_statement: str, transcript_file_paths: Optional[List[str]] = None) -> str:
#         if self.verbose:
#             logger.info(f"ClientRepAgent: Running with input_statement (preview): {input_statement[:100]}...")
#             logger.info(f"ClientRepAgent: Processing {len(transcript_file_paths) if transcript_file_paths else 0} transcript files.")

#         # The `input_statement` is the primary text to analyze for client characteristics OR the document to review.
#         # The prompt structure implies that `input_statement` for the extraction_chain should contain client info.
#         # And `input_to_review` for feedback_chain is the document.
#         # In the orchestrator, we decided `input_statement` for feedback generation includes both persona and reviewable doc.
#         # Let's assume `input_statement` here is the text that *describes* the client or from which characteristics are inferred.
#         # And `transcript_file_paths` provide additional context for this inference.

#         # For extraction phase, combine input_statement and file contents.
#         # The `input_statement` passed by orchestrator is `feedback_input_statement`
#         # which has `---CLIENT PERSONA GUIDANCE---` and `---DOCUMENT TO REVIEW---`
#         # The extraction prompt needs to work on the "CLIENT PERSONA GUIDANCE" part.

#         # Let's refine: the `input_statement` passed to this agent's `run` method will be the one from the orchestrator.
#         # This statement ALREADY contains:
#         # 1. Guidance on client persona (from client_prompt, which itself was derived from market_report etc.)
#         # 2. The document to be reviewed (e.g., the interview_report).
#         # So, `input_statement` is the "document to review" and also contains info to infer persona.
        
#         client_characteristics_source = input_statement # Primary source for characteristics
#         document_to_review_by_client = input_statement # The same input is reviewed

#         if transcript_file_paths:
#             for file_path in transcript_file_paths:
#                 file_content = self._read_file_content(file_path)
#                 if file_content:
#                     # Append file content to the source for characteristics extraction
#                     client_characteristics_source += f"\n\n--- Additional Context from File: {os.path.basename(file_path)} ---\n{file_content}"


#         logger.info(f"ClientRepAgent: Combined input for persona extraction (preview): {client_characteristics_source[:200]}...")

#         # Step 1: Extract client persona, tone, values, priorities
#         try:
#             # The extraction chain is run on the combined text.
#             # The prompt guides it to find persona, priorities, values, tone.
#             raw_extracted_json_str = self.extraction_chain.run({"input_statement": client_characteristics_source})
            
#             # LLMs sometimes add markdown or other text around JSON. Try to clean it.
#             match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_extracted_json_str, re.IGNORECASE)
#             if match:
#                 extracted_json_str = match.group(1)
#             else:
#                 # If no markdown, assume it might be just the JSON or JSON with leading/trailing whitespace
#                 extracted_json_str = raw_extracted_json_str.strip()

#             logger.info(f"ClientRepAgent: Raw extracted JSON string from LLM: {raw_extracted_json_str}")
#             logger.info(f"ClientRepAgent: Cleaned extracted JSON string for parsing: {extracted_json_str}")

#             if self.verbose:
#                 print("ClientRepAgent Extracted Traits (Raw):\n", raw_extracted_json_str)
#         except Exception as extraction_error:
#             logger.error(f"ClientRepAgent: Error during extraction chain run: {extraction_error}")
#             return f"Error: Could not extract client characteristics. Details: {extraction_error}"

#         try:
#             extracted_data = json.loads(extracted_json_str)
#             # Validate essential keys - though the prompt asks for them, LLM might miss
#             expected_keys = ["client_persona", "client_priorities", "client_values", "client_tone"]
#             for key in expected_keys:
#                 if key not in extracted_data:
#                     logger.warning(f"ClientRepAgent: Extracted data missing key '{key}'. Using default.")
#                     extracted_data[key] = f"Not specified or inferable (default for {key})"

#         except json.JSONDecodeError as e:
#             logger.error(f"ClientRepAgent: Failed to parse extracted client data (JSONDecodeError): {e}. Raw string: '{extracted_json_str}'")
#             # Fallback: try to use the raw string if it looks somewhat like a description, or provide a generic error
#             # This is risky as it might pass a non-JSON string to the next chain. Better to error out.
#             return f"Error: Failed to parse client characteristics from LLM output. JSON was malformed. Raw output: {extracted_json_str}"
#         except Exception as e: # Catch any other parsing errors
#             logger.error(f"ClientRepAgent: Unexpected error parsing extracted client data: {e}")
#             return f"Error: Unexpected issue processing client characteristics. Details: {e}"

#         # Step 2: Generate feedback using inferred traits
#         feedback_input_payload = {
#             "client_persona": extracted_data.get("client_persona", "Default Persona: Professional and Objective"),
#             "client_priorities": extracted_data.get("client_priorities", "Default Priorities: Clarity, Accuracy, Relevance"),
#             "client_values": extracted_data.get("client_values", "Default Values: Professionalism, Usefulness"),
#             "client_tone": extracted_data.get("client_tone", "Default Tone: Constructive and Polite"),
#             "input_to_review": document_to_review_by_client.strip(), # The original input statement is what's reviewed
#         }
        
#         if self.verbose:
#             logger.info(f"ClientRepAgent: Payload for feedback generation: {feedback_input_payload}")

#         try:
#             feedback_response = self.feedback_chain.run(feedback_input_payload)
#             # If using ChatOpenAI and the chain returns a dict:
#             # feedback_response_dict = self.feedback_chain.invoke(feedback_input_payload)
#             # feedback_response = feedback_response_dict.get('text', "Error: No feedback text generated.")

#             logger.info("ClientRepAgent: Feedback generated successfully.")
#             return feedback_response
#         except Exception as feedback_error:
#             logger.error(f"ClientRepAgent: Error during feedback chain run: {feedback_error}")
#             return f"Error: Could not generate client feedback. Details: {feedback_error}"
import os
from typing import Optional, List
from dotenv import load_dotenv
from langchain_community.llms import OpenAI 
from langchain_community.chat_models import ChatOpenAI 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import fitz  
import logging
import re

logger = logging.getLogger(__name__)

def is_input_valid(text: str, min_length: int = 10, short_text_threshold: int = 20, min_unique_chars_for_short_text: int = 3, allow_short_natural_lang: bool = False) -> bool:
    if not text or not text.strip():
        return False
    stripped_text = text.strip()
    
    # If short natural language is allowed, very short inputs might be valid commands
    if allow_short_natural_lang and len(stripped_text) < min_length:
        # Basic check for some alphanumeric content if it's very short
        if len(stripped_text) >= 1 and any(char.isalnum() for char in stripped_text):
            return True # e.g., "Client is tech CEO" could be valid short description
        else:
            return False

    if len(stripped_text) < min_length:
        return False
    if len(stripped_text) <= short_text_threshold:
        alnum_text_part = ''.join(filter(str.isalnum, stripped_text))
        if not alnum_text_part and len(stripped_text) > 0:
            return False
        elif alnum_text_part and len(set(alnum_text_part.lower())) < min_unique_chars_for_short_text:
            return False
    return True

class ClientRepresentativeAgent:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        load_dotenv() 
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("ClientRepresentativeAgent: OpenAI API key not found.")
            raise ValueError("OPENAI_API_KEY not found. Set environment variable.")

        try:
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                temperature=0.7,
                max_tokens=3000, 
                model_name="gpt-3.5-turbo" 
            )
        except Exception as e:
            logger.exception(f"ClientRepresentativeAgent: Error initializing OpenAI LLM: {e}")
            raise

        self._create_extraction_prompt()
        self._create_feedback_prompt()
        self._create_chains()
        if self.verbose:
            logger.info("ClientRepresentativeAgent initialized.")

    def _create_extraction_prompt(self):
        self.extraction_template = PromptTemplate(
            input_variables=["input_statement"],
            template="""
    From the following statement (which may include context, documents, explicit descriptions, or even a brief natural language request about a client type), infer the client's key characteristics.
    If the statement is very brief (e.g., "a skeptical investor" or "busy executive persona"), use your general knowledge to flesh out the persona based on that core idea.

    Statement:
    --- STATEMENT START ---
    {input_statement}
    --- STATEMENT END ---

    Your goal is to distill this into a structured JSON object.
    Return ONLY the JSON object with the following keys:
    {{
        "client_persona": "A brief description of the client's professional role and general demeanor (e.g., 'Results-oriented executive', 'Detail-focused technical lead', 'Collaborative team player').",
        "client_priorities": "List 2-3 key priorities the client seems to emphasize (e.g., 'Meeting deadlines', 'Cost efficiency', 'Innovation', 'Quality of deliverables').",
        "client_values": "Identify 1-2 core values that appear important to the client (e.g., 'Transparency', 'Thoroughness', 'Creativity', 'Practicality').",
        "client_tone": "Describe the client's typical communication style and tone (e.g., 'Formal and direct', 'Friendly and encouraging', 'Skeptical and questioning', 'Concise and data-driven')."
    }}
    """
        )

    def _create_feedback_prompt(self):
        self.feedback_template = PromptTemplate(
            input_variables=[
                "client_persona", "client_priorities", "client_values",
                "client_tone", "input_to_review", 
            ],
            template="""
    You are acting as a Client Representative. Your personality and responses should strictly adhere to the defined client characteristics.

    Your Client Profile:
    - Client Persona: {client_persona}
    - Client Priorities: {client_priorities}
    - Client Values: {client_values}
    - Client Tone and Communication Style: {client_tone}

    You have been provided with the following input to review. This input might be a formal document, or it could be a brief statement if the original request was very short.
    --- INPUT TO REVIEW START ---
    {input_to_review}
    --- INPUT TO REVIEW END ---

    Your Task:
    Provide comprehensive feedback on the "Input to Review" as if you ARE this client.
    - Adopt the specified client_tone in your language and style.
    - Evaluate the input based on the client_priorities and client_values.
    - Offer constructive feedback. Point out what aligns well and what needs improvement.
    - If appropriate for the persona, ask clarifying questions or express specific concerns.
    - Be specific in your comments, referencing parts of the input if possible.
    - Aim for a thorough response of approximately 300-500 words.

    Your feedback should include:
    - An overall impression.
    - Specific points of strength related to priorities/values.
    - Specific areas for improvement or points of concern, also related to priorities/values.
    - Any questions you (as the client) might have.
    - A concluding remark that fits the client_tone.

    Client Representative Feedback:
    """
        )

    def _create_chains(self):
        self.extraction_chain = LLMChain(llm=self.llm, prompt=self.extraction_template, verbose=self.verbose)
        self.feedback_chain = LLMChain(llm=self.llm, prompt=self.feedback_template, verbose=self.verbose)

    def _read_file_content(self, file_path: str) -> Optional[str]:
        # ... (file reading logic remains the same)
        try:
            ext = os.path.splitext(file_path)[-1].lower()
            header = f"\n\n--- Content from file: {os.path.basename(file_path)} ---\n"
            content_text = ""
            if ext == ".pdf":
                with fitz.open(file_path) as doc: content_text = "".join(page.get_text() for page in doc)
            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f: content_text = f.read()
            elif ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f: json_data = json.load(f)
                content_text = json.dumps(json_data, indent=2)
            else: logger.warning(f"ClientRepAgent: Skipping unsupported file: {file_path}"); return None
            return header + content_text.strip() if content_text.strip() else ""
        except Exception as e: logger.error(f"ClientRepAgent: Error reading {file_path}: {e}"); return ""

    def run(self, input_statement: str, transcript_file_paths: Optional[List[str]] = None) -> str:
        if self.verbose:
            logger.info(f"ClientRepAgent: Running with input_statement (preview): {input_statement[:100]}...")
        
        # Allow shorter, natural language for input_statement when it's about persona
        is_statement_weak = input_statement and not is_input_valid(input_statement, min_length=5, allow_short_natural_lang=True)
        
        if is_statement_weak and not transcript_file_paths:
             logger.error(f"ClientRepAgent: input_statement is invalid/empty and no files: {input_statement[:100]}")
             return "Error: Input statement is insufficient or non-meaningful, and no supporting files provided."

        client_characteristics_source = input_statement.strip() if input_statement else ""
        
        if transcript_file_paths:
            for file_path in transcript_file_paths:
                file_content = self._read_file_content(file_path)
                if file_content: client_characteristics_source += file_content

        if not client_characteristics_source.strip():
            logger.error("ClientRepAgent: Combined input for persona extraction is empty.")
            return "Error: No usable content for client characteristic extraction."
        
        # Validate combined source - be more lenient if initial input_statement was very short (natural lang)
        # The prompt for extraction already guides the LLM to handle brief inputs.
        if not is_input_valid(client_characteristics_source, min_length=5, allow_short_natural_lang=True, short_text_threshold=50, min_unique_chars_for_short_text=2):
            logger.error(f"ClientRepAgent: Combined input for persona extraction non-meaningful: {client_characteristics_source[:200]}...")
            return "Error: Combined input for client characteristics appears non-meaningful."

        logger.info(f"ClientRepAgent: Combined input for persona extraction (preview): {client_characteristics_source[:200]}...")

        try:
            raw_extracted_json_str = self.extraction_chain.run({"input_statement": client_characteristics_source})
            match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_extracted_json_str, re.IGNORECASE)
            extracted_json_str = match.group(1) if match else raw_extracted_json_str.strip()
            if not (extracted_json_str.startswith("{") and extracted_json_str.endswith("}")):
                json_match_curly = re.search(r"(\{[\s\S]*\})", extracted_json_str)
                if json_match_curly: extracted_json_str = json_match_curly.group(1)
            # ... (logging as before)
        except Exception as extraction_error:
            logger.error(f"ClientRepAgent: Error during extraction chain run: {extraction_error}")
            return f"Error: Could not extract client characteristics. Details: {extraction_error}"

        try:
            extracted_data = json.loads(extracted_json_str)
            for key in ["client_persona", "client_priorities", "client_values", "client_tone"]:
                val = extracted_data.get(key)
                # Allow extracted fields to be short if they are meaningful (e.g. "Innovative")
                if not val or not isinstance(val, str) or not val.strip() or \
                   not is_input_valid(val, min_length=3, allow_short_natural_lang=True, short_text_threshold=15, min_unique_chars_for_short_text=2): # min_length 3 for single words like "CEO" or "Agile"
                    logger.warning(f"ClientRepAgent: Extracted data for '{key}' ('{val}') non-meaningful/missing. Using default.")
                    extracted_data[key] = f"Default value (original content for {key} was insufficient or non-meaningful)"
        except Exception as e: 
            logger.error(f"ClientRepAgent: Error parsing extracted client data: {e}. Raw: '{extracted_json_str}'")
            return f"Error: Failed to parse/validate client characteristics. Raw LLM output: {extracted_json_str}"
        
        doc_to_review_match = re.search(r"---DOCUMENT TO REVIEW---([\s\S]*)", client_characteristics_source, re.IGNORECASE | re.DOTALL)
        actual_document_content = doc_to_review_match.group(1).strip() if doc_to_review_match else input_statement.strip()

        # Document to review should have some substance
        if not is_input_valid(actual_document_content, min_length=10, allow_short_natural_lang=False): # Don't allow short natural lang for the document itself
            logger.error(f"ClientRepAgent: Document for review invalid/empty: {actual_document_content[:100]}...")
            return "Error: Document content for review is missing, too short, or non-meaningful."

        feedback_input_payload = {
            "client_persona": extracted_data["client_persona"], "client_priorities": extracted_data["client_priorities"],
            "client_values": extracted_data["client_values"], "client_tone": extracted_data["client_tone"],
            "input_to_review": actual_document_content,
        }
        
        # ... (feedback chain run and logging as before)
        try:
            if hasattr(self.feedback_chain, 'invoke'):
                res_dict = self.feedback_chain.invoke(feedback_input_payload)
                feedback_response = res_dict.get('text', str(res_dict))
            else: feedback_response = self.feedback_chain.run(feedback_input_payload)
            logger.info("ClientRepAgent: Feedback generated.")
            return feedback_response
        except Exception as feedback_error:
            logger.error(f"ClientRepAgent: Error in feedback chain: {feedback_error}")
            return f"Error: Could not generate client feedback. Details: {feedback_error}"