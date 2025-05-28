# import os
# from typing import Optional, List
# from dotenv import load_dotenv
# from langchain_community.llms import OpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# import json
# import fitz  # PyMuPDF for PDF reading
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class ClientRepresentativeAgent:
#     def __init__(self, verbose: bool = False):
#         self.verbose = verbose
#         load_dotenv() # Load environment variables
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key:
#             logger.error("OpenAI API key not found in environment variables.")
#             raise ValueError("OPENAI_API_KEY not found.  Set environment variable.") # Prevents agent initialization

#         try:
#             self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7, max_tokens=3000)
#         except Exception as e:
#             logger.exception(f"Error initializing OpenAI LLM: {e}")
#             raise  # Re-raise the exception to prevent agent from working.

#         self._create_extraction_prompt()
#         self._create_feedback_prompt()
#         self._create_chains()

#     def _create_extraction_prompt(self):
#         self.extraction_template = PromptTemplate(
#             input_variables=["input_statement"],
#             template="""
#     From the following statement, infer the client's characteristics:

#     Statement:
#     {input_statement}

#     Return as JSON:
#     {{
#         "client_persona": "...",
#         "client_priorities": "...",
#         "client_values": "...",
#         "client_tone": "..."
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
#                 "input_to_review",
#             ],
#             template="""
#     You are a Client Representative in a project. Review the following input as if you were the client.

#     Client Persona: {client_persona}  
#     Client Priorities: {client_priorities}  
#     Client Values: {client_values}  
#     Client Tone and Style: {client_tone}  

#     Input to Review:  
#     {input_to_review}

#     Your job:
#     - Emulate the client's tone
#     - Reflect their priorities and values
#     - Offer constructive feedback
#     - Ask for clarity or express concerns
#     - Be specific and helpful

#     Write a thorough response of approximately **500 words** that reflects how the client would engage with this input.
#     Include:
#     - Clear feedback on alignment with goals and tone
#     - Specific strengths and weaknesses
#     - Clarifying questions if needed
#     - Suggestions for improvement or refinement

#     Client Representative Feedback:
#     """
#         )

#     def _create_chains(self):
#         self.extraction_chain = LLMChain(llm=self.llm, prompt=self.extraction_template)
#         self.feedback_chain = LLMChain(llm=self.llm, prompt=self.feedback_template)

#     def run(self, input_statement: str, transcript_file_paths: Optional[list[str]] = None) -> str:
#         logger.info(f"Agent running with input: {input_statement[:100]}...") # Log initial input
#         combined_input = input_statement or ""

#         # If files are provided, extract and append their content
#         if transcript_file_paths:
#             logger.info(f"Processing transcript files: {transcript_file_paths}")  # Log file paths
#             for file_path in transcript_file_paths:
#                 try:
#                     ext = os.path.splitext(file_path)[-1].lower()
#                     if ext == ".pdf":
#                         # Extract text from PDF
#                         try:
#                             with fitz.open(file_path) as doc:
#                                 pdf_text = "".join(page.get_text() for page in doc)
#                             combined_input += "\n" + pdf_text
#                             logger.info(f"Successfully extracted text from PDF: {file_path}")
#                         except Exception as pdf_e:
#                             logger.error(f"Failed to extract text from PDF {file_path}: {pdf_e}")
#                             if self.verbose:
#                                 print(f"Failed to read PDF file {file_path}: {pdf_e}")  #Keep verbose output.

#                     elif ext == ".txt":
#                         try:
#                             with open(file_path, "r", encoding="utf-8") as f:
#                                 txt_text = f.read()
#                             combined_input += "\n" + txt_text
#                             logger.info(f"Successfully read text from TXT: {file_path}")
#                         except Exception as txt_e:
#                             logger.error(f"Failed to read text from TXT {file_path}: {txt_e}")
#                             if self.verbose:
#                                 print(f"Failed to read TXT file {file_path}: {txt_e}")

#                     else:
#                         logger.warning(f"Skipping unsupported file type: {file_path}")
#                         if self.verbose:
#                             print(f"Skipping unsupported file type: {file_path}")
#                 except Exception as e:
#                     logger.error(f"General file processing error for {file_path}: {e}") #Catch file reading exceptions.
#                     if self.verbose:
#                         print(f"Failed to read file {file_path}: {e}")

#         logger.info(f"Combined input after file processing: {combined_input[:200]}...")  # Log combined input
#         # Step 1: Extract client persona, tone, values, priorities
#         try:
#             extracted_json_str = self.extraction_chain.run({"input_statement": combined_input})
#             logger.info(f"Extracted JSON string: {extracted_json_str}") #Log output
#             if self.verbose:
#                 print("Extracted Traits:\n", extracted_json_str)
#         except Exception as extraction_error:
#             logger.error(f"Error during extraction chain run: {extraction_error}")
#             raise ValueError(f"Error during extraction: {extraction_error}")

#         try:
#             extracted_data = json.loads(extracted_json_str)
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to parse extracted client data (JSONDecodeError): {e}, raw string: {extracted_json_str}")
#             raise ValueError(f"Failed to parse extracted client data: {e}. Raw string: {extracted_json_str}")
#         except Exception as e:
#             logger.error(f"Failed to parse extracted client data (General Exception): {e}")
#             raise ValueError(f"Failed to parse extracted client data: {e}")

#         # Step 2: Generate feedback using inferred traits
#         feedback_input = {
#             **extracted_data,
#             "input_to_review": combined_input.strip()
#         }

#         try:
#             feedback_response = self.feedback_chain.run(feedback_input)
#             logger.info("Feedback generated successfully.")
#             return feedback_response
#         except Exception as feedback_error:
#             logger.error(f"Error during feedback chain run: {feedback_error}")
#             raise ValueError(f"Error during feedback generation: {feedback_error}")

import os
from typing import Optional, List
from dotenv import load_dotenv
from langchain_community.llms import OpenAI # Keep if you prefer, or switch to ChatOpenAI
from langchain_community.chat_models import ChatOpenAI # Recommended for newer features/models
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import fitz  # PyMuPDF for PDF reading
import logging

# Configure logging (if not already configured at app level, this is fine)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get logger for this module

class ClientRepresentativeAgent:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        load_dotenv() # Load environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("ClientRepresentativeAgent: OpenAI API key not found in environment variables.")
            raise ValueError("OPENAI_API_KEY not found. Set environment variable.")

        try:
            # Using ChatOpenAI is generally recommended over the base OpenAI LLM
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                temperature=0.7,
                max_tokens=3000, # Max output tokens
                model_name="gpt-3.5-turbo" # Or gpt-4 if preferred/available
            )
            # If you must use the older OpenAI class:
            # self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7, max_tokens=3000)
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
    From the following statement (which may include context, documents, or explicit descriptions), infer the client's key characteristics.
    Focus on identifying their professional persona, main priorities, core values, and typical communication tone.

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
                "client_persona",
                "client_priorities",
                "client_values",
                "client_tone",
                "input_to_review", # This will be the document/text the "client" is reviewing
            ],
            template="""
    You are acting as a Client Representative. Your personality and responses should strictly adhere to the defined client characteristics.

    Your Client Profile:
    - Client Persona: {client_persona}
    - Client Priorities: {client_priorities}
    - Client Values: {client_values}
    - Client Tone and Communication Style: {client_tone}

    You have been provided with the following input to review:
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
        """Reads content from PDF, TXT, or JSON files."""
        try:
            ext = os.path.splitext(file_path)[-1].lower()
            content = f"\n\n--- Content from file: {os.path.basename(file_path)} ---\n"
            if ext == ".pdf":
                with fitz.open(file_path) as doc:
                    text = "".join(page.get_text() for page in doc)
                content += text
                logger.info(f"ClientRepAgent: Successfully extracted text from PDF: {file_path}")
            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                content += text
                logger.info(f"ClientRepAgent: Successfully read text from TXT: {file_path}")
            elif ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                # Convert JSON to a readable string format for the LLM
                content += json.dumps(json_data, indent=2)
                logger.info(f"ClientRepAgent: Successfully read and formatted JSON from: {file_path}")
            else:
                logger.warning(f"ClientRepAgent: Skipping unsupported file type: {file_path}")
                return None # Indicate unsupported type
            return content
        except FileNotFoundError:
            logger.error(f"ClientRepAgent: File not found: {file_path}")
        except Exception as e:
            logger.error(f"ClientRepAgent: Failed to read or process file {file_path}: {e}")
        return "" # Return empty string on error to avoid None issues later, or None if strict handling preferred

    def run(self, input_statement: str, transcript_file_paths: Optional[List[str]] = None) -> str:
        if self.verbose:
            logger.info(f"ClientRepAgent: Running with input_statement (preview): {input_statement[:100]}...")
            logger.info(f"ClientRepAgent: Processing {len(transcript_file_paths) if transcript_file_paths else 0} transcript files.")

        # The `input_statement` is the primary text to analyze for client characteristics OR the document to review.
        # The prompt structure implies that `input_statement` for the extraction_chain should contain client info.
        # And `input_to_review` for feedback_chain is the document.
        # In the orchestrator, we decided `input_statement` for feedback generation includes both persona and reviewable doc.
        # Let's assume `input_statement` here is the text that *describes* the client or from which characteristics are inferred.
        # And `transcript_file_paths` provide additional context for this inference.

        # For extraction phase, combine input_statement and file contents.
        # The `input_statement` passed by orchestrator is `feedback_input_statement`
        # which has `---CLIENT PERSONA GUIDANCE---` and `---DOCUMENT TO REVIEW---`
        # The extraction prompt needs to work on the "CLIENT PERSONA GUIDANCE" part.

        # Let's refine: the `input_statement` passed to this agent's `run` method will be the one from the orchestrator.
        # This statement ALREADY contains:
        # 1. Guidance on client persona (from client_prompt, which itself was derived from market_report etc.)
        # 2. The document to be reviewed (e.g., the interview_report).
        # So, `input_statement` is the "document to review" and also contains info to infer persona.
        
        client_characteristics_source = input_statement # Primary source for characteristics
        document_to_review_by_client = input_statement # The same input is reviewed

        if transcript_file_paths:
            for file_path in transcript_file_paths:
                file_content = self._read_file_content(file_path)
                if file_content:
                    # Append file content to the source for characteristics extraction
                    client_characteristics_source += f"\n\n--- Additional Context from File: {os.path.basename(file_path)} ---\n{file_content}"


        logger.info(f"ClientRepAgent: Combined input for persona extraction (preview): {client_characteristics_source[:200]}...")

        # Step 1: Extract client persona, tone, values, priorities
        try:
            # The extraction chain is run on the combined text.
            # The prompt guides it to find persona, priorities, values, tone.
            raw_extracted_json_str = self.extraction_chain.run({"input_statement": client_characteristics_source})
            
            # LLMs sometimes add markdown or other text around JSON. Try to clean it.
            match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_extracted_json_str, re.IGNORECASE)
            if match:
                extracted_json_str = match.group(1)
            else:
                # If no markdown, assume it might be just the JSON or JSON with leading/trailing whitespace
                extracted_json_str = raw_extracted_json_str.strip()

            logger.info(f"ClientRepAgent: Raw extracted JSON string from LLM: {raw_extracted_json_str}")
            logger.info(f"ClientRepAgent: Cleaned extracted JSON string for parsing: {extracted_json_str}")

            if self.verbose:
                print("ClientRepAgent Extracted Traits (Raw):\n", raw_extracted_json_str)
        except Exception as extraction_error:
            logger.error(f"ClientRepAgent: Error during extraction chain run: {extraction_error}")
            return f"Error: Could not extract client characteristics. Details: {extraction_error}"

        try:
            extracted_data = json.loads(extracted_json_str)
            # Validate essential keys - though the prompt asks for them, LLM might miss
            expected_keys = ["client_persona", "client_priorities", "client_values", "client_tone"]
            for key in expected_keys:
                if key not in extracted_data:
                    logger.warning(f"ClientRepAgent: Extracted data missing key '{key}'. Using default.")
                    extracted_data[key] = f"Not specified or inferable (default for {key})"

        except json.JSONDecodeError as e:
            logger.error(f"ClientRepAgent: Failed to parse extracted client data (JSONDecodeError): {e}. Raw string: '{extracted_json_str}'")
            # Fallback: try to use the raw string if it looks somewhat like a description, or provide a generic error
            # This is risky as it might pass a non-JSON string to the next chain. Better to error out.
            return f"Error: Failed to parse client characteristics from LLM output. JSON was malformed. Raw output: {extracted_json_str}"
        except Exception as e: # Catch any other parsing errors
            logger.error(f"ClientRepAgent: Unexpected error parsing extracted client data: {e}")
            return f"Error: Unexpected issue processing client characteristics. Details: {e}"

        # Step 2: Generate feedback using inferred traits
        feedback_input_payload = {
            "client_persona": extracted_data.get("client_persona", "Default Persona: Professional and Objective"),
            "client_priorities": extracted_data.get("client_priorities", "Default Priorities: Clarity, Accuracy, Relevance"),
            "client_values": extracted_data.get("client_values", "Default Values: Professionalism, Usefulness"),
            "client_tone": extracted_data.get("client_tone", "Default Tone: Constructive and Polite"),
            "input_to_review": document_to_review_by_client.strip(), # The original input statement is what's reviewed
        }
        
        if self.verbose:
            logger.info(f"ClientRepAgent: Payload for feedback generation: {feedback_input_payload}")

        try:
            feedback_response = self.feedback_chain.run(feedback_input_payload)
            # If using ChatOpenAI and the chain returns a dict:
            # feedback_response_dict = self.feedback_chain.invoke(feedback_input_payload)
            # feedback_response = feedback_response_dict.get('text', "Error: No feedback text generated.")

            logger.info("ClientRepAgent: Feedback generated successfully.")
            return feedback_response
        except Exception as feedback_error:
            logger.error(f"ClientRepAgent: Error during feedback chain run: {feedback_error}")
            return f"Error: Could not generate client feedback. Details: {feedback_error}"