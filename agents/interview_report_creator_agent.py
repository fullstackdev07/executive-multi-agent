# import os
# import json
# import re
# from typing import List, Optional
# from langchain_community.llms import OpenAI # Keep or switch
# from langchain_community.chat_models import ChatOpenAI # Recommended
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import fitz # PyMuPDF for PDF reading
# import logging

# load_dotenv()
# logger = logging.getLogger(__name__)

# class InterviewReportCreatorAgent:
#     def __init__(self, verbose=False):
#         self.role = "an expert talent assessment consultant specializing in executive search"
#         self.goal = (
#             "to generate concise (~300-400 words per main section), insightful, and structured interview reports for client presentations. "
#             "The report must clearly articulate the candidate's suitability for the role, backed by evidence from their CV, interview, and any consultant assessments."
#         )
#         self.verbose = verbose
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key:
#             logger.error("InterviewReportCreatorAgent: OPENAI_API_KEY not found.")
#             raise ValueError("OPENAI_API_KEY not found.")
        
#         try:
#             self.llm = ChatOpenAI(
#                 openai_api_key=self.openai_api_key,
#                 temperature=0.6, # Slightly lower for more factual reporting
#                 max_tokens=3500, # Max output tokens
#                 model_name="gpt-4" # Or gpt-3.5-turbo-16k for longer context if gpt-4 is too expensive
#             )
#             # self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7, max_tokens=3500)
#         except Exception as e:
#             logger.exception(f"InterviewReportCreatorAgent: Error initializing OpenAI LLM: {e}")
#             raise

#         self._create_prompt()
#         self._create_chain()
#         if self.verbose:
#             logger.info("InterviewReportCreatorAgent initialized.")

#     def _create_prompt(self):
#         template = """You are {role}. Your primary goal is {goal}.

# You are tasked with creating a professional interview report based on the provided information.
# The report should follow this specific structure, with each main section being approximately 300-400 words:

# **INTERVIEW REPORT**

# **Candidate:** [Name, if inferable, otherwise "The Candidate"]
# **Role:** [Role Title, if inferable, otherwise "The Discussed Role"]
# **Date of Report:** [Today's Date - you don't need to fill this, it's for structure]

# **1. Candidate Overview & Background (Approx. 300-400 words):**
#    - Start with a concise, impactful opening statement about the candidate (e.g., "XXX is a highly accomplished and articulate professional...").
#    - Provide a summary of their overall career trajectory, key experiences, and significant achievements as detailed in their CV and discussed in the interview.
#    - Mention educational background if relevant and prominent.

# **2. Alignment with Role & Career Motivation (Approx. 300-400 words):**
#    - Analyze how the candidate's skills, experience, and past accomplishments align with the requirements of the target role (as per Job Spec and Scorecard, if available).
#    - Discuss their stated motivations for considering this new opportunity and how it fits into their career development aspirations.
#    - Highlight specific examples from their background that demonstrate suitability for key responsibilities.

# **3. Consultant's Assessment & Recommendation (Approx. 300-400 words):**
#    - This section should heavily reflect the "Consultant's Assessment" input.
#    - Summarize the interviewing consultant's overall impression of the candidate.
#    - Detail perceived strengths relevant to the role.
#    - Candidly discuss any potential weaknesses, areas for development, or aspects that might require further exploration or validation.
#    - Conclude with a clear recommendation (e.g., "Highly Recommended for Next Steps," "Recommended with Considerations," "Not Recommended at This Time") and a brief justification.

# **Key Information Provided:**
# --- Job Specification ---
# {job_spec}
# --- End Job Specification ---

# --- Role Scorecard (if available) ---
# {scorecard}
# --- End Role Scorecard ---

# --- Candidate's CV / Resume ---
# {candidate_cv}
# --- End Candidate's CV / Resume ---

# --- Interview Transcript / Notes ---
# {interview_transcript}
# --- End Interview Transcript / Notes ---

# --- Consultant's Raw Assessment / Notes (Prioritize this for Section 3) ---
# {consultant_assessment}
# --- End Consultant's Raw Assessment ---

# Synthesize all available information to produce a coherent, well-written report.
# If some information (e.g., scorecard) is "Not provided," acknowledge that and focus on available data.
# Maintain a professional, objective, yet insightful tone.
# """
#         self.prompt_template = PromptTemplate(
#             input_variables=[
#                 "role",
#                 "goal",
#                 "job_spec",
#                 "scorecard",
#                 "candidate_cv",
#                 "interview_transcript",
#                 "consultant_assessment",
#             ],
#             template=template,
#         )

#     def _create_chain(self):
#         self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=self.verbose)

#     def _read_pdf(self, path: str) -> str:
#         text = ""
#         try:
#             with fitz.open(path) as doc:
#                 for page in doc:
#                     text += page.get_text()
#             logger.info(f"InterviewReportAgent: Successfully extracted text from PDF: {path}")
#         except Exception as e:
#             logger.error(f"InterviewReportAgent: Failed to extract text from PDF {path}: {e}")
#             return ""
#         return text.strip()

#     def _read_txt(self, path: str) -> str:
#         try:
#             with open(path, "r", encoding="utf-8") as f:
#                 text = f.read()
#             logger.info(f"InterviewReportAgent: Successfully read text from TXT: {path}")
#             return text.strip()
#         except Exception as e:
#             logger.error(f"InterviewReportAgent: Failed to read TXT file {path}: {e}")
#             return ""

#     def _read_json_as_text(self, path: str) -> str: # Renamed for clarity
#         try:
#             with open(path, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#             # Convert JSON to a readable string. If it's a transcript, try to format it nicely.
#             # Heuristic for transcripts (often a list of dicts with 'speaker' and 'text')
#             if isinstance(data, list) and data and isinstance(data[0], dict):
#                 lines = []
#                 for item in data:
#                     speaker = item.get('speaker', item.get('Speaker', 'Unknown'))
#                     line = item.get('text', item.get('line', item.get('transcript', '')))
#                     if line: # Only add if there's text
#                          lines.append(f"{speaker}: {line}")
#                 if lines:
#                     formatted_text = "\n".join(lines)
#                     logger.info(f"InterviewReportAgent: Formatted JSON transcript from: {path}")
#                     return formatted_text.strip()
            
#             # Fallback for other JSON structures: pretty print
#             text_content = json.dumps(data, indent=2)
#             logger.info(f"InterviewReportAgent: Read and formatted generic JSON from: {path}")
#             return text_content.strip()
#         except Exception as e:
#             logger.error(f"InterviewReportAgent: Failed to read or process JSON file {path}: {e}")
#             return ""

#     def _extract_text_from_file(self, file_path: str) -> str:
#         if not file_path or not os.path.exists(file_path):
#             logger.warning(f"InterviewReportAgent: File path is invalid or file does not exist: {file_path}")
#             return ""
            
#         ext = os.path.splitext(file_path)[-1].lower()
#         content = ""
#         try:
#             if ext == ".pdf":
#                 content = self._read_pdf(file_path)
#             elif ext == ".txt":
#                 content = self._read_txt(file_path)
#             elif ext == ".json":
#                 content = self._read_json_as_text(file_path) # Use the renamed method
#             else:
#                 if self.verbose:
#                     logger.warning(f"InterviewReportAgent: Unsupported file type {ext} for {file_path}. Ignored.")
#                 return "" # Explicitly return empty for unsupported
            
#             # Add a header to the content indicating its source file
#             if content:
#                 return f"\n\n--- Content from File: {os.path.basename(file_path)} ---\n{content}"
#             return "" # If file processing yielded no content
            
#         except Exception as e: # Catch-all for unexpected errors during file processing
#             if self.verbose:
#                 logger.error(f"InterviewReportAgent: Error processing file {file_path}: {e}")
#             return ""

#     def _parse_input_text_sections(self, text: str) -> dict: # Renamed for clarity
#         """
#         Parses structured sections from input text if the user included delimiters.
#         Delimiters are case-insensitive.
#         """
#         sections = {
#             "job_spec": "",
#             "scorecard": "",
#             "candidate_cv": "",
#             "interview_transcript": "",
#             "consultant_assessment": "",
#         }
#         if not text: # No text to parse
#             return sections

#         # Define patterns for each section. Using re.IGNORECASE and re.DOTALL
#         # (?s) is equivalent to re.DOTALL for inline.
#         # Using lookaheads to ensure non-greedy matching up to the next delimiter or end of string.
#         patterns = {
#             "job_spec": r"---JOB SPEC---\s*(.*?)(?=---[A-Z\s]+---|$)",
#             "scorecard": r"---SCORECARD---\s*(.*?)(?=---[A-Z\s]+---|$)",
#             "candidate_cv": r"---CANDIDATE CV---\s*(.*?)(?=---[A-Z\s]+---|$)",
#             "interview_transcript": r"---INTERVIEW TRANSCRIPT---\s*(.*?)(?=---[A-Z\s]+---|$)",
#             "consultant_assessment": r"---CONSULTANT ASSESSMENT---\s*(.*?)(?=---[A-Z\s]+---|$)",
#         }

#         remaining_text = text # For default assignment if no sections match
#         found_any_section = False

#         for key, pattern in patterns.items():
#             match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
#             if match:
#                 sections[key] = match.group(1).strip()
#                 found_any_section = True
#                 # Remove matched part from remaining_text (simplistic, might need refinement for overlapping)
#                 # This is mainly for the fallback if consultant_assessment is not explicitly tagged
#                 remaining_text = remaining_text.replace(match.group(0), "", 1) 


#         # If no specific sections were parsed but input_text was provided,
#         # and consultant_assessment is still empty, assume the whole input_text is the consultant_assessment.
#         if not found_any_section and text.strip() and not sections["consultant_assessment"]:
#             sections["consultant_assessment"] = text.strip()
#             if self.verbose:
#                 logger.info("InterviewReportAgent: No delimiters found in input_text, treating all as consultant_assessment.")
#         elif not sections["consultant_assessment"] and remaining_text.strip():
#             # If some sections were parsed, but consultant_assessment wasn't,
#             # and there's other text not captured by delimiters, assign it to consultant_assessment.
#             sections["consultant_assessment"] = remaining_text.strip()
#             if self.verbose:
#                 logger.info("InterviewReportAgent: Assigning remaining untagged input_text to consultant_assessment.")


#         if self.verbose:
#             for key, value in sections.items():
#                 logger.info(f"InterviewReportAgent: Parsed from input_text for '{key}': {value[:100]}...")
#         return sections

#     def run(self, input_text: str = "", attachment_paths: Optional[List[str]] = None) -> str:
#         if self.verbose:
#             logger.info(f"InterviewReportAgent: Running with input_text (preview): {input_text[:100]}...")
#             logger.info(f"InterviewReportAgent: Processing {len(attachment_paths) if attachment_paths else 0} attachment files.")

#         # Initialize content placeholders
#         content_map = {
#             "job_spec": "",
#             "scorecard": "",
#             "candidate_cv": "",
#             "interview_transcript": "",
#             "consultant_assessment": "",
#         }

#         # 1. Parse structured sections from input_text first
#         if input_text:
#             parsed_sections = self._parse_input_text_sections(input_text)
#             for key, value in parsed_sections.items():
#                 if value: # Only update if parsing yielded content
#                     content_map[key] = value

#         # 2. Process attachment files, appending their content
#         #    Files can override or add to parsed sections based on keywords in filename.
#         if attachment_paths:
#             for path in attachment_paths:
#                 file_content = self._extract_text_from_file(path) # This now includes a header
#                 if not file_content:
#                     continue # Skip if file reading failed or unsupported

#                 fname_lower = os.path.basename(path).lower()
#                 assigned_to_key = False

#                 # Prioritize assignment based on keywords
#                 if "job" in fname_lower or "spec" in fname_lower or "jd" in fname_lower:
#                     content_map["job_spec"] += f"\n{file_content}"
#                     assigned_to_key = True
#                 elif "scorecard" in fname_lower or "criteria" in fname_lower:
#                     content_map["scorecard"] += f"\n{file_content}"
#                     assigned_to_key = True
#                 elif "cv" in fname_lower or "resume" in fname_lower or "candidate" in fname_lower: # Candidate might also be interview notes
#                     content_map["candidate_cv"] += f"\n{file_content}"
#                     assigned_to_key = True
#                 elif "interview" in fname_lower or "transcript" in fname_lower or "notes" in fname_lower:
#                     content_map["interview_transcript"] += f"\n{file_content}"
#                     assigned_to_key = True
#                 elif "assessment" in fname_lower or "consultant" in fname_lower or "evaluation" in fname_lower:
#                     content_map["consultant_assessment"] += f"\n{file_content}"
#                     assigned_to_key = True
                
#                 if not assigned_to_key:
#                     # Fallback: if filename doesn't match specific keywords,
#                     # add its content to a general pool, perhaps 'interview_transcript' or a new 'other_documents' field.
#                     # For now, adding to 'interview_transcript' as a common place for miscellaneous notes.
#                     content_map["interview_transcript"] += f"\n{file_content}"
#                     if self.verbose:
#                         logger.info(f"InterviewReportAgent: File {fname_lower} assigned to 'interview_transcript' by fallback.")
        
#         # Prepare final input for the LLM, ensuring no Nones and providing defaults
#         final_llm_input = {
#             "role": self.role,
#             "goal": self.goal,
#             "job_spec": content_map["job_spec"].strip() or "Not provided.",
#             "scorecard": content_map["scorecard"].strip() or "Not provided.",
#             "candidate_cv": content_map["candidate_cv"].strip() or "Not provided.",
#             "interview_transcript": content_map["interview_transcript"].strip() or "Not provided.",
#             # Consultant assessment is crucial. If completely empty, use a placeholder.
#             "consultant_assessment": content_map["consultant_assessment"].strip() or "No specific consultant assessment provided. Please synthesize from other available information.",
#         }

#         if self.verbose:
#             logger.info("InterviewReportAgent: Final input to LLM:")
#             for k, v_preview in final_llm_input.items():
#                 logger.info(f"--- Key: {k} ---")
#                 logger.info(f"{str(v_preview)[:500]}...\n")
        
#         try:
#             report_result = self.chain.run(final_llm_input)
#             # If using ChatModel and invoke:
#             # report_dict = self.chain.invoke(final_llm_input)
#             # report_result = report_dict.get('text', "Error: Could not generate interview report.")
#             logger.info("InterviewReportAgent: Report generated successfully.")
#             return report_result.strip()
#         except Exception as e:
#             logger.exception("InterviewReportAgent: Error during LLM chain execution for report generation.")
#             return f"Error: Could not generate interview report. Details: {str(e)}"
import os, json, re, logging, fitz
from typing import List, Optional
from langchain_community.chat_models import ChatOpenAI 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def is_input_valid(text: str, min_length: int = 10, short_text_threshold: int = 20, min_unique_chars_for_short_text: int = 3, allow_short_natural_lang: bool = False) -> bool:
    if not text or not text.strip(): return False
    stripped_text = text.strip()
    if allow_short_natural_lang and len(stripped_text) < min_length:
        return bool(len(stripped_text) >=1 and any(c.isalnum() for c in stripped_text)) # Allow very short commands
    if len(stripped_text) < min_length: return False
    if len(stripped_text) <= short_text_threshold:
        alnum = ''.join(filter(str.isalnum, stripped_text))
        if not alnum and len(stripped_text) > 0: return False
        elif alnum and len(set(alnum.lower())) < min_unique_chars_for_short_text: return False
    return True

class InterviewReportCreatorAgent:
    def __init__(self, verbose=False):
        self.role = "an expert talent assessment consultant" # Simplified for brevity
        self.goal = "to generate concise, insightful, structured interview reports."
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key: logger.error("Interview Agent: API key missing."); raise ValueError("OPENAI_API_KEY not found.")
        
        try:
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key, temperature=0.6, 
                max_tokens=3500, model_name="gpt-4" 
            )
        except Exception as e: logger.exception(f"Interview Agent: LLM init error: {e}"); raise
        self._create_prompt()
        self._create_chain()
        if self.verbose: logger.info("InterviewReportCreatorAgent initialized.")

    def _create_prompt(self):
        template = """You are {role}. Your primary goal is {goal}.
You are creating a professional interview report. The input might include detailed sections or a very brief instruction like "report for candidate John Doe for AI Engineer role", with most details expected in attached files (CV, job spec, transcript, consultant notes).
If specific sections like "Job Specification" or "Consultant's Assessment" are marked "Not provided" or are very brief in the input below, you MUST rely heavily on the content of attached files (identified by keywords in their names like 'cv', 'job_spec', 'transcript', 'assessment') to gather the necessary information.

**INTERVIEW REPORT**
**Candidate:** [Infer Name from CV/Input, else "The Candidate"]
**Role:** [Infer Role from Job Spec/Input, else "The Discussed Role"]

**1. Candidate Overview & Background (Approx. 300-400 words):**
   - Opening statement. Summary of career, achievements (from CV, interview). Education.
**2. Alignment with Role & Career Motivation (Approx. 300-400 words):**
   - Skills/experience vs. role requirements (from Job Spec, Scorecard). Motivation. Examples.
**3. Consultant's Assessment & Recommendation (Approx. 300-400 words):**
   - Reflect "Consultant's Assessment" input heavily. Overall impression. Strengths. Weaknesses/Development areas. Clear recommendation.

**Key Information Provided (might be brief or point to files):**
--- Job Specification ---
{job_spec}
--- End Job Specification ---
--- Role Scorecard (if available) ---
{scorecard}
--- End Role Scorecard ---
--- Candidate's CV / Resume ---
{candidate_cv}
--- End Candidate's CV / Resume ---
--- Interview Transcript / Notes ---
{interview_transcript}
--- End Interview Transcript / Notes ---
--- Consultant's Raw Assessment / Notes (Prioritize this for Section 3) ---
{consultant_assessment}
--- End Consultant's Raw Assessment ---

Synthesize all available information. If critical info (e.g., CV content for Candidate Overview) is missing from both direct input and inferable file content, state that the section cannot be fully completed. Maintain a professional, objective, insightful tone.
"""
        self.prompt_template = PromptTemplate(
            input_variables=[
                "role", "goal", "job_spec", "scorecard", "candidate_cv",
                "interview_transcript", "consultant_assessment",
            ], template=template,
        )

    def _create_chain(self):
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=self.verbose)

    def _read_file(self, path: str, ext:str) -> str: # Combined file reader
        content = ""
        try:
            if ext == ".pdf":
                with fitz.open(path) as doc: content = "".join(p.get_text("text") for p in doc)
            elif ext == ".txt":
                with open(path, "r", encoding="utf-8") as f: content = f.read()
            elif ext == ".json":
                with open(path, "r", encoding="utf-8") as f: data = json.load(f)
                # Basic JSON to text, can be improved
                if isinstance(data, list): content = "\n".join(map(str, data))
                else: content = json.dumps(data, indent=2)
            return content.strip()
        except Exception as e: logger.error(f"Interview Agent: Error reading {path}: {e}"); return ""

    def _extract_text_from_file(self, file_path: str) -> str:
        if not file_path or not os.path.exists(file_path): return ""
        ext = os.path.splitext(file_path)[-1].lower()
        content = self._read_file(file_path, ext)
        if content: return f"\n\n--- Content from File: {os.path.basename(file_path)} ---\n{content}"
        return ""

    def _parse_input_text_sections(self, text: str) -> dict: 
        # ... (parsing logic with regex remains useful for structured text)
        sections = {"job_spec": "", "scorecard": "", "candidate_cv": "", "interview_transcript": "", "consultant_assessment": ""}
        if not text: return sections
        patterns = { # Simplified patterns for robustness
            "job_spec": r"---JOB SPEC(?:IFICATION)?---\s*(.*?)(?=---[A-Z\s]+---|$)",
            "scorecard": r"---(?:ROLE )?SCORECARD---\s*(.*?)(?=---[A-Z\s]+---|$)",
            "candidate_cv": r"---CANDIDATE CV|RESUME---\s*(.*?)(?=---[A-Z\s]+---|$)",
            "interview_transcript": r"---INTERVIEW TRANSCRIPT|NOTES---\s*(.*?)(?=---[A-Z\s]+---|$)",
            "consultant_assessment": r"---CONSULTANT(?:'S)? ASSESSMENT|NOTES---\s*(.*?)(?=---[A-Z\s]+---|$)",
        }
        remaining_text = text
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[key] = match.group(1).strip()
                remaining_text = remaining_text.replace(match.group(0), "", 1)
        if not sections["consultant_assessment"] and remaining_text.strip(): # Fallback for untagged
            sections["consultant_assessment"] = remaining_text.strip()
        elif not any(sections.values()) and text.strip(): # No tags at all
             sections["consultant_assessment"] = text.strip() # Assume all of it is assessment/general instruction
        return sections

    def run(self, input_text: str = "", attachment_paths: Optional[List[str]] = None) -> str:
        final_input_text = input_text.strip() if input_text else ""
        
        # Allow very short input_text like "report for John Doe, AI dev"
        # min_length 5. It's okay if it's not super meaningful on its own IF files are provided.
        is_input_text_weak = final_input_text and not is_input_valid(final_input_text, min_length=5, allow_short_natural_lang=True)

        if is_input_text_weak and not attachment_paths:
            logger.error("Interview Agent: Input text weak/empty and no attachments.")
            return "Error: Insufficient input. Provide meaningful text or attachment files."

        # If input_text is weak but files exist, LLM is prompted to rely on files.
        # The parsed sections from a weak input_text might be empty or trivial.
        
        content_map = {"job_spec": "", "scorecard": "", "candidate_cv": "", "interview_transcript": "", "consultant_assessment": ""}
        if final_input_text: # Parse even if weak, might contain candidate/role name
            parsed = self._parse_input_text_sections(final_input_text)
            for k,v in parsed.items():
                if v and v.strip(): content_map[k] = v
        
        # Accumulate file content for validation and LLM
        all_files_text_for_llm = ""
        if attachment_paths:
            temp_files_content_list = []
            for path in attachment_paths:
                file_text = self._extract_text_from_file(path) # Includes header
                if file_text and file_text.strip():
                    temp_files_content_list.append(file_text)
                    # Assign to content_map based on filename keywords (as before)
                    fname_lower = os.path.basename(path).lower()
                    # ... (keyword logic from previous version to populate content_map fields from files)
                    if any(k in fname_lower for k in ["job spec", "job_spec", "jd"]): content_map["job_spec"] += f"\n{file_text}"
                    elif any(k in fname_lower for k in ["scorecard", "criteria"]): content_map["scorecard"] += f"\n{file_text}"
                    elif any(k in fname_lower for k in ["cv", "resume"]): content_map["candidate_cv"] += f"\n{file_text}"
                    elif any(k in fname_lower for k in ["consultant", "assessment", "evaluation"]): content_map["consultant_assessment"] += f"\n{file_text}"
                    elif any(k in fname_lower for k in ["interview", "transcript", "notes"]): content_map["interview_transcript"] += f"\n{file_text}"
                    elif "candidate" in fname_lower and not content_map["candidate_cv"].strip(): content_map["candidate_cv"] += f"\n{file_text}" # Fallback for candidate file
                    else: content_map["interview_transcript"] += f"\n{file_text}" # General fallback for other files

            all_files_text_for_llm = "\n".join(temp_files_content_list)

        # Check if combined essential info is sufficient
        # The LLM is now guided to look into files if direct inputs are sparse.
        # We primarily ensure that *something* meaningful is passed.
        has_meaningful_cv = is_input_valid(content_map["candidate_cv"], min_length=20)
        has_meaningful_assessment = is_input_valid(content_map["consultant_assessment"], min_length=20)
        has_meaningful_job_spec = is_input_valid(content_map["job_spec"], min_length=20)

        # If input_text was the ONLY source and it was weak, AND no files, this is an error.
        if is_input_text_weak and not attachment_paths: # Already caught, but for clarity
            return "Error: Input text was weak/non-meaningful and no files were provided."
        
        # If there are no files and the input_text (even if not "weak" by short_natural_lang) didn't result in meaningful parsed sections
        if not attachment_paths and not (has_meaningful_assessment or has_meaningful_cv or has_meaningful_job_spec):
            if not is_input_valid(final_input_text, min_length=30): # If the original full input text was also not substantial
                logger.error("Interview Agent: No files and input_text yielded no substantial sections.")
                return "Error: Insufficient meaningful content from input text and no files provided to generate report."

        llm_input_payload = {
            "role": self.role, "goal": self.goal,
            "job_spec": content_map["job_spec"].strip() or "Not provided. Expected in files if relevant.",
            "scorecard": content_map["scorecard"].strip() or "Not provided.",
            "candidate_cv": content_map["candidate_cv"].strip() or "Not provided. Expected in files.",
            "interview_transcript": content_map["interview_transcript"].strip() or "Not provided. May be in files.",
            "consultant_assessment": content_map["consultant_assessment"].strip() or "Not explicitly provided. Infer from notes/files or general context if available.",
        }
        if self.verbose: logger.info(f"Interview Agent LLM Input: { {k: (v[:100] + '...' if len(v)>100 else v) for k,v in llm_input_payload.items()} }")
        
        try:
            if hasattr(self.chain, 'invoke'):
                res_dict = self.chain.invoke(llm_input_payload); report = res_dict.get('text', str(res_dict))
            else: report = self.chain.run(llm_input_payload)
            if not report or not report.strip(): return "Error: Generated report was empty."
            return report.strip()
        except Exception as e:
            logger.exception(f"Interview Agent: LLM chain error: {e}")
            return f"Error: Could not generate interview report. Details: {e}"