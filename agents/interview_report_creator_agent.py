# import os, json, re, logging, fitz
# from typing import List, Optional
# from langchain_community.chat_models import ChatOpenAI 
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()
# logger = logging.getLogger(__name__)

# def is_input_valid(text: str, min_length: int = 10, short_text_threshold: int = 20, min_unique_chars_for_short_text: int = 3, allow_short_natural_lang: bool = False) -> bool:
#     if not text or not text.strip(): return False
#     stripped_text = text.strip()
#     if allow_short_natural_lang and len(stripped_text) < min_length:
#         return bool(len(stripped_text) >=1 and any(c.isalnum() for c in stripped_text)) # Allow very short commands
#     if len(stripped_text) < min_length: return False
#     if len(stripped_text) <= short_text_threshold:
#         alnum = ''.join(filter(str.isalnum, stripped_text))
#         if not alnum and len(stripped_text) > 0: return False
#         elif alnum and len(set(alnum.lower())) < min_unique_chars_for_short_text: return False
#     return True

# class InterviewReportCreatorAgent:
#     def __init__(self, verbose=False):
#         self.role = "an expert talent assessment consultant" # Simplified for brevity
#         self.goal = "to generate concise, insightful, structured interview reports."
#         self.verbose = verbose
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key: logger.error("Interview Agent: API key missing."); raise ValueError("OPENAI_API_KEY not found.")
        
#         try:
#             self.llm = ChatOpenAI(
#                 openai_api_key=self.openai_api_key, temperature=0.6, 
#                 max_tokens=3500, model_name="gpt-4" 
#             )
#         except Exception as e: logger.exception(f"Interview Agent: LLM init error: {e}"); raise
#         self._create_prompt()
#         self._create_chain()
#         if self.verbose: logger.info("InterviewReportCreatorAgent initialized.")

#     def _create_prompt(self):
#         template = """You are {role}. Your primary goal is {goal}.
# You are creating a professional interview report. The input might include detailed sections or a very brief instruction like "report for candidate John Doe for AI Engineer role", with most details expected in attached files (CV, job spec, transcript, consultant notes).
# If specific sections like "Job Specification" or "Consultant's Assessment" are marked "Not provided" or are very brief in the input below, you MUST rely heavily on the content of attached files (identified by keywords in their names like 'cv', 'job_spec', 'transcript', 'assessment') to gather the necessary information.

# **INTERVIEW REPORT**
# **Candidate:** [Infer Name from CV/Input, else "The Candidate"]
# **Role:** [Infer Role from Job Spec/Input, else "The Discussed Role"]

# **1. Candidate Overview & Background (Approx. 300-400 words):**
#    - Opening statement. Summary of career, achievements (from CV, interview). Education.
# **2. Alignment with Role & Career Motivation (Approx. 300-400 words):**
#    - Skills/experience vs. role requirements (from Job Spec, Scorecard). Motivation. Examples.
# **3. Consultant's Assessment & Recommendation (Approx. 300-400 words):**
#    - Reflect "Consultant's Assessment" input heavily. Overall impression. Strengths. Weaknesses/Development areas. Clear recommendation.

# **Key Information Provided (might be brief or point to files):**
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

# Synthesize all available information. If critical info (e.g., CV content for Candidate Overview) is missing from both direct input and inferable file content, state that the section cannot be fully completed. Maintain a professional, objective, insightful tone.
# """
#         self.prompt_template = PromptTemplate(
#             input_variables=[
#                 "role", "goal", "job_spec", "scorecard", "candidate_cv",
#                 "interview_transcript", "consultant_assessment",
#             ], template=template,
#         )

#     def _create_chain(self):
#         self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=self.verbose)

#     def _read_file(self, path: str, ext:str) -> str: # Combined file reader
#         content = ""
#         try:
#             if ext == ".pdf":
#                 with fitz.open(path) as doc: content = "".join(p.get_text("text") for p in doc)
#             elif ext == ".txt":
#                 with open(path, "r", encoding="utf-8") as f: content = f.read()
#             elif ext == ".json":
#                 with open(path, "r", encoding="utf-8") as f: data = json.load(f)
#                 # Basic JSON to text, can be improved
#                 if isinstance(data, list): content = "\n".join(map(str, data))
#                 else: content = json.dumps(data, indent=2)
#             return content.strip()
#         except Exception as e: logger.error(f"Interview Agent: Error reading {path}: {e}"); return ""

#     def _extract_text_from_file(self, file_path: str) -> str:
#         if not file_path or not os.path.exists(file_path): return ""
#         ext = os.path.splitext(file_path)[-1].lower()
#         content = self._read_file(file_path, ext)
#         if content: return f"\n\n--- Content from File: {os.path.basename(file_path)} ---\n{content}"
#         return ""

#     def _parse_input_text_sections(self, text: str) -> dict: 
#         # ... (parsing logic with regex remains useful for structured text)
#         sections = {"job_spec": "", "scorecard": "", "candidate_cv": "", "interview_transcript": "", "consultant_assessment": ""}
#         if not text: return sections
#         patterns = { # Simplified patterns for robustness
#             "job_spec": r"---JOB SPEC(?:IFICATION)?---\s*(.*?)(?=---[A-Z\s]+---|$)",
#             "scorecard": r"---(?:ROLE )?SCORECARD---\s*(.*?)(?=---[A-Z\s]+---|$)",
#             "candidate_cv": r"---CANDIDATE CV|RESUME---\s*(.*?)(?=---[A-Z\s]+---|$)",
#             "interview_transcript": r"---INTERVIEW TRANSCRIPT|NOTES---\s*(.*?)(?=---[A-Z\s]+---|$)",
#             "consultant_assessment": r"---CONSULTANT(?:'S)? ASSESSMENT|NOTES---\s*(.*?)(?=---[A-Z\s]+---|$)",
#         }
#         remaining_text = text
#         for key, pattern in patterns.items():
#             match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
#             if match:
#                 sections[key] = match.group(1).strip()
#                 remaining_text = remaining_text.replace(match.group(0), "", 1)
#         if not sections["consultant_assessment"] and remaining_text.strip(): # Fallback for untagged
#             sections["consultant_assessment"] = remaining_text.strip()
#         elif not any(sections.values()) and text.strip(): # No tags at all
#              sections["consultant_assessment"] = text.strip() # Assume all of it is assessment/general instruction
#         return sections

#     def run(self, input_text: str = "", attachment_paths: Optional[List[str]] = None) -> str:
#         final_input_text = input_text.strip() if input_text else ""
        
#         # Allow very short input_text like "report for John Doe, AI dev"
#         # min_length 5. It's okay if it's not super meaningful on its own IF files are provided.
#         is_input_text_weak = final_input_text and not is_input_valid(final_input_text, min_length=5, allow_short_natural_lang=True)

#         if is_input_text_weak and not attachment_paths:
#             logger.error("Interview Agent: Input text weak/empty and no attachments.")
#             return "Error: Insufficient input. Provide meaningful text or attachment files."

#         # If input_text is weak but files exist, LLM is prompted to rely on files.
#         # The parsed sections from a weak input_text might be empty or trivial.
        
#         content_map = {"job_spec": "", "scorecard": "", "candidate_cv": "", "interview_transcript": "", "consultant_assessment": ""}
#         if final_input_text: # Parse even if weak, might contain candidate/role name
#             parsed = self._parse_input_text_sections(final_input_text)
#             for k,v in parsed.items():
#                 if v and v.strip(): content_map[k] = v
        
#         # Accumulate file content for validation and LLM
#         all_files_text_for_llm = ""
#         if attachment_paths:
#             temp_files_content_list = []
#             for path in attachment_paths:
#                 file_text = self._extract_text_from_file(path) # Includes header
#                 if file_text and file_text.strip():
#                     temp_files_content_list.append(file_text)
#                     # Assign to content_map based on filename keywords (as before)
#                     fname_lower = os.path.basename(path).lower()
#                     # ... (keyword logic from previous version to populate content_map fields from files)
#                     if any(k in fname_lower for k in ["job spec", "job_spec", "jd"]): content_map["job_spec"] += f"\n{file_text}"
#                     elif any(k in fname_lower for k in ["scorecard", "criteria"]): content_map["scorecard"] += f"\n{file_text}"
#                     elif any(k in fname_lower for k in ["cv", "resume"]): content_map["candidate_cv"] += f"\n{file_text}"
#                     elif any(k in fname_lower for k in ["consultant", "assessment", "evaluation"]): content_map["consultant_assessment"] += f"\n{file_text}"
#                     elif any(k in fname_lower for k in ["interview", "transcript", "notes"]): content_map["interview_transcript"] += f"\n{file_text}"
#                     elif "candidate" in fname_lower and not content_map["candidate_cv"].strip(): content_map["candidate_cv"] += f"\n{file_text}" # Fallback for candidate file
#                     else: content_map["interview_transcript"] += f"\n{file_text}" # General fallback for other files

#             all_files_text_for_llm = "\n".join(temp_files_content_list)

#         # Check if combined essential info is sufficient
#         # The LLM is now guided to look into files if direct inputs are sparse.
#         # We primarily ensure that *something* meaningful is passed.
#         has_meaningful_cv = is_input_valid(content_map["candidate_cv"], min_length=20)
#         has_meaningful_assessment = is_input_valid(content_map["consultant_assessment"], min_length=20)
#         has_meaningful_job_spec = is_input_valid(content_map["job_spec"], min_length=20)

#         # If input_text was the ONLY source and it was weak, AND no files, this is an error.
#         if is_input_text_weak and not attachment_paths: # Already caught, but for clarity
#             return "Error: Input text was weak/non-meaningful and no files were provided."
        
#         # If there are no files and the input_text (even if not "weak" by short_natural_lang) didn't result in meaningful parsed sections
#         if not attachment_paths and not (has_meaningful_assessment or has_meaningful_cv or has_meaningful_job_spec):
#             if not is_input_valid(final_input_text, min_length=30): # If the original full input text was also not substantial
#                 logger.error("Interview Agent: No files and input_text yielded no substantial sections.")
#                 return "Error: Insufficient meaningful content from input text and no files provided to generate report."

#         llm_input_payload = {
#             "role": self.role, "goal": self.goal,
#             "job_spec": content_map["job_spec"].strip() or "Not provided. Expected in files if relevant.",
#             "scorecard": content_map["scorecard"].strip() or "Not provided.",
#             "candidate_cv": content_map["candidate_cv"].strip() or "Not provided. Expected in files.",
#             "interview_transcript": content_map["interview_transcript"].strip() or "Not provided. May be in files.",
#             "consultant_assessment": content_map["consultant_assessment"].strip() or "Not explicitly provided. Infer from notes/files or general context if available.",
#         }
#         if self.verbose: logger.info(f"Interview Agent LLM Input: { {k: (v[:100] + '...' if len(v)>100 else v) for k,v in llm_input_payload.items()} }")
        
#         try:
#             if hasattr(self.chain, 'invoke'):
#                 res_dict = self.chain.invoke(llm_input_payload); report = res_dict.get('text', str(res_dict))
#             else: report = self.chain.run(llm_input_payload)
#             if not report or not report.strip(): return "Error: Generated report was empty."
#             return report.strip()
#         except Exception as e:
#             logger.exception(f"Interview Agent: LLM chain error: {e}")
#             return f"Error: Could not generate interview report. Details: {e}"

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
        return bool(len(stripped_text) >= 1 and any(c.isalnum() for c in stripped_text))
    if len(stripped_text) < min_length: return False
    if len(stripped_text) <= short_text_threshold:
        alnum = ''.join(filter(str.isalnum, stripped_text))
        if not alnum and len(stripped_text) > 0: return False
        elif alnum and len(set(alnum.lower())) < min_unique_chars_for_short_text: return False
    return True

class InterviewReportCreatorAgent:
    def __init__(self, verbose=False):
        self.role = "an expert talent assessment consultant"
        self.goal = "to generate concise, insightful, structured interview reports."
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("Interview Agent: API key missing.")
            raise ValueError("OPENAI_API_KEY not found.")
        
        try:
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                temperature=0.6,
                max_tokens=3500,
                model_name="gpt-4"
            )
        except Exception as e:
            logger.exception(f"Interview Agent: LLM init error: {e}")
            raise

        self._create_prompt()
        self._create_chain()
        if self.verbose:
            logger.info("InterviewReportCreatorAgent initialized.")

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
            ],
            template=template,
        )

    def _create_chain(self):
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=self.verbose)

    def _read_file(self, path: str, ext: str) -> str:
        try:
            if ext == ".pdf":
                with fitz.open(path) as doc:
                    return "".join(p.get_text("text") for p in doc)
            elif ext == ".txt":
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            elif ext == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return json.dumps(data, indent=2) if isinstance(data, dict) else "\n".join(map(str, data))
        except Exception as e:
            logger.error(f"Interview Agent: Error reading {path}: {e}")
        return ""

    def _extract_text_from_file(self, file_path: str) -> str:
        if not file_path or not os.path.exists(file_path): return ""
        ext = os.path.splitext(file_path)[-1].lower()
        content = self._read_file(file_path, ext)
        return f"\n\n--- Content from File: {os.path.basename(file_path)} ---\n{content}" if content else ""

    def _parse_input_text_sections(self, text: str) -> dict:
        sections = {"job_spec": "", "scorecard": "", "candidate_cv": "", "interview_transcript": "", "consultant_assessment": ""}
        if not text: return sections
        patterns = {
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
        if not sections["consultant_assessment"] and remaining_text.strip():
            sections["consultant_assessment"] = remaining_text.strip()
        elif not any(sections.values()) and text.strip():
            sections["consultant_assessment"] = text.strip()
        return sections

    def run(self, input_text: str = "", attachment_paths: Optional[List[str]] = None) -> str:
        final_input_text = input_text.strip() if input_text else ""
        is_input_text_weak = final_input_text and not is_input_valid(final_input_text, min_length=5, allow_short_natural_lang=True)

        if is_input_text_weak and not attachment_paths:
            logger.error("Interview Agent: Input text weak/empty and no attachments.")
            return "Error: Insufficient input. Provide meaningful text or attachment files."

        content_map = {"job_spec": "", "scorecard": "", "candidate_cv": "", "interview_transcript": "", "consultant_assessment": ""}
        if final_input_text:
            parsed = self._parse_input_text_sections(final_input_text)
            for k, v in parsed.items():
                if v.strip(): content_map[k] = v

        if attachment_paths:
            for path in attachment_paths:
                file_text = self._extract_text_from_file(path)
                if not file_text.strip(): continue
                fname_lower = os.path.basename(path).lower()
                if any(k in fname_lower for k in ["job spec", "job_spec", "jd"]): content_map["job_spec"] += f"\n{file_text}"
                elif any(k in fname_lower for k in ["scorecard", "criteria"]): content_map["scorecard"] += f"\n{file_text}"
                elif any(k in fname_lower for k in ["cv", "resume"]): content_map["candidate_cv"] += f"\n{file_text}"
                elif any(k in fname_lower for k in ["consultant", "assessment", "evaluation"]): content_map["consultant_assessment"] += f"\n{file_text}"
                elif any(k in fname_lower for k in ["interview", "transcript", "notes"]): content_map["interview_transcript"] += f"\n{file_text}"
                elif "candidate" in fname_lower and not content_map["candidate_cv"].strip(): content_map["candidate_cv"] += f"\n{file_text}"
                else: content_map["interview_transcript"] += f"\n{file_text}"

        llm_input_payload = {
            "role": self.role,
            "goal": self.goal,
            "job_spec": content_map["job_spec"].strip() or "Not provided. Expected in files if relevant.",
            "scorecard": content_map["scorecard"].strip() or "Not provided.",
            "candidate_cv": content_map["candidate_cv"].strip() or "Not provided. Expected in files.",
            "interview_transcript": content_map["interview_transcript"].strip() or "Not provided. May be in files.",
            "consultant_assessment": content_map["consultant_assessment"].strip() or "Not explicitly provided. Infer from notes/files or general context if available.",
        }

        try:
            if hasattr(self.chain, 'invoke'):
                res_dict = self.chain.invoke(llm_input_payload)
                report = res_dict.get('text', str(res_dict))
            else:
                report = self.chain.run(llm_input_payload)
            return report.strip() if report else "Error: Generated report was empty."
        except Exception as e:
            logger.exception(f"Interview Agent: LLM chain error: {e}")
            return f"Error: Could not generate interview report. Details: {e}"