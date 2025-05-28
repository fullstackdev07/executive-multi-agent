# import os
# import json
# import re
# from langchain_community.llms import OpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader

# load_dotenv()

# class InterviewReportCreatorAgent:
#     def __init__(self, verbose=False):
#         self.role = "an expert interviewer compiling reports for client presentations"
#         self.goal = (
#             "to generate concise (~300 words), structured interview reports for clients, "
#             "emphasizing the consultant's assessment and the candidate's fit for the role."
#         )
#         self.verbose = verbose
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key:
#             raise ValueError("OPENAI_API_KEY not found.")
#         self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7, max_tokens=3500)
#         self._create_prompt()
#         self._create_chain()

#     def _create_prompt(self):
#         template = """You are {role}. Your goal is {goal}.

#         Write a 300-word interview report following this structure:
#         1. A global description of the candidate, starting with a general statement (e.g. "XXX is a very metropolitan and well educated candidate..."), followed by a summary of their experience.
#         2. A paragraph that outlines their career development and how it aligns with the current role.
#         3. A paragraph that summarizes the consultant's recommendation, and highlights any areas that need further exploration.

#         Here is the job description: {job_spec}

#         Here is the role scorecard (if available): {scorecard}

#         Here is the candidate's CV: {candidate_cv}

#         Here is the interview transcript: {interview_transcript}

#         Here is the consultant's assessment (weigh this heavily): {consultant_assessment}
#         """
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
#         self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template,)

#     def _read_pdf(self, path):
#         reader = PdfReader(path)
#         return "\n".join(page.extract_text() or "" for page in reader.pages)

#     def _read_txt(self, path):
#         with open(path, "r", encoding="utf-8") as f:
#             return f.read()

#     def _read_json(self, path):
#         with open(path, "r", encoding="utf-8") as f:
#             return json.load(f)

#     def _extract_text_from_file(self, file_path):
#         ext = os.path.splitext(file_path)[-1].lower()
#         try:
#             if ext == ".pdf":
#                 return self._read_pdf(file_path)
#             elif ext == ".txt":
#                 return self._read_txt(file_path)
#             elif ext == ".json":
#                 data = self._read_json(file_path)
#                 return json.dumps(data)
#             else:
#                 if self.verbose:
#                     print(f"Unsupported file type {ext}. Ignored.")
#                 return ""
#         except Exception as e:
#             if self.verbose:
#                 print(f"Error reading {file_path}: {e}")
#             return ""

#     def run(self, input_text: str = "", attachment_paths: list = None):
#         # Initialize content placeholders
#         job_spec = ""
#         scorecard = ""
#         candidate_cv = ""
#         interview_transcript = ""
#         consultant_assessment = ""

#         # If user provided structured content in text, try to extract using delimiters
#         parsed_sections = self._parse_input_text(input_text)

#         job_spec = parsed_sections["job_spec"]
#         scorecard = parsed_sections["scorecard"]
#         candidate_cv = parsed_sections["candidate_cv"]
#         interview_transcript = parsed_sections["interview_transcript"]
#         consultant_assessment = parsed_sections["consultant_assessment"]

#         # Process files if any
#         if attachment_paths:
#             for path in attachment_paths:
#                 content = self._extract_text_from_file(path)
#                 fname = os.path.basename(path).lower()

#                 if not content:
#                     continue
#                 if "job" in fname or "spec" in fname:
#                     job_spec += "\n\n" + content
#                 elif "scorecard" in fname:
#                     scorecard += "\n\n" + content
#                 elif "cv" in fname or "resume" in fname:
#                     candidate_cv += "\n\n" + content
#                 elif "interview" in fname or "transcript" in fname:
#                     interview_transcript += "\n\n" + content
#                 elif "assessment" in fname or "consultant" in fname:
#                     consultant_assessment += "\n\n" + content
#                 else:
#                     interview_transcript += "\n\n" + content  # Fallback

#         # Final input defaults
#         job_spec = job_spec.strip() or "Not provided."
#         scorecard = scorecard.strip() or "Not provided."
#         candidate_cv = candidate_cv.strip() or "Not provided."
#         interview_transcript = interview_transcript.strip() or "Not provided."
#         consultant_assessment = consultant_assessment.strip() or input_text or "Not provided."

#         input_data = {
#             "role": self.role,
#             "goal": self.goal,
#             "job_spec": job_spec,
#             "scorecard": scorecard,
#             "candidate_cv": candidate_cv,
#             "interview_transcript": interview_transcript,
#             "consultant_assessment": consultant_assessment,
#         }

#         if self.verbose:
#             print("Final input to LLM:")
#             for k, v in input_data.items():
#                 print(f"\n--- {k} ---\n{v[:500]}\n")

#         return self.chain.run(input_data)

#     def _parse_input_text(self, text):
#         """
#         Try to parse structured sections from input text if the user included delimiters.
#         """
#         sections = {
#             "job_spec": "",
#             "scorecard": "",
#             "candidate_cv": "",
#             "interview_transcript": "",
#             "consultant_assessment": "",
#         }

#         patterns = {
#             "job_spec": r"---JOB SPEC---\s*(.+?)(?=---|$)",
#             "scorecard": r"---SCORECARD---\s*(.+?)(?=---|$)",
#             "candidate_cv": r"---CANDIDATE CV---\s*(.+?)(?=---|$)",
#             "interview_transcript": r"---INTERVIEW TRANSCRIPT---\s*(.+?)(?=---|$)",
#             "consultant_assessment": r"---CONSULTANT ASSESSMENT---\s*(.+?)(?=---|$)",
#         }

#         for key, pattern in patterns.items():
#             match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
#             if match:
#                 sections[key] = match.group(1).strip()

#         return sections

import os
import json
import re
from typing import List, Optional
from langchain_community.llms import OpenAI # Keep or switch
from langchain_community.chat_models import ChatOpenAI # Recommended
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import fitz # PyMuPDF for PDF reading
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class InterviewReportCreatorAgent:
    def __init__(self, verbose=False):
        self.role = "an expert talent assessment consultant specializing in executive search"
        self.goal = (
            "to generate concise (~300-400 words per main section), insightful, and structured interview reports for client presentations. "
            "The report must clearly articulate the candidate's suitability for the role, backed by evidence from their CV, interview, and any consultant assessments."
        )
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("InterviewReportCreatorAgent: OPENAI_API_KEY not found.")
            raise ValueError("OPENAI_API_KEY not found.")
        
        try:
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                temperature=0.6, # Slightly lower for more factual reporting
                max_tokens=3500, # Max output tokens
                model_name="gpt-4" # Or gpt-3.5-turbo-16k for longer context if gpt-4 is too expensive
            )
            # self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7, max_tokens=3500)
        except Exception as e:
            logger.exception(f"InterviewReportCreatorAgent: Error initializing OpenAI LLM: {e}")
            raise

        self._create_prompt()
        self._create_chain()
        if self.verbose:
            logger.info("InterviewReportCreatorAgent initialized.")

    def _create_prompt(self):
        template = """You are {role}. Your primary goal is {goal}.

You are tasked with creating a professional interview report based on the provided information.
The report should follow this specific structure, with each main section being approximately 300-400 words:

**INTERVIEW REPORT**

**Candidate:** [Name, if inferable, otherwise "The Candidate"]
**Role:** [Role Title, if inferable, otherwise "The Discussed Role"]
**Date of Report:** [Today's Date - you don't need to fill this, it's for structure]

**1. Candidate Overview & Background (Approx. 300-400 words):**
   - Start with a concise, impactful opening statement about the candidate (e.g., "XXX is a highly accomplished and articulate professional...").
   - Provide a summary of their overall career trajectory, key experiences, and significant achievements as detailed in their CV and discussed in the interview.
   - Mention educational background if relevant and prominent.

**2. Alignment with Role & Career Motivation (Approx. 300-400 words):**
   - Analyze how the candidate's skills, experience, and past accomplishments align with the requirements of the target role (as per Job Spec and Scorecard, if available).
   - Discuss their stated motivations for considering this new opportunity and how it fits into their career development aspirations.
   - Highlight specific examples from their background that demonstrate suitability for key responsibilities.

**3. Consultant's Assessment & Recommendation (Approx. 300-400 words):**
   - This section should heavily reflect the "Consultant's Assessment" input.
   - Summarize the interviewing consultant's overall impression of the candidate.
   - Detail perceived strengths relevant to the role.
   - Candidly discuss any potential weaknesses, areas for development, or aspects that might require further exploration or validation.
   - Conclude with a clear recommendation (e.g., "Highly Recommended for Next Steps," "Recommended with Considerations," "Not Recommended at This Time") and a brief justification.

**Key Information Provided:**
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

Synthesize all available information to produce a coherent, well-written report.
If some information (e.g., scorecard) is "Not provided," acknowledge that and focus on available data.
Maintain a professional, objective, yet insightful tone.
"""
        self.prompt_template = PromptTemplate(
            input_variables=[
                "role",
                "goal",
                "job_spec",
                "scorecard",
                "candidate_cv",
                "interview_transcript",
                "consultant_assessment",
            ],
            template=template,
        )

    def _create_chain(self):
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=self.verbose)

    def _read_pdf(self, path: str) -> str:
        text = ""
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text()
            logger.info(f"InterviewReportAgent: Successfully extracted text from PDF: {path}")
        except Exception as e:
            logger.error(f"InterviewReportAgent: Failed to extract text from PDF {path}: {e}")
            return ""
        return text.strip()

    def _read_txt(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            logger.info(f"InterviewReportAgent: Successfully read text from TXT: {path}")
            return text.strip()
        except Exception as e:
            logger.error(f"InterviewReportAgent: Failed to read TXT file {path}: {e}")
            return ""

    def _read_json_as_text(self, path: str) -> str: # Renamed for clarity
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Convert JSON to a readable string. If it's a transcript, try to format it nicely.
            # Heuristic for transcripts (often a list of dicts with 'speaker' and 'text')
            if isinstance(data, list) and data and isinstance(data[0], dict):
                lines = []
                for item in data:
                    speaker = item.get('speaker', item.get('Speaker', 'Unknown'))
                    line = item.get('text', item.get('line', item.get('transcript', '')))
                    if line: # Only add if there's text
                         lines.append(f"{speaker}: {line}")
                if lines:
                    formatted_text = "\n".join(lines)
                    logger.info(f"InterviewReportAgent: Formatted JSON transcript from: {path}")
                    return formatted_text.strip()
            
            # Fallback for other JSON structures: pretty print
            text_content = json.dumps(data, indent=2)
            logger.info(f"InterviewReportAgent: Read and formatted generic JSON from: {path}")
            return text_content.strip()
        except Exception as e:
            logger.error(f"InterviewReportAgent: Failed to read or process JSON file {path}: {e}")
            return ""

    def _extract_text_from_file(self, file_path: str) -> str:
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"InterviewReportAgent: File path is invalid or file does not exist: {file_path}")
            return ""
            
        ext = os.path.splitext(file_path)[-1].lower()
        content = ""
        try:
            if ext == ".pdf":
                content = self._read_pdf(file_path)
            elif ext == ".txt":
                content = self._read_txt(file_path)
            elif ext == ".json":
                content = self._read_json_as_text(file_path) # Use the renamed method
            else:
                if self.verbose:
                    logger.warning(f"InterviewReportAgent: Unsupported file type {ext} for {file_path}. Ignored.")
                return "" # Explicitly return empty for unsupported
            
            # Add a header to the content indicating its source file
            if content:
                return f"\n\n--- Content from File: {os.path.basename(file_path)} ---\n{content}"
            return "" # If file processing yielded no content
            
        except Exception as e: # Catch-all for unexpected errors during file processing
            if self.verbose:
                logger.error(f"InterviewReportAgent: Error processing file {file_path}: {e}")
            return ""

    def _parse_input_text_sections(self, text: str) -> dict: # Renamed for clarity
        """
        Parses structured sections from input text if the user included delimiters.
        Delimiters are case-insensitive.
        """
        sections = {
            "job_spec": "",
            "scorecard": "",
            "candidate_cv": "",
            "interview_transcript": "",
            "consultant_assessment": "",
        }
        if not text: # No text to parse
            return sections

        # Define patterns for each section. Using re.IGNORECASE and re.DOTALL
        # (?s) is equivalent to re.DOTALL for inline.
        # Using lookaheads to ensure non-greedy matching up to the next delimiter or end of string.
        patterns = {
            "job_spec": r"---JOB SPEC---\s*(.*?)(?=---[A-Z\s]+---|$)",
            "scorecard": r"---SCORECARD---\s*(.*?)(?=---[A-Z\s]+---|$)",
            "candidate_cv": r"---CANDIDATE CV---\s*(.*?)(?=---[A-Z\s]+---|$)",
            "interview_transcript": r"---INTERVIEW TRANSCRIPT---\s*(.*?)(?=---[A-Z\s]+---|$)",
            "consultant_assessment": r"---CONSULTANT ASSESSMENT---\s*(.*?)(?=---[A-Z\s]+---|$)",
        }

        remaining_text = text # For default assignment if no sections match
        found_any_section = False

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[key] = match.group(1).strip()
                found_any_section = True
                # Remove matched part from remaining_text (simplistic, might need refinement for overlapping)
                # This is mainly for the fallback if consultant_assessment is not explicitly tagged
                remaining_text = remaining_text.replace(match.group(0), "", 1) 


        # If no specific sections were parsed but input_text was provided,
        # and consultant_assessment is still empty, assume the whole input_text is the consultant_assessment.
        if not found_any_section and text.strip() and not sections["consultant_assessment"]:
            sections["consultant_assessment"] = text.strip()
            if self.verbose:
                logger.info("InterviewReportAgent: No delimiters found in input_text, treating all as consultant_assessment.")
        elif not sections["consultant_assessment"] and remaining_text.strip():
            # If some sections were parsed, but consultant_assessment wasn't,
            # and there's other text not captured by delimiters, assign it to consultant_assessment.
            sections["consultant_assessment"] = remaining_text.strip()
            if self.verbose:
                logger.info("InterviewReportAgent: Assigning remaining untagged input_text to consultant_assessment.")


        if self.verbose:
            for key, value in sections.items():
                logger.info(f"InterviewReportAgent: Parsed from input_text for '{key}': {value[:100]}...")
        return sections

    def run(self, input_text: str = "", attachment_paths: Optional[List[str]] = None) -> str:
        if self.verbose:
            logger.info(f"InterviewReportAgent: Running with input_text (preview): {input_text[:100]}...")
            logger.info(f"InterviewReportAgent: Processing {len(attachment_paths) if attachment_paths else 0} attachment files.")

        # Initialize content placeholders
        content_map = {
            "job_spec": "",
            "scorecard": "",
            "candidate_cv": "",
            "interview_transcript": "",
            "consultant_assessment": "",
        }

        # 1. Parse structured sections from input_text first
        if input_text:
            parsed_sections = self._parse_input_text_sections(input_text)
            for key, value in parsed_sections.items():
                if value: # Only update if parsing yielded content
                    content_map[key] = value

        # 2. Process attachment files, appending their content
        #    Files can override or add to parsed sections based on keywords in filename.
        if attachment_paths:
            for path in attachment_paths:
                file_content = self._extract_text_from_file(path) # This now includes a header
                if not file_content:
                    continue # Skip if file reading failed or unsupported

                fname_lower = os.path.basename(path).lower()
                assigned_to_key = False

                # Prioritize assignment based on keywords
                if "job" in fname_lower or "spec" in fname_lower or "jd" in fname_lower:
                    content_map["job_spec"] += f"\n{file_content}"
                    assigned_to_key = True
                elif "scorecard" in fname_lower or "criteria" in fname_lower:
                    content_map["scorecard"] += f"\n{file_content}"
                    assigned_to_key = True
                elif "cv" in fname_lower or "resume" in fname_lower or "candidate" in fname_lower: # Candidate might also be interview notes
                    content_map["candidate_cv"] += f"\n{file_content}"
                    assigned_to_key = True
                elif "interview" in fname_lower or "transcript" in fname_lower or "notes" in fname_lower:
                    content_map["interview_transcript"] += f"\n{file_content}"
                    assigned_to_key = True
                elif "assessment" in fname_lower or "consultant" in fname_lower or "evaluation" in fname_lower:
                    content_map["consultant_assessment"] += f"\n{file_content}"
                    assigned_to_key = True
                
                if not assigned_to_key:
                    # Fallback: if filename doesn't match specific keywords,
                    # add its content to a general pool, perhaps 'interview_transcript' or a new 'other_documents' field.
                    # For now, adding to 'interview_transcript' as a common place for miscellaneous notes.
                    content_map["interview_transcript"] += f"\n{file_content}"
                    if self.verbose:
                        logger.info(f"InterviewReportAgent: File {fname_lower} assigned to 'interview_transcript' by fallback.")
        
        # Prepare final input for the LLM, ensuring no Nones and providing defaults
        final_llm_input = {
            "role": self.role,
            "goal": self.goal,
            "job_spec": content_map["job_spec"].strip() or "Not provided.",
            "scorecard": content_map["scorecard"].strip() or "Not provided.",
            "candidate_cv": content_map["candidate_cv"].strip() or "Not provided.",
            "interview_transcript": content_map["interview_transcript"].strip() or "Not provided.",
            # Consultant assessment is crucial. If completely empty, use a placeholder.
            "consultant_assessment": content_map["consultant_assessment"].strip() or "No specific consultant assessment provided. Please synthesize from other available information.",
        }

        if self.verbose:
            logger.info("InterviewReportAgent: Final input to LLM:")
            for k, v_preview in final_llm_input.items():
                logger.info(f"--- Key: {k} ---")
                logger.info(f"{str(v_preview)[:500]}...\n")
        
        try:
            report_result = self.chain.run(final_llm_input)
            # If using ChatModel and invoke:
            # report_dict = self.chain.invoke(final_llm_input)
            # report_result = report_dict.get('text', "Error: Could not generate interview report.")
            logger.info("InterviewReportAgent: Report generated successfully.")
            return report_result.strip()
        except Exception as e:
            logger.exception("InterviewReportAgent: Error during LLM chain execution for report generation.")
            return f"Error: Could not generate interview report. Details: {str(e)}"