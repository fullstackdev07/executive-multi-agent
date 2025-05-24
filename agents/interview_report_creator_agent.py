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
#         self.role = "an expert in creating interview reports"
#         self.goal = "to generate comprehensive and informative interview reports"
#         self.verbose = verbose
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key:
#             raise ValueError("OPENAI_API_KEY not found.")
#         self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7)
#         self._create_prompt()
#         self._create_chain()

#     def _create_prompt(self):
#         template = """You are {role}. Your goal is {goal}.

# {guidance_prompt}

# Here is the job spec: {job_spec}

# Here is the candidate's CV: {candidate_cv}

# Here is the interview transcript: {interview_transcript}

# Create a structured interview report including:
# 1. Candidate Summary
# 2. Qualifications Assessment
# 3. Interview Performance
# 4. Strengths and Weaknesses
# 5. Overall Recommendation
# """
#         self.prompt_template = PromptTemplate(
#             input_variables=[
#                 "role",
#                 "goal",
#                 "guidance_prompt",
#                 "job_spec",
#                 "candidate_cv",
#                 "interview_transcript",
#             ],
#             template=template,
#         )

#     def _create_chain(self):
#         self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

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
#                 # For JSON return stringified or extract relevant keys as string
#                 data = self._read_json(file_path)
#                 # Flatten all text from JSON for now
#                 return json.dumps(data)
#             else:
#                 if self.verbose:
#                     print(f"Unsupported file type {ext}. Ignored.")
#                 return ""
#         except Exception as e:
#             if self.verbose:
#                 print(f"Error reading {file_path}: {e}")
#             return ""

#     def parse_manual_input(self, manual_input: str):
#         # Parse text based on delimiters (case insensitive)
#         sections = {
#             "job_spec": "",
#             "candidate_cv": "",
#             "interview_transcript": "",
#             "guidance_prompt": "",
#         }
#         patterns = {
#             "job_spec": r"---JOB SPEC---\s*(.+?)(?=---|$)",
#             "candidate_cv": r"---CANDIDATE CV---\s*(.+?)(?=---|$)",
#             "interview_transcript": r"---INTERVIEW TRANSCRIPT---\s*(.+?)(?=---|$)",
#             "guidance_prompt": r"---GUIDANCE PROMPT---\s*(.+?)(?=---|$)",
#         }
#         for key, pattern in patterns.items():
#             match = re.search(pattern, manual_input, re.DOTALL | re.IGNORECASE)
#             if match:
#                 sections[key] = match.group(1).strip()
#         return sections

#     def run(self, manual_input: str, attachment_paths: list = None):
#         # Parse manual input box
#         parsed_inputs = self.parse_manual_input(manual_input)

#         # Initialize combined content placeholders
#         combined_job_spec = []
#         combined_cv = []
#         combined_transcript = []

#         # Process multiple attachments
#         if attachment_paths:
#             for path in attachment_paths:
#                 text = self._extract_text_from_file(path)
#                 # Simple heuristic to classify the content by keywords in filename
#                 lower_name = os.path.basename(path).lower()
#                 if "job" in lower_name or "spec" in lower_name:
#                     combined_job_spec.append(text)
#                 elif "cv" in lower_name or "resume" in lower_name:
#                     combined_cv.append(text)
#                 elif "transcript" in lower_name or "interview" in lower_name:
#                     combined_transcript.append(text)
#                 else:
#                     # If no clue, append to transcript as fallback
#                     combined_transcript.append(text)

#         # Use manual input first if available, else fallback to combined file texts
#         job_spec = parsed_inputs["job_spec"] or "\n\n".join(combined_job_spec) or "Not provided."
#         candidate_cv = parsed_inputs["candidate_cv"] or "\n\n".join(combined_cv) or "Not provided."
#         interview_transcript = parsed_inputs["interview_transcript"] or "\n\n".join(combined_transcript) or "Not provided."
#         guidance_prompt = parsed_inputs["guidance_prompt"] or "Use your best judgment."

#         input_data = {
#             "role": self.role,
#             "goal": self.goal,
#             "guidance_prompt": guidance_prompt,
#             "job_spec": job_spec,
#             "candidate_cv": candidate_cv,
#             "interview_transcript": interview_transcript,
#         }

#         if self.verbose:
#             print("Final input to LLM:")
#             for k, v in input_data.items():
#                 print(f"--- {k} ---\n{v[:500]}\n")

#         return self.chain.run(input_data)

import os
import json
import re
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

class InterviewReportCreatorAgent:
    def __init__(self, verbose=False):
        self.role = "an expert interviewer compiling reports for client presentations"
        self.goal = (
            "to generate concise (~300 words), structured interview reports for clients, "
            "emphasizing the consultant's assessment and the candidate's fit for the role."
        )
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found.")
        self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7, max_tokens=3500)
        self._create_prompt()
        self._create_chain()

    def _create_prompt(self):
        template = """You are {role}. Your goal is {goal}.

        Write a 300-word interview report following this structure:
        1. A global description of the candidate, starting with a general statement (e.g. "XXX is a very metropolitan and well educated candidate..."), followed by a summary of their experience.
        2. A paragraph that outlines their career development and how it aligns with the current role.
        3. A paragraph that summarizes the consultant's recommendation, and highlights any areas that need further exploration.

        Here is the job description: {job_spec}

        Here is the role scorecard (if available): {scorecard}

        Here is the candidate's CV: {candidate_cv}

        Here is the interview transcript: {interview_transcript}

        Here is the consultant's assessment (weigh this heavily): {consultant_assessment}
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
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template,)

    def _read_pdf(self, path):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _read_txt(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _read_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_text_from_file(self, file_path):
        ext = os.path.splitext(file_path)[-1].lower()
        try:
            if ext == ".pdf":
                return self._read_pdf(file_path)
            elif ext == ".txt":
                return self._read_txt(file_path)
            elif ext == ".json":
                data = self._read_json(file_path)
                return json.dumps(data)
            else:
                if self.verbose:
                    print(f"Unsupported file type {ext}. Ignored.")
                return ""
        except Exception as e:
            if self.verbose:
                print(f"Error reading {file_path}: {e}")
            return ""

    def run(self, input_text: str = "", attachment_paths: list = None):
        # Initialize content placeholders
        job_spec = ""
        scorecard = ""
        candidate_cv = ""
        interview_transcript = ""
        consultant_assessment = ""

        # If user provided structured content in text, try to extract using delimiters
        parsed_sections = self._parse_input_text(input_text)

        job_spec = parsed_sections["job_spec"]
        scorecard = parsed_sections["scorecard"]
        candidate_cv = parsed_sections["candidate_cv"]
        interview_transcript = parsed_sections["interview_transcript"]
        consultant_assessment = parsed_sections["consultant_assessment"]

        # Process files if any
        if attachment_paths:
            for path in attachment_paths:
                content = self._extract_text_from_file(path)
                fname = os.path.basename(path).lower()

                if not content:
                    continue
                if "job" in fname or "spec" in fname:
                    job_spec += "\n\n" + content
                elif "scorecard" in fname:
                    scorecard += "\n\n" + content
                elif "cv" in fname or "resume" in fname:
                    candidate_cv += "\n\n" + content
                elif "interview" in fname or "transcript" in fname:
                    interview_transcript += "\n\n" + content
                elif "assessment" in fname or "consultant" in fname:
                    consultant_assessment += "\n\n" + content
                else:
                    interview_transcript += "\n\n" + content  # Fallback

        # Final input defaults
        job_spec = job_spec.strip() or "Not provided."
        scorecard = scorecard.strip() or "Not provided."
        candidate_cv = candidate_cv.strip() or "Not provided."
        interview_transcript = interview_transcript.strip() or "Not provided."
        consultant_assessment = consultant_assessment.strip() or input_text or "Not provided."

        input_data = {
            "role": self.role,
            "goal": self.goal,
            "job_spec": job_spec,
            "scorecard": scorecard,
            "candidate_cv": candidate_cv,
            "interview_transcript": interview_transcript,
            "consultant_assessment": consultant_assessment,
        }

        if self.verbose:
            print("Final input to LLM:")
            for k, v in input_data.items():
                print(f"\n--- {k} ---\n{v[:500]}\n")

        return self.chain.run(input_data)

    def _parse_input_text(self, text):
        """
        Try to parse structured sections from input text if the user included delimiters.
        """
        sections = {
            "job_spec": "",
            "scorecard": "",
            "candidate_cv": "",
            "interview_transcript": "",
            "consultant_assessment": "",
        }

        patterns = {
            "job_spec": r"---JOB SPEC---\s*(.+?)(?=---|$)",
            "scorecard": r"---SCORECARD---\s*(.+?)(?=---|$)",
            "candidate_cv": r"---CANDIDATE CV---\s*(.+?)(?=---|$)",
            "interview_transcript": r"---INTERVIEW TRANSCRIPT---\s*(.+?)(?=---|$)",
            "consultant_assessment": r"---CONSULTANT ASSESSMENT---\s*(.+?)(?=---|$)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[key] = match.group(1).strip()

        return sections