import os
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()

class InterviewReportCreatorAgent:
    """
    A search agent that generates interview reports based on job specs, resumes, interview transcripts, prompts the user for consultant assessment.
    """
    def __init__(self, verbose: bool = False):
        self.name = "Interview Report Creator"
        self.role = "an expert in creating interview reports"
        self.goal = "to generate comprehensive and informative interview reports"
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found.  Please set the OPENAI_API_KEY environment variable.")
        self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7)
        self.prompt_template = None
        self.chain = None

    def _create_prompt(self) -> PromptTemplate:
        template = """You are {role}. Your goal is {goal}.

        You will use the following information to generate a comprehensive interview report on the candidate [Candidate Name].

        Here is the job spec: {job_spec}

        Here is the candidate's CV: {candidate_cv}

        Here is the interview transcript: {interview_transcript}

        You will prompt the user to provide you with a high-level assessment of the candidate.
        Once you get the prompt, create a structured interview report that includes the following sections:

        1. Candidate Summary: A brief overview of the candidate's background and experience.
        2. Qualifications Assessment: An analysis of the candidate's qualifications against the job requirements.
        3. Interview Performance: A summary of the candidate's performance during the interview, including their communication skills, problem-solving abilities, and overall demeanor.
        4. Strengths and Weaknesses: A detailed assessment of the candidate's strengths and weaknesses.
        5. Overall Recommendation: Your overall recommendation regarding the candidate's suitability for the role.

        Use clear, concise language and provide specific examples from the interview transcript to support your assessment."""

        return PromptTemplate(
            input_variables=[
                "role",
                "goal",
                "job_spec",
                "candidate_cv",
                "interview_transcript",
            ],
            template=template,
        )

    def _create_chain(self) -> LLMChain:
        if not self.prompt_template:
            raise ValueError("Prompt template must be created before creating the LLMChain.")
        return LLMChain(llm=self.llm, prompt=self.prompt_template)

    def run(self, input_data: dict, consultant_assessment: str) -> str:
        if not self.chain:
            self.prompt_template = self._create_prompt()
            self.chain = self._create_chain()

        if self.verbose:
            print(f"\nRunning {self.name} with input: {input_data}")

        input_data["consultant_assessment"] = consultant_assessment

        response = self.chain.run(input_data)

        if self.verbose:
            print(f"\n{self.name} response: {response}")

        return response

    def load_text_from_file(self, filepath: str) -> str:
        """
        Loads text from a file.

        Args:
            filepath: The path to the file.

        Returns:
            A string representation of the file contents.
        """
        try:
            with open(filepath, 'r') as f:
                text = f.read()
            return text
        except FileNotFoundError:
            return "Error: File not found."

    def save_report_to_file(self, report: str, filepath: str):
        """
        Saves the interview report to a file.

        Args:
            report: The interview report.
            filepath: The path to the file.
        """
        try:
            with open(filepath, 'w') as f:
                f.write(report)
            print(f"Interview report saved to {filepath}")
        except Exception as e:
            print(f"Error saving interview report: {e}")

if __name__ == "__main__":
    agent = InterviewReportCreatorAgent(verbose=True)

    job_spec_filepath = '../output_data/job_description.txt' 
    candidate_cv_filepath = '../regular_data/candidate_cv.txt'
    interview_transcript_filepath = '../regular_data/interview_transcript.txt'

    job_spec = agent.load_text_from_file(job_spec_filepath)
    candidate_cv = agent.load_text_from_file(candidate_cv_filepath)
    interview_transcript = agent.load_text_from_file(interview_transcript_filepath)

    consultant_assessment = input("Enter the consultant's high-level assessment of the candidate: ")

    input_data = {
        "role": agent.role,
        "goal": agent.goal,
        "job_spec": job_spec,
        "candidate_cv": candidate_cv,
        "interview_transcript": interview_transcript,
    }

    report = agent.run(input_data, consultant_assessment)
    print("\nInterview Report:\n", report)

    report_filepath = '../output_data/interview_report.txt'
    agent.save_report_to_file(report, report_filepath)