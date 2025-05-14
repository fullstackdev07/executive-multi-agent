import os
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()

class JobDescriptionWriterAgent:
    """
    A search agent that provides compelling and accurate job descriptions.
    """
    def __init__(self, verbose: bool = False):
        self.name = "Job Description Writer"
        self.role = "an expert job description writer"
        self.goal = "to create compelling and accurate job descriptions"
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found.  Please set the OPENAI_API_KEY environment variable.")
        self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7)
        self.prompt_template = None
        self.chain = None

    def _create_prompt(self) -> PromptTemplate:
        template = """You are {role}. Your goal is {goal}.

        Use the following information to create a job description for {job_title} at {client_name}.
        Market Intelligence Report: {market_intelligence}
        Call Transcript: {call_transcript}
        Job Spec (if any): {job_spec}
        Additional Documents: {additional_documents}

        Focus on highlighting the key responsibilities, required skills, and company culture.

        Job Description:"""

        return PromptTemplate(
            input_variables=[
                "role",
                "goal",
                "job_title",
                "client_name",
                "market_intelligence",
                "call_transcript",
                "job_spec",
                "additional_documents",
            ],
            template=template,
        )

    def _create_chain(self) -> LLMChain:
        if not self.prompt_template:
            raise ValueError("Prompt template must be created before creating the LLMChain.")
        return LLMChain(llm=self.llm, prompt=self.prompt_template)

    def run(self, input_data: dict) -> str:
        if not self.chain:
            self.prompt_template = self._create_prompt()
            self.chain = self._create_chain()

        if self.verbose:
            print(f"\nRunning {self.name} with input: {input_data}")

        response = self.chain.run(input_data)

        if self.verbose:
            print(f"\n{self.name} response: {response}")

        return response

    def load_call_transcript_from_json(self, filepath: str) -> str:
        """
        Loads a call transcript from a JSON file and formats it into a string.

        Args:
            filepath: The path to the JSON file.

        Returns:
            A string representation of the call transcript.  Returns an empty string if there's an error.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            transcript_text = ""
            for entry in data:
                speaker = entry.get("speaker", "Unknown") 
                text = entry.get("text", "")
                transcript_text += f"{speaker}: {text}\n"
            return transcript_text
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return ""
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in file {filepath}")
            return ""
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return ""

    def load_report_from_file(self, filepath: str) -> str:
        """
        Loads the market intelligence report from a file.

        Args:
            filepath: The path to the file.

        Returns:
            A string representation of the market intelligence report.
        """
        try:
            with open(filepath, 'r') as f:
                report = f.read()
            return report
        except FileNotFoundError:
            return "Error: Market intelligence report file not found."

    def save_job_description_to_file(self, job_description: str, filepath: str):
        """
        Saves the job description to a file.

        Args:
            job_description: The job description.
            filepath: The path to the file.
        """
        try:
            with open(filepath, 'w') as f:
                f.write(job_description)
            print(f"Job description saved to {filepath}")
        except Exception as e:
            print(f"Error saving job description: {e}")

    def extract_job_details(self, conversation: str) -> dict:
        """
        Extracts job title, client name, and other relevant details from the conversation transcript.

        Args:
            conversation: A string containing the conversation transcript.

        Returns:
            A dictionary containing the extracted job details.
        """
        extract_prompt_template = """You are an expert at extracting job-related information from conversations.

        You are given the following conversation transcript:
        {conversation}

        Your task is to extract the following information:
        1. Job Title: The title of the job being discussed.
        2. Client Name: The name of the client company.
        3. Job Specification: A summary of the client specifications.
        4. Additonal Documennts: Any mention of additonal documennts

        Provide the information in the following format:
        Job Title: [Job Title]
        Client Name: [Client Name]
        Job Specification: [Job Specification]
        Additional Documents: [Additional Documents]"""

        extract_prompt = PromptTemplate(
            input_variables=["conversation"],
            template=extract_prompt_template,
        )

        extract_chain = LLMChain(llm=self.llm, prompt=extract_prompt)
        information = extract_chain.run(conversation)

        try:
            job_title = information.split("Job Title: ")[1].split("\n")[0].strip()
            client_name = information.split("Client Name: ")[1].split("\n")[0].strip()
            job_spec = information.split("Job Specification: ")[1].split("\n")[0].strip()
            additional_documents = information.split("Additional Documents: ")[1].strip()
            return {
                "job_title": job_title,
                "client_name": client_name,
                "job_spec": job_spec,
                "additional_documents": additional_documents,
            }
        except IndexError:
            print("Error: Could not extract job details from the conversation.")
            return {
                "job_title": "Unknown",
                "client_name": "Unknown",
                "job_spec": "No information available.",
                "additional_documents": "No information available."
            }


if __name__ == "__main__":
    agent = JobDescriptionWriterAgent(verbose=True)
    market_report_filepath = '../output_data/market_report.txt'
    call_transcript_filepath = '../regular_data/transcript.json'
    market_intelligence = agent.load_report_from_file(market_report_filepath)
    call_transcript = agent.load_call_transcript_from_json(call_transcript_filepath)

    job_details = agent.extract_job_details(call_transcript)
    job_title = job_details.get('job_title', 'Unknown Job Title')
    client_name = job_details.get('client_name', 'Unknown Client')
    job_spec = job_details.get('job_spec', 'No Job Specification')
    additional_documents = job_details.get('additional_documents', 'No additional documents')

    jd_input = {
        "job_title": job_title,
        "client_name": client_name,
        "market_intelligence": market_intelligence,
        "call_transcript": call_transcript,
        "job_spec": job_spec,
        "additional_documents": additional_documents,
        "role": agent.role,
        "goal": agent.goal
    }
    job_description = agent.run(jd_input)
    print("\nJob Description:\n", job_description)

    job_description_filepath = '../output_data/job_description.txt'
    agent.save_job_description_to_file(job_description, job_description_filepath)