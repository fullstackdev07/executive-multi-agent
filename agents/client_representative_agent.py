# import os
# from langchain_community.llms import OpenAI 
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import json

# load_dotenv()

# class ClientRepresentativeAgent:
#     """
#     A search agent that provides feedback from the client's perspective, extracting client information from a conversation.
#     """
#     def __init__(self, verbose: bool = False):
#         self.name = "Client Representative"
#         self.role = "acting as the client"
#         self.goal = "to review documents and provide feedback from the client's perspective"
#         self.verbose = verbose
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key:
#             raise ValueError("OpenAI API key not found.  Please set the OPENAI_API_KEY environment variable.")
#         self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7)
#         self.prompt_template = None
#         self.chain = None

#     def _create_prompt(self) -> PromptTemplate:
#         template = """You are acting as {client_name}'s {client_title}. You have the following characteristics: {client_characteristics}.

#         You are reviewing the following document: {document}.

#         Provide feedback on the document from the perspective of {client_name}'s {client_title}.
#         Focus on whether the document accurately reflects the role's responsibilities, required skills, and the company's culture.  Be critical and offer specific suggestions for improvement.

#         Feedback:"""

#         return PromptTemplate(
#             input_variables=[
#                 "role",
#                 "goal",
#                 "client_name",
#                 "client_title",
#                 "client_characteristics",
#                 "document",
#             ],
#             template=template,
#         )

#     def _create_chain(self) -> LLMChain:
#         if not self.prompt_template:
#             raise ValueError("Prompt template must be created before creating the LLMChain.")
#         return LLMChain(llm=self.llm, prompt=self.prompt_template)

#     def run(self, input_data: dict) -> str:
#         if not self.chain:
#             self.prompt_template = self._create_prompt()
#             self.chain = self._create_chain()

#         if self.verbose:
#             print(f"\nRunning {self.name} with input: {input_data}")

#         response = self.chain.run(input_data)

#         if self.verbose:
#             print(f"\n{self.name} response: {response}")

#         return response

#     def load_job_description_from_file(self, filepath: str) -> str:
#         """
#         Loads the job description from a file.

#         Args:
#             filepath: The path to the file.

#         Returns:
#             A string representation of the job description.
#         """
#         try:
#             with open(filepath, 'r') as f:
#                 job_description = f.read()
#             return job_description
#         except FileNotFoundError:
#             return "Error: Job description file not found."

#     def save_feedback_to_file(self, feedback: str, filepath: str):
#         """
#         Saves the client feedback to a file.

#         Args:
#             feedback: The client feedback.
#             filepath: The path to the file.
#         """
#         try:
#             with open(filepath, 'w') as f:
#                 f.write(feedback)
#             print(f"Client feedback saved to {filepath}")
#         except Exception as e:
#             print(f"Error saving client feedback: {e}")

#     def extract_client_information(self, conversation: str) -> dict:
#         """
#         Extracts client name, title, and characteristics from a conversation transcript.

#         Args:
#             conversation: A string containing the conversation transcript.

#         Returns:
#             A dictionary containing the client's name, title, and characteristics.
#         """
#         extract_prompt_template = """You are an expert at extracting information from conversations.

#         You are given the following conversation transcript:
#         {conversation}

#         Your task is to extract the following information about the client:
#         1. Client Name: The name of the client.
#         2. Client Title: The client's job title.
#         3. Client Characteristics: A brief description of the client's personality, communication style, priorities, and background.

#         Provide the information in the following format:
#         Client Name: [Client Name]
#         Client Title: [Client Title]
#         Client Characteristics: [Client Characteristics]"""

#         extract_prompt = PromptTemplate(
#             input_variables=["conversation"],
#             template=extract_prompt_template,
#         )

#         extract_chain = LLMChain(llm=self.llm, prompt=extract_prompt)
#         information = extract_chain.run(conversation)

#         try:
#             client_name = information.split("Client Name: ")[1].split("\n")[0].strip()
#             client_title = information.split("Client Title: ")[1].split("\n")[0].strip()
#             client_characteristics = information.split("Client Characteristics: ")[1].strip()
#             return {
#                 "client_name": client_name,
#                 "client_title": client_title,
#                 "client_characteristics": client_characteristics,
#             }
#         except IndexError:
#             print("Error: Could not extract client information from the conversation.")
#             return {
#                 "client_name": "Unknown",
#                 "client_title": "Unknown",
#                 "client_characteristics": "No information available.",
#             }


# if __name__ == "__main__":
#     agent = ClientRepresentativeAgent(verbose=True)
#     job_description_filepath = '../output_data/job_description.txt'
#     job_description = agent.load_job_description_from_file(job_description_filepath)

#     transcript_filepath = '../regular_data/transcript.json'
#     f = open(transcript_filepath)
#     data = json.load(f)
#     transcript_text = ""
#     for entry in data:
#         speaker = entry.get("speaker", "Unknown") 
#         text = entry.get("text", "")
#         transcript_text += f"{speaker}: {text}\n"

#     client_info = agent.extract_client_information(transcript_text)

#     client_rep_input = {
#         "client_name": client_info["client_name"],
#         "client_title": client_info["client_title"],
#         "client_characteristics": client_info["client_characteristics"],
#         "document": job_description,
#         "role": agent.role,
#         "goal": agent.goal
#     }
#     client_feedback = agent.run(client_rep_input)
#     print("\nClient Feedback:\n", client_feedback)

#     feedback_filepath = '../output_data/client_feedback.txt'
#     agent.save_feedback_to_file(client_feedback, feedback_filepath)

# client_representative_agent.py
import os
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()

class ClientRepresentativeAgent:
    """
    A search agent that provides feedback from the client's perspective, extracting client information from a conversation.
    """
    def __init__(self, verbose: bool = False):
        self.name = "Client Representative"
        self.role = "acting as the client"
        self.goal = "to review documents and provide feedback from the client's perspective"
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7)
        self.prompt_template = None
        self.chain = None
        self._create_prompt()
        self._create_chain()

    def _create_prompt(self) -> PromptTemplate:
        template = """You are acting as {client_name}'s {client_title}. You have the following characteristics: {client_characteristics}.

        You are reviewing the following document: {document}.

        Provide feedback on the document from the perspective of {client_name}'s {client_title}.
        Focus on whether the document accurately reflects the role's responsibilities, required skills, and the company's culture.  Be critical and offer specific suggestions for improvement.

        Feedback:"""

        self.prompt_template = PromptTemplate(
            input_variables=[
                "role",
                "goal",
                "client_name",
                "client_title",
                "client_characteristics",
                "document",
            ],
            template=template,
        )

    def _create_chain(self) -> LLMChain:
        if not self.prompt_template:
            raise Exception("Prompt template must be created before creating the LLMChain.")
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def run(self, client_name: str, client_title: str, client_characteristics: str, document: str) -> str:
        input_data = {
            "client_name": client_name,
            "client_title": client_title,
            "client_characteristics": client_characteristics,
            "document": document,
            "role": self.role,
            "goal": self.goal
        }
        response = self.chain.run(input_data)
        return response

    def extract_client_information(self, conversation: str) -> dict:
        """
        Extracts client name, title, and characteristics from a conversation transcript.

        Args:
            conversation: A string containing the conversation transcript.

        Returns:
            A dictionary containing the client's name, title, and characteristics.
        """
        extract_prompt_template = """You are an expert at extracting information from conversations.

        You are given the following conversation transcript:
        {conversation}

        Your task is to extract the following information about the client:
        1. Client Name: The name of the client.
        2. Client Title: The client's job title.
        3. Client Characteristics: A brief description of the client's personality, communication style, priorities, and background.

        Provide the information in the following format:
        Client Name: [Client Name]
        Client Title: [Client Title]
        Client Characteristics: [Client Characteristics]"""

        extract_prompt = PromptTemplate(
            input_variables=["conversation"],
            template=extract_prompt_template,
        )

        extract_chain = LLMChain(llm=self.llm, prompt=extract_prompt)
        information = extract_chain.run(conversation)

        try:
            client_name = information.split("Client Name: ")[1].split("\n")[0].strip()
            client_title = information.split("Client Title: ")[1].split("\n")[0].strip()
            client_characteristics = information.split("Client Characteristics: ")[1].strip()
            return {
                "client_name": client_name,
                "client_title": client_title,
                "client_characteristics": client_characteristics,
            }
        except IndexError:
            print("Error: Could not extract client information from the conversation.")
            return {
                "client_name": "Unknown",
                "client_title": "Unknown",
                "client_characteristics": "No information available.",
            }

# Removed the `if __name__ == "__main__":` block