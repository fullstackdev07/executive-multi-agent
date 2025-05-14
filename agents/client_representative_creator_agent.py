import os
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class ClientRepresentativeCreatorAgent:
    """
    A search agent that creates instructions for the ClientRepresentativeAgent by asking questions about the client, maintaining conversation history.
    """
    def __init__(self, verbose: bool = False):
        self.name = "Client Representative Creator"
        self.role = "an expert in creating client personas"
        self.goal = "to gather information and create detailed instructions for the ClientRepresentativeAgent"
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found.  Please set the OPENAI_API_KEY environment variable.")
        self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7)
        self.prompt_template = None
        self.chain = None
        self.conversation_history = "" 

    def _create_prompt(self) -> PromptTemplate:
        template = """You are {role}. Your goal is {goal}.

        You are engaging in a conversation with a user to gather information needed to accurately represent a client in the form of custom instructions. Remember the data to maintain conversation history.

        Here are some example questions:
        * What is the client's job title and role within the company?
        * What is the client's personality and communication style?
        * What are the client's priorities and concerns?
        * What is the client's background and experience?
        * What are the client's expectations for the [Job Title] role?
        * How does the client typically give feedback?
        * Is the client hands-on or high-level?
        * What is the client's tolerance for risk?
        * What are the client's biases or pet peeves?

        Use the following conversation history as reference:
        {conversation_history}

        Continue asking questions until you have a comprehensive understanding of the client's personality and perspective.

        Once you have gathered enough information, summarize the client's persona in a detailed instruction set that can be used by the ClientRepresentativeAgent.

        For example: '[Client Name] is the CEO of a fast-growing SaaS company. He is a highly technical leader with a strong focus on innovation and customer satisfaction. He is direct and to-the-point in his communication style and expects quick results. He is detail oriented, and has a low tolerance for error.'

        Begin! If this is the beginning, start by introducing yourself and asking the first question. If this is not the begining, continue the coversation."""

        return PromptTemplate(
            input_variables=["role", "goal", "conversation_history"],
            template=template,
        )

    def _create_chain(self) -> LLMChain:
        if not self.prompt_template:
            raise ValueError("Prompt template must be created before creating the LLMChain.")
        return LLMChain(llm=self.llm, prompt=self.prompt_template)

    def run(self, user_input: str = "") -> str: 
        if not self.chain:
            self.prompt_template = self._create_prompt()
            self.chain = self._create_chain()

        if user_input:
            self.conversation_history += f"\nUser: {user_input}"

        input_data = {"role": self.role, "goal": self.goal, "conversation_history": self.conversation_history}
        response = self.chain.run(input_data)

        self.conversation_history += f"\nAgent: {response}"

        if self.verbose:
            print(f"\nAgent Response: {response}")

        return response

    def save_client_characteristics_to_file(self, characteristics: str, filepath: str):
        """
        Saves the client characteristics to a file.

        Args:
            characteristics: The client characteristics.
            filepath: The path to the file.
        """
        try:
            with open(filepath, 'w') as f:
                f.write(characteristics)
            print(f"Client characteristics saved to {filepath}")
        except Exception as e:
            print(f"Error saving client characteristics: {e}")

    def generate_and_save_characteristics(self, filepath: str):
        """
        Generates the characteristics and saves to a file
        """

        characteristics = self.run()

        while True:
            user_input = input("Your Reply (Or press enter to save the file): ")
            if not user_input:
                break
            characteristics = self.run(user_input)


        self.save_client_characteristics_to_file(self.conversation_history, filepath)

if __name__ == "__main__":
    agent = ClientRepresentativeCreatorAgent(verbose=True)
    print("Agent is active now, will ask the user to enter data and save")

    file_path = '../output_data/client_characteristics.txt'

    agent.generate_and_save_characteristics(file_path)

    print("\nAgent finished asking questions and creating profile, ensure it is saved to the correct location\n")