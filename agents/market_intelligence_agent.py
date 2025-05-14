import os
from langchain_community.llms import OpenAI  
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json  

load_dotenv()

class MarketIntelligenceAgent:
    """
    A search agent that analyzes a conversation from a file and provides market intelligence insights.
    """
    def __init__(self, verbose: bool = False):
        self.name = "Market Intelligence Analyst"
        self.role = "a highly skilled market research analyst"
        self.goal = "to analyze a conversation and provide market intelligence insights"
        self.verbose = verbose 
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found.  Please set the OPENAI_API_KEY environment variable.")
        self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7) 
        self.prompt_template = None 
        self.chain = None 

    def _create_prompt(self) -> PromptTemplate:
        template = """You are {role}. Your goal is {goal}.

        You are provided with a conversation between two people. Your task is to analyze the conversation and provide market intelligence insights.
        The conversation is:
        {conversation}

        Provide a comprehensive market intelligence report based on the conversation.
        Focus on identifying key themes, market trends, potential opportunities, and competitive threats that are discussed or implied in the conversation.

        Structure the report into the following sections:
        1. Executive Summary
        2. Key Themes
        3. Market Trends
        4. Potential Opportunities
        5. Competitive Threats

        Report:"""

        return PromptTemplate(
            input_variables=["role", "goal", "conversation"],
            template=template,
        )

    def _create_chain(self) -> LLMChain:
        """
        Initializes the LLMChain using the prompt template and LLM.
        """
        if not self.prompt_template:
            raise ValueError("Prompt template must be created before creating the LLMChain.")
        return LLMChain(llm=self.llm, prompt=self.prompt_template)

    def run(self, input_data: dict) -> str:
        """
        Executes the agent based on the provided input data.
        """
        if not self.chain:
            self.prompt_template = self._create_prompt()
            self.chain = self._create_chain()

        if self.verbose:
            print(f"\nRunning {self.name} with input: {input_data}")

        response = self.chain.run(input_data)

        if self.verbose:
            print(f"\n{self.name} response: {response}")

        return response

    def load_conversation_from_file(self, filepath: str) -> str:
        """
        Loads a conversation from a text file.

        Args:
            filepath: The path to the text file.

        Returns:
            A string representation of the conversation.
        """
        try:
            with open(filepath, 'r') as f:
                conversation = f.read()
            return conversation
        except FileNotFoundError:
            return "Error: Conversation file not found."


    def save_report_to_file(self, report: str, filepath: str):
        """
        Saves the market intelligence report to a file.

        Args:
            report: The market intelligence report.
            filepath: The path to the file.
        """
        try:
            with open(filepath, 'w') as f:
                f.write(report)
            print(f"Report saved to {filepath}")
        except Exception as e:
            print(f"Error saving report: {e}")


if __name__ == "__main__":
    agent = MarketIntelligenceAgent(verbose=True)
    conversation_filepath = '../regular_data/conversation.txt'
    conversation = agent.load_conversation_from_file(conversation_filepath)

    market_input = {
        "conversation": conversation,
        "role": agent.role,
        "goal": agent.goal
    }
    market_report = agent.run(market_input)
    print("\nMarket Intelligence Report:\n", market_report)

    report_filepath = '../output_data/market_report.txt' 
    agent.save_report_to_file(market_report, report_filepath)