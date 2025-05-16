# import os
# from langchain_community.llms import OpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# class MarketIntelligenceAgent:
#     """
#     A search agent that analyzes a conversation from a file and provides market intelligence insights.
#     """
#     def __init__(self, verbose: bool = False):
#         self.name = "Market Intelligence Analyst"
#         self.role = "a highly skilled market research analyst"
#         self.goal = "to analyze a conversation and provide market intelligence insights"
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key:
#             raise ValueError("OpenAI API key not found.  Please set the OPENAI_API_KEY environment variable.")
#         self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7)
#         self.prompt_template = None
#         self.chain = None

#     def _create_prompt(self) -> PromptTemplate:
#         template = """You are {role}. Your goal is {goal}.

#         You are provided with a conversation between two people. Analyze the conversation and provide market intelligence insights.
#         The conversation:
#         {conversation}

#         Provide a market intelligence report, focusing on key themes, market trends, potential opportunities, and competitive threats.

#         Report:"""

#         return PromptTemplate(
#             input_variables=["role", "goal", "conversation"],
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

#         response = self.chain.run(input_data)
#         return response

#     def load_conversation_from_file(self, filepath: str) -> str:
#         try:
#             with open(filepath, 'r') as f:
#                 conversation = f.read()
#             return conversation
#         except FileNotFoundError:
#             return "Error: Conversation file not found."

#     def save_report_to_file(self, report: str, filepath: str):
#         try:
#             with open(filepath, 'w') as f:
#                 f.write(report)
#             print(f"Report saved to {filepath}")
#         except Exception as e:
#             print(f"Error saving report: {e}")

# if __name__ == "__main__":
#     agent = MarketIntelligenceAgent()
#     conversation_filepath = '../regular_data/conversation.txt'
#     conversation = agent.load_conversation_from_file(conversation_filepath)

#     market_input = {
#         "conversation": conversation,
#         "role": agent.role,
#         "goal": agent.goal
#     }
#     market_report = agent.run(market_input)
#     agent.save_report_to_file(market_report, '../output_data/market_report.txt')
#     print("Agent finished and created the report")
# market_intelligence_agent.py

import os
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()

class MarketIntelligenceAgent:
    """
    A search agent that analyzes a conversation from a JSON file and provides market intelligence insights.
    """
    def __init__(self, verbose: bool = False):
        self.name = "Market Intelligence Analyst"
        self.role = "a highly skilled market research analyst"
        self.goal = "to analyze a conversation and provide market intelligence insights"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        self.llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0.7)
        self.prompt_template = None
        self.chain = None
        self._create_prompt()
        self._create_chain()

    def _create_prompt(self) -> PromptTemplate:
        template = """You are {role}. Your goal is {goal}.

You are provided with a conversation between two people. Analyze the conversation and provide market intelligence insights.

The conversation:
{conversation}

Provide a market intelligence report, focusing on key themes, market trends, potential opportunities, and competitive threats.

Report:"""

        self.prompt_template = PromptTemplate(
            input_variables=["role", "goal", "conversation"],
            template=template,
        )

    def _create_chain(self) -> LLMChain:
        if not self.prompt_template:
            raise Exception("Prompt template must be created before creating the LLMChain.")
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def run(self, conversation: str) -> str:
        input_data = {
            "conversation": conversation,
            "role": self.role,
            "goal": self.goal
        }
        response = self.chain.run(input_data)
        return response

# Removed the `if __name__ == "__main__":` block