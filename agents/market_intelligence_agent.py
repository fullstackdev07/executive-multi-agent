import os
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import requests
import re
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

app = Flask(__name__)
CORS(app)

def ensure_str(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    return str(value)


class MarketIntelligenceAgent:
    def __init__(self, verbose: bool = False, max_tokens: int = 1024):
        self.name = "Market Intelligence Analyst"
        self.role = "a highly skilled market research analyst"
        self.goal = "to gather information about a company and generate a detailed market intelligence report"
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0.7,
            model_name="gpt-3.5-turbo-16k",
            max_tokens=max_tokens,
        )

        self.prompt_template = None
        self.chain = None

    def _create_prompt(self) -> PromptTemplate:
        template = """You are {role}. Your goal is {goal}.

    You are provided with the following information about a company:
    - Company Name: {company_name}
    - Location: {company_location}
    - Geography of Focus: {geography}
    - Supporting Documents: {supporting_documents}

    Your task is to generate a comprehensive market intelligence report outlining:

    1. **Company Data:** Research and describe the company "{company_name}" in detail. Include revenue, employees, locations, business lines, products, history, and ownership. Use the documents provided and your own knowledge.
    2. **Market Overview:** An overview of the market in which the company operates (trends, challenges, competitors). Use both provided documents and your own knowledge.
    3. **Company Positioning:** The company's positioning in the market. Compare with competitors, and identify strengths, weaknesses, opportunities, and threats.

    Structure the report into:
    1. Executive Summary
    2. Company Overview
    3. Market Overview
    4. Competitive Landscape
    5. Company Positioning and SWOT Analysis
    6. Conclusion and Recommendations

    Report:"""

        return PromptTemplate(
            input_variables=[
                "role",
                "goal",
                "company_name",
                "company_location",
                "geography",
                "supporting_documents",
            ],
            template=template,
        )

    def _create_chain(self) -> LLMChain:
        if not self.prompt_template:
            raise ValueError("Prompt template must be created before creating the LLMChain.")
        return LLMChain(llm=self.llm, prompt=self.prompt_template)

    def run(self, input_data: dict) -> str:
        """Executes the agent based on the provided input data."""
        if not self.chain:
            self.prompt_template = self._create_prompt()
            self.chain = self._create_chain()

        if self.verbose:
            print(f"\nRunning {self.name} with input: {input_data}")

        # Validate company information
        if input_data.get("company_name", "").strip().lower() == "unknown":
            return json.dumps("Could not extract a valid company name. Please provide more detailed or accurate input.")

        response = self.chain.run(input_data)

        if self.verbose:
            print(f"\n{self.name} response: {response}")

        return response

    def load_text_from_file(self, filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except FileNotFoundError:
            print(f"Warning: File not found: {filepath}")
            return ""
        except UnicodeDecodeError as e:
            print(f"Warning: Error decoding file (UTF-8): {e}")
            return ""

    def load_pdf_from_file(self, filepath: str) -> str:
        try:
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
        except FileNotFoundError:
            print(f"Warning: File not found: {filepath}")
            return ""
        except Exception as e:
            return f"Warning: Error loading PDF: {e}"

    def save_report_to_file(self, report: str, filepath: str):
        try:
            with open(filepath, 'w') as f:
                f.write(report)
            print(f"Report saved to {filepath}")
        except Exception as e:
            print(f"Error saving report: {e}")

    def extract_company_details(self, company_information: str) -> dict:
        extract_prompt_template = """You are an expert at extracting company information from unstructured text.

You are given the following text:
{company_information}

Extract:
1. Company Name
2. Company Location
3. Geography of Focus

Return a JSON object with keys: "company_name", "company_location", "geography".
If unknown, return "Unknown" for that field.
"""

        extract_prompt = PromptTemplate(
            input_variables=["company_information"],
            template=extract_prompt_template,
        )

        extract_chain = LLMChain(llm=self.llm, prompt=extract_prompt)
        information = extract_chain.run(company_information)

        try:
            return json.loads(information)
        except json.JSONDecodeError:
            print(f"Warning: Could not extract company details from input. Raw Information: {information}")
            return {
                "company_name": "Unknown",
                "company_location": "Unknown",
                "geography": "Unknown",
            }

if __name__ == '__main__':
    agent = MarketIntelligenceAgent(verbose=True)
