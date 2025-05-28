# import os
# from langchain_community.llms import OpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import json
# import requests
# import re
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.document_loaders import PyPDFLoader

# load_dotenv()

# app = Flask(__name__)
# CORS(app)

# def ensure_str(value):
#     if isinstance(value, bytes):
#         return value.decode('utf-8', errors='ignore')
#     return str(value)


# class MarketIntelligenceAgent:
#     def __init__(self, verbose: bool = False, max_tokens: int = 1024):
#         self.name = "Market Intelligence Analyst"
#         self.role = "a highly skilled market research analyst"
#         self.goal = "to gather information about a company and generate a detailed market intelligence report"
#         self.verbose = verbose
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key:
#             raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

#         self.llm = ChatOpenAI(
#             openai_api_key=self.openai_api_key,
#             temperature=0.7,
#             model_name="gpt-3.5-turbo-16k",
#             max_tokens=max_tokens,
#         )

#         self.prompt_template = None
#         self.chain = None

#     def _create_prompt(self) -> PromptTemplate:
#         template = """You are {role}. Your goal is {goal}.

#     You are provided with the following information about a company:
#     - Company Name: {company_name}
#     - Location: {company_location}
#     - Geography of Focus: {geography}
#     - Supporting Documents: {supporting_documents}

#     Your task is to generate a comprehensive market intelligence report outlining:

#     1. **Company Data:** Research and describe the company "{company_name}" in detail. Include revenue, employees, locations, business lines, products, history, and ownership. Use the documents provided and your own knowledge.
#     2. **Market Overview:** An overview of the market in which the company operates (trends, challenges, competitors). Use both provided documents and your own knowledge.
#     3. **Company Positioning:** The company's positioning in the market. Compare with competitors, and identify strengths, weaknesses, opportunities, and threats.

#     Structure the report into:
#     1. Executive Summary
#     2. Company Overview
#     3. Market Overview
#     4. Competitive Landscape
#     5. Company Positioning and SWOT Analysis
#     6. Conclusion and Recommendations

#     Report:"""

#         return PromptTemplate(
#             input_variables=[
#                 "role",
#                 "goal",
#                 "company_name",
#                 "company_location",
#                 "geography",
#                 "supporting_documents",
#             ],
#             template=template,
#         )

#     def _create_chain(self) -> LLMChain:
#         if not self.prompt_template:
#             raise ValueError("Prompt template must be created before creating the LLMChain.")
#         return LLMChain(llm=self.llm, prompt=self.prompt_template)

#     def run(self, input_data: dict) -> str:
#         """Executes the agent based on the provided input data."""
#         if not self.chain:
#             self.prompt_template = self._create_prompt()
#             self.chain = self._create_chain()

#         if self.verbose:
#             print(f"\nRunning {self.name} with input: {input_data}")

#         # Validate company information
#         if input_data.get("company_name", "").strip().lower() == "unknown":
#             return json.dumps({
#                 "error": "Could not extract a valid company name. Please provide more detailed or accurate input."
#             })

#         response = self.chain.run(input_data)

#         if self.verbose:
#             print(f"\n{self.name} response: {response}")

#         return response

#     def load_text_from_file(self, filepath: str) -> str:
#         try:
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 text = f.read()
#             return text
#         except FileNotFoundError:
#             print(f"Warning: File not found: {filepath}")
#             return ""
#         except UnicodeDecodeError as e:
#             print(f"Warning: Error decoding file (UTF-8): {e}")
#             return ""

#     def load_pdf_from_file(self, filepath: str) -> str:
#         try:
#             loader = PyPDFLoader(filepath)
#             documents = loader.load()
#             return "\n".join([doc.page_content for doc in documents])
#         except FileNotFoundError:
#             print(f"Warning: File not found: {filepath}")
#             return ""
#         except Exception as e:
#             return f"Warning: Error loading PDF: {e}"

#     def save_report_to_file(self, report: str, filepath: str):
#         try:
#             with open(filepath, 'w') as f:
#                 f.write(report)
#             print(f"Report saved to {filepath}")
#         except Exception as e:
#             print(f"Error saving report: {e}")

#     def extract_company_details(self, company_information: str) -> dict:
#         extract_prompt_template = """You are an expert at extracting company information from unstructured text.

# You are given the following text:
# {company_information}

# Extract:
# 1. Company Name
# 2. Company Location
# 3. Geography of Focus

# Return a JSON object with keys: "company_name", "company_location", "geography".
# If unknown, return "Unknown" for that field.
# """

#         extract_prompt = PromptTemplate(
#             input_variables=["company_information"],
#             template=extract_prompt_template,
#         )

#         extract_chain = LLMChain(llm=self.llm, prompt=extract_prompt)
#         information = extract_chain.run(company_information)

#         try:
#             return json.loads(information)
#         except json.JSONDecodeError:
#             print(f"Warning: Could not extract company details from input. Raw Information: {information}")
#             return {
#                 "company_name": "Unknown",
#                 "company_location": "Unknown",
#                 "geography": "Unknown",
#             }

# if __name__ == '__main__':
#     agent = MarketIntelligenceAgent(verbose=True)

import os
from langchain_community.llms import OpenAI # Keep for older compatibility if needed
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
#from flask import Flask, request, jsonify # Not needed for agent logic itself
#from flask_cors import CORS # Not needed for agent logic itself
import json
#import requests # Not needed for current agent logic
import re
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# def ensure_str(value): # This helper is not used within the class currently
#     if isinstance(value, bytes):
#         return value.decode('utf-8', errors='ignore')
#     return str(value)


class MarketIntelligenceAgent:
    def __init__(self, verbose: bool = False, max_tokens: int = 2048): # Increased max_tokens for output
        self.name = "Market Intelligence Analyst"
        self.role = "a highly skilled market research analyst"
        self.goal = "to gather information about a company and generate a detailed market intelligence report"
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("MarketIntelligenceAgent: OpenAI API key not found.")
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0.7,
            model_name="gpt-3.5-turbo-16k", # Good for longer reports
            max_tokens=max_tokens, # Max tokens for the *output*
        )

        self.prompt_template = self._create_prompt() # Initialize prompt template
        self.chain = self._create_chain() # Initialize chain
        if self.verbose:
            logger.info("MarketIntelligenceAgent initialized.")

    def _create_prompt(self) -> PromptTemplate:
        template = """You are {role}. Your goal is {goal}.

You have been provided with information about a company and supporting documents.
- Company Name: {company_name}
- Company Location: {company_location}
- Geography of Focus for this analysis: {geography}

--- Supporting Documents Provided ---
{supporting_documents}
--- End of Supporting Documents ---

Based on ALL the information above (company details, geography, and ALL supporting documents) AND your own general knowledge, generate a comprehensive market intelligence report. The report should be well-structured and detailed.

Structure the report as follows:

1.  **Executive Summary:** (Approx. 200-300 words)
    *   A concise overview of the key findings, including the company's current standing, market dynamics, and primary opportunities/threats.

2.  **Company Overview:** (Approx. 400-500 words)
    *   Detailed description of "{company_name}".
    *   Include available details on: revenue, number of employees, key office locations, main business lines/divisions, core products/services, significant historical milestones, and ownership structure (e.g., public, private, VC-backed).
    *   Synthesize information from provided documents and your general knowledge.

3.  **Market Overview for {geography}:** (Approx. 400-500 words)
    *   Analyze the specific market in which "{company_name}" operates, focusing on the specified "{geography}".
    *   Discuss current market size (if data available or estimable), key trends (e.g., technological, regulatory, consumer behavior), significant challenges, and growth opportunities within this geography.
    *   Reference provided documents and supplement with your general market knowledge relevant to "{geography}".

4.  **Competitive Landscape in {geography}:** (Approx. 300-400 words)
    *   Identify key competitors of "{company_name}" operating within "{geography}".
    *   Briefly profile 2-3 major competitors, outlining their strengths and weaknesses relative to "{company_name}".

5.  **Company Positioning and SWOT Analysis for {geography}:** (Approx. 400-500 words)
    *   Evaluate "{company_name}"'s current positioning within the market in "{geography}".
    *   Conduct a SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) for "{company_name}" specifically in the context of "{geography}".
        *   **Strengths:** Internal positive attributes.
        *   **Weaknesses:** Internal negative attributes.
        *   **Opportunities:** External factors the company can leverage.
        *   **Threats:** External factors that could pose a risk.

6.  **Conclusion and Strategic Recommendations:** (Approx. 300-400 words)
    *   Summarize the main conclusions of the report.
    *   Provide 2-3 actionable strategic recommendations for "{company_name}" to enhance its market position or capitalize on opportunities within "{geography}".

Ensure the report is professional, insightful, and directly addresses all sections.

Market Intelligence Report:
"""

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
            logger.error("MarketIntelligenceAgent: Prompt template must be created before creating the LLMChain.")
            raise ValueError("Prompt template must be created before creating the LLMChain.")
        return LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=self.verbose)

    def run(self, input_data: dict) -> str:
        """Executes the agent based on the provided input data."""
        if self.verbose:
            logger.info(f"MarketIntelligenceAgent: Running with input keys: {input_data.keys()}")
            logger.info(f"MarketIntelligenceAgent: Company Name: {input_data.get('company_name')}")
            logger.info(f"MarketIntelligenceAgent: Supporting docs (preview): {str(input_data.get('supporting_documents'))[:200]}...")


        if not input_data.get("company_name") or input_data.get("company_name", "").strip().lower() == "unknown":
            logger.warning("MarketIntelligenceAgent: Valid company name not provided or extracted as 'Unknown'.")
            return json.dumps({
                "error": "Could not extract a valid company name. Please provide more detailed or accurate input. Market intelligence report cannot be generated without a company name."
            })
        
        # Ensure all required keys are present, providing defaults for robustness
        payload = {
            "role": self.role,
            "goal": self.goal,
            "company_name": input_data.get("company_name", "Not Specified"),
            "company_location": input_data.get("company_location", "Not Specified"),
            "geography": input_data.get("geography", "Global"), # Default to Global if not specified
            "supporting_documents": input_data.get("supporting_documents", "No supporting documents provided."),
        }
        
        try:
            response = self.chain.run(payload)
        except Exception as e:
            logger.exception(f"MarketIntelligenceAgent: Error during LLM chain execution: {e}")
            return json.dumps({"error": f"Failed to generate market report due to an internal error: {str(e)}"})


        if self.verbose:
            logger.info(f"MarketIntelligenceAgent response (preview): {response[:300]}...")

        return response

    def load_text_from_file(self, filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            if self.verbose: logger.info(f"MarketIntelligenceAgent: Loaded text from {filepath}")
            return text
        except FileNotFoundError:
            logger.warning(f"MarketIntelligenceAgent: File not found: {filepath}")
            return ""
        except UnicodeDecodeError as e:
            logger.warning(f"MarketIntelligenceAgent: Error decoding file (UTF-8) {filepath}: {e}")
            return ""
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent: General error loading text file {filepath}: {e}")
            return ""


    def load_pdf_from_file(self, filepath: str) -> str:
        try:
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            content = "\n".join([doc.page_content for doc in documents])
            if self.verbose: logger.info(f"MarketIntelligenceAgent: Loaded PDF from {filepath}, length: {len(content)}")
            return content
        except FileNotFoundError:
            logger.warning(f"MarketIntelligenceAgent: PDF File not found: {filepath}")
            return ""
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent: Error loading PDF {filepath}: {e}")
            return f"Warning: Error loading PDF content from {os.path.basename(filepath)}."

    def _load_json_as_text_from_file(self, filepath: str) -> str:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            # Pretty print for LLM readability
            text_content = json.dumps(json_data, indent=2)
            if self.verbose: logger.info(f"MarketIntelligenceAgent: Loaded JSON from {filepath}, length: {len(text_content)}")
            return text_content
        except FileNotFoundError:
            logger.warning(f"MarketIntelligenceAgent: JSON File not found: {filepath}")
            return ""
        except json.JSONDecodeError as e:
            logger.warning(f"MarketIntelligenceAgent: Error decoding JSON file {filepath}: {e}")
            return f"Warning: Error decoding JSON content from {os.path.basename(filepath)}."
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent: Error loading JSON file {filepath}: {e}")
            return f"Warning: Error loading JSON file {os.path.basename(filepath)}."

    def save_report_to_file(self, report: str, filepath: str): # This is a utility, not directly used by run
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"MarketIntelligenceAgent: Report saved to {filepath}")
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent: Error saving report to {filepath}: {e}")

    def extract_company_details(self, company_information: str) -> dict:
        if not company_information or not company_information.strip():
            logger.warning("MarketIntelligenceAgent: Empty company_information string for extraction.")
            return {
                "company_name": "Unknown",
                "company_location": "Unknown",
                "geography": "Unknown",
            }

        extract_prompt_template = """You are an expert at extracting specific company information from unstructured text.

Given the following text:
--- TEXT START ---
{company_information}
--- TEXT END ---

Your task is to extract the following three pieces of information:
1.  **Company Name:** The primary legal or commonly known name of the company.
2.  **Company Location:** The city and country of the company's headquarters or main office mentioned. If multiple locations, pick the most prominent or first mentioned as HQ.
3.  **Geography of Focus:** The primary geographical market or region the company operates in or is targeting, as suggested by the text. This might be a country, a continent, or terms like "global", "EMEA", "North America".

Return the extracted information strictly as a JSON object with the keys: "company_name", "company_location", "geography".
If any piece of information cannot be reliably determined from the text, use the string "Unknown" as its value.

Example for good extraction:
Text: "Innovatech Solutions, based in London, UK, is expanding its operations across Europe."
JSON:
{{
    "company_name": "Innovatech Solutions",
    "company_location": "London, UK",
    "geography": "Europe"
}}

Example if information is missing:
Text: "A tech company is looking for investment."
JSON:
{{
    "company_name": "Unknown",
    "company_location": "Unknown",
    "geography": "Unknown"
}}

Provide only the JSON object in your response.
"""

        extract_prompt = PromptTemplate(
            input_variables=["company_information"],
            template=extract_prompt_template,
        )

        # Use a less powerful/cheaper model for this simple extraction if desired, e.g. gpt-3.5-turbo
        # For consistency, using the same LLM instance here.
        extract_chain = LLMChain(llm=self.llm, prompt=extract_prompt)
        
        if self.verbose:
            logger.info(f"MarketIntelligenceAgent: Extracting company details from: {company_information[:100]}...")
        
        extracted_json_str = extract_chain.run({"company_information": company_information})

        try:
            # Clean the output if LLM adds markdown ```json ... ```
            match = re.search(r"```json\s*([\s\S]*?)\s*```", extracted_json_str, re.IGNORECASE)
            if match:
                extracted_json_str = match.group(1)
            
            details = json.loads(extracted_json_str)
            # Ensure all keys are present, defaulting to "Unknown"
            final_details = {
                "company_name": details.get("company_name", "Unknown"),
                "company_location": details.get("company_location", "Unknown"),
                "geography": details.get("geography", "Unknown"),
            }
            if self.verbose:
                logger.info(f"MarketIntelligenceAgent: Extracted company details: {final_details}")
            return final_details
        except json.JSONDecodeError:
            logger.error(f"MarketIntelligenceAgent: Failed to decode JSON from LLM for company details. Raw output: {extracted_json_str}")
            # Attempt to find fields with regex as a fallback if JSON parsing fails badly
            name_match = re.search(r'"company_name":\s*"([^"]*)"', extracted_json_str, re.IGNORECASE)
            loc_match = re.search(r'"company_location":\s*"([^"]*)"', extracted_json_str, re.IGNORECASE)
            geo_match = re.search(r'"geography":\s*"([^"]*)"', extracted_json_str, re.IGNORECASE)
            
            return {
                "company_name": name_match.group(1) if name_match else "Unknown",
                "company_location": loc_match.group(1) if loc_match else "Unknown",
                "geography": geo_match.group(1) if geo_match else "Unknown",
            }
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent: Unexpected error parsing extracted company details. Raw: {extracted_json_str}. Error: {e}")
            return {
                "company_name": "Unknown",
                "company_location": "Unknown",
                "geography": "Unknown",
            }

# if __name__ == '__main__': # Example usage
#     agent = MarketIntelligenceAgent(verbose=True)
#     # Example 1: Direct input
#     # test_company_info = "Global PetroCorp, headquartered in Houston, Texas, is a major player in the North American energy sector. They are looking to expand into South American markets."
#     # details = agent.extract_company_details(test_company_info)
#     # print("Extracted Details:", details)
#     # report = agent.run({**details, "supporting_documents": "Some initial report about their finances."})
#     # print("\nGenerated Report:\n", report)

#     # Example 2: With file (create dummy files first)
#     # with open("dummy_company_info.txt", "w") as f:
#     #     f.write("Innovate Inc. is a startup from Berlin, Germany. They focus on the European AI market.")
#     # with open("dummy_report.pdf", "w")as f: # PyPDFLoader needs a real PDF, this is just for filename
#     #     f.write("This is not a real PDF but will cause PyPDFLoader to try and fail gracefully or succeed if it's a text PDF.")


#     # Assuming you have actual files or more robust dummy files
#     # file_paths_for_market_agent = ["dummy_company_info.txt"] # Add path to a dummy PDF if you have one
#     # loaded_docs_texts = []
#     # for f_path in file_paths_for_market_agent:
#     #     if f_path.lower().endswith(".pdf"):
#     #         loaded_docs_texts.append(agent.load_pdf_from_file(f_path))
#     #     elif f_path.lower().endswith(".txt"):
#     #         loaded_docs_texts.append(agent.load_text_from_file(f_path))
#     # all_docs_text = "\n".join(loaded_docs_texts)

#     # extracted = agent.extract_company_details(agent.load_text_from_file("dummy_company_info.txt"))
#     # market_run_input = {
#     #     "company_name": extracted.get("company_name"),
#     #     "company_location": extracted.get("company_location"),
#     #     "geography": extracted.get("geography"),
#     #     "supporting_documents": all_docs_text,
#     # }
#     # market_rep = agent.run(market_run_input)
#     # print("\n--- Market Report from File ---")
#     # print(market_rep)

#     # os.remove("dummy_company_info.txt")