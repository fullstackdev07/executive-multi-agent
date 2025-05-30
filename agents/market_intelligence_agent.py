import os
from langchain_community.llms import OpenAI 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import re
from langchain_community.chat_models import ChatOpenAI
import fitz 
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Helper function to check for basic input validity
def is_input_valid(text: str, min_length: int = 10, short_text_threshold: int = 20, min_unique_chars_for_short_text: int = 3) -> bool:
    if not text or not text.strip():
        return False
    stripped_text = text.strip()
    if len(stripped_text) < min_length:
        return False
    if len(stripped_text) <= short_text_threshold:
        alnum_text_part = ''.join(filter(str.isalnum, stripped_text))
        if not alnum_text_part and len(stripped_text) > 0:
             return False
        elif alnum_text_part and len(set(alnum_text_part.lower())) < min_unique_chars_for_short_text:
            return False
    return True

class MarketIntelligenceAgent:
    def __init__(self, verbose: bool = False, max_tokens: int = 3000): 
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
            model_name="gpt-3.5-turbo-16k", 
            max_tokens=max_tokens, 
        )

        self.prompt_template = self._create_prompt() 
        self.chain = self._create_chain() 
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
        if self.verbose:
            logger.info(f"MarketIntelligenceAgent: Running with input keys: {list(input_data.keys())}")
            logger.info(f"MarketIntelligenceAgent: Company Name from input_data: {input_data.get('company_name')}")
            logger.info(f"MarketIntelligenceAgent: Supporting docs (preview): {str(input_data.get('supporting_documents'))[:200]}...")

        company_name = input_data.get("company_name", "Unknown")
        supporting_docs = input_data.get("supporting_documents", "")

        # Validate company_name
        if company_name.lower() == "unknown" or not is_input_valid(company_name, min_length=2, short_text_threshold=10, min_unique_chars_for_short_text=1): # Company name can be 2 chars e.g. GE, HP
            logger.warning(f"MarketIntelligenceAgent: Company name '{company_name}' is invalid or 'Unknown'. Report generation might be impacted or fail.")
            # Return error if no supporting docs to potentially infer from.
            if not (supporting_docs and is_input_valid(supporting_docs, min_length=50)):
                 return json.dumps({
                    "error": f"Company name '{company_name}' is invalid or 'Unknown', and no substantial supporting documents were provided. Cannot generate report."
                })
            # If supporting docs exist, LLM might infer or report will be generic.

        # Validate supporting_documents if provided
        if supporting_docs and not is_input_valid(supporting_docs, min_length=50): # Combined docs should have some substance
            logger.warning(f"MarketIntelligenceAgent: Supporting documents appear non-meaningful or too short. Preview: {supporting_docs[:100]}...")
            # If company name is also weak, this is a problem.
            if company_name.lower() == "unknown" or not is_input_valid(company_name, min_length=2, short_text_threshold=10, min_unique_chars_for_short_text=1):
                return json.dumps({
                    "error": "Company name is invalid/Unknown and supporting documents are also non-meaningful. Cannot generate report."
                })
            # Otherwise, proceed, but report quality might be low.
            supporting_docs_for_llm = "Note: Provided supporting documents were minimal or appeared non-meaningful."
        else:
            supporting_docs_for_llm = supporting_docs if supporting_docs else "No supporting documents provided."
        
        payload = {
            "role": self.role,
            "goal": self.goal,
            "company_name": company_name, # Use the validated/original name
            "company_location": input_data.get("company_location", "Not Specified"),
            "geography": input_data.get("geography", "Global"), 
            "supporting_documents": supporting_docs_for_llm,
        }
        
        try:
            if hasattr(self.chain, 'invoke'):
                response_dict = self.chain.invoke(payload)
                response = response_dict.get('text')
                if response is None: 
                    response = str(response_dict)
            else: 
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
            return text.strip() # Ensure stripping
        except FileNotFoundError:
            logger.warning(f"MarketIntelligenceAgent: File not found: {filepath}")
            return ""
        # ... (other exceptions)
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent: General error loading text file {filepath}: {e}")
            return ""


    def load_pdf_from_file(self, filepath: str) -> str:
        text = ""
        try:
            with fitz.open(filepath) as doc:
                for page in doc:
                    text += page.get_text("text") 
            if self.verbose: logger.info(f"MarketIntelligenceAgent: Loaded PDF (via fitz) from {filepath}, length: {len(text)}")
            return text.strip()
        # ... (exceptions)
        except Exception as e: 
            logger.error(f"MarketIntelligenceAgent: Error loading PDF {filepath} with fitz: {e}")
            return f"Warning: Error loading PDF content from {os.path.basename(filepath)} using fitz."


    def _load_json_as_text_from_file(self, filepath: str) -> str:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            text_content = json.dumps(json_data, indent=2)
            if self.verbose: logger.info(f"MarketIntelligenceAgent: Loaded JSON from {filepath}, length: {len(text_content)}")
            return text_content.strip() # Ensure stripping
        # ... (exceptions)
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent: Error loading JSON file {filepath}: {e}")
            return f"Warning: Error loading JSON file {os.path.basename(filepath)}."

    def save_report_to_file(self, report: str, filepath: str): 
        # ... (no changes needed here for gibberish input)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"MarketIntelligenceAgent: Report saved to {filepath}")
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent: Error saving report to {filepath}: {e}")


    def extract_company_details(self, company_information: str) -> dict:
        # Use more lenient min_length for company info as it can be short phrases.
        if not is_input_valid(company_information, min_length=5, short_text_threshold=30, min_unique_chars_for_short_text=2):
            logger.warning(f"MarketIntelligenceAgent: company_information for extraction appears non-meaningful: {company_information[:100]}...")
            return { "company_name": "Unknown", "company_location": "Unknown", "geography": "Unknown" }

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
Provide only the JSON object in your response.
""" # Simplified prompt for brevity, actual prompt is longer.

        extract_prompt = PromptTemplate(
            input_variables=["company_information"],
            template=extract_prompt_template,
        )
        
        extract_chain = LLMChain(llm=self.llm, prompt=extract_prompt) 
        
        if self.verbose:
            logger.info(f"MarketIntelligenceAgent: Extracting company details from: {company_information[:100]}...")
        
        if hasattr(extract_chain, 'invoke'):
            extracted_response = extract_chain.invoke({"company_information": company_information})
            extracted_json_str = extracted_response.get('text', str(extracted_response))
        else:
            extracted_json_str = extract_chain.run({"company_information": company_information})

        try:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", extracted_json_str, re.IGNORECASE)
            if match:
                json_str_to_parse = match.group(1).strip()
            else:
                curly_match = re.search(r"(\{[\s\S]*\})", extracted_json_str)
                if curly_match:
                    json_str_to_parse = curly_match.group(1).strip()
                else:
                    json_str_to_parse = extracted_json_str.strip()
            
            details = json.loads(json_str_to_parse)
            final_details = {
                "company_name": details.get("company_name", "Unknown"),
                "company_location": details.get("company_location", "Unknown"),
                "geography": details.get("geography", "Unknown"),
            }
            for key in final_details: # Ensure "Unknown" if empty or not valid
                if not final_details[key] or not isinstance(final_details[key], str) or \
                   (final_details[key].lower() != "unknown" and not is_input_valid(final_details[key], min_length=2, short_text_threshold=10, min_unique_chars_for_short_text=1)):
                    final_details[key] = "Unknown"
            if self.verbose:
                logger.info(f"MarketIntelligenceAgent: Extracted company details (parsed JSON): {final_details}")
            return final_details
        except json.JSONDecodeError:
            logger.error(f"MarketIntelligenceAgent: Failed to decode JSON from LLM for company details. Raw output: '{extracted_json_str}'")
            # Basic regex fallback
            name_match = re.search(r'"company_name":\s*"([^"]*)"', extracted_json_str, re.IGNORECASE)
            loc_match = re.search(r'"company_location":\s*"([^"]*)"', extracted_json_str, re.IGNORECASE)
            geo_match = re.search(r'"geography":\s*"([^"]*)"', extracted_json_str, re.IGNORECASE)
            
            return {
                "company_name": name_match.group(1) if name_match and name_match.group(1) else "Unknown",
                "company_location": loc_match.group(1) if loc_match and loc_match.group(1) else "Unknown",
                "geography": geo_match.group(1) if geo_match and geo_match.group(1) else "Unknown",
            }
        except Exception as e: 
            logger.error(f"MarketIntelligenceAgent: Unexpected error parsing extracted company details. Raw: '{extracted_json_str}'. Error: {e}")
            return {"company_name": "Unknown", "company_location": "Unknown", "geography": "Unknown"}