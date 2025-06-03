# import os
# from langchain_community.llms import OpenAI 
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import json
# import re
# from langchain_community.chat_models import ChatOpenAI
# import fitz 
# import logging

# load_dotenv()
# logger = logging.getLogger(__name__)

# # Helper function to check for basic input validity
# def is_input_valid(text: str, min_length: int = 10, short_text_threshold: int = 20, min_unique_chars_for_short_text: int = 3) -> bool:
#     if not text or not text.strip():
#         return False
#     stripped_text = text.strip()
#     if len(stripped_text) < min_length:
#         return False
#     if len(stripped_text) <= short_text_threshold:
#         alnum_text_part = ''.join(filter(str.isalnum, stripped_text))
#         if not alnum_text_part and len(stripped_text) > 0:
#              return False
#         elif alnum_text_part and len(set(alnum_text_part.lower())) < min_unique_chars_for_short_text:
#             return False
#     return True

# class MarketIntelligenceAgent:
#     def __init__(self, verbose: bool = False, max_tokens: int = 3000): 
#         self.name = "Market Intelligence Analyst"
#         self.role = "a highly skilled market research analyst"
#         self.goal = "to gather information about a company and generate a detailed market intelligence report"
#         self.verbose = verbose
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key:
#             logger.error("MarketIntelligenceAgent: OpenAI API key not found.")
#             raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

#         self.llm = ChatOpenAI(
#             openai_api_key=self.openai_api_key,
#             temperature=0.7,
#             model_name="gpt-3.5-turbo-16k", 
#             max_tokens=max_tokens, 
#         )

#         self.prompt_template = self._create_prompt() 
#         self.chain = self._create_chain() 
#         if self.verbose:
#             logger.info("MarketIntelligenceAgent initialized.")

#     def _create_prompt(self) -> PromptTemplate:
#         template = """You are {role}. Your goal is {goal}.

# You have been provided with information about a company and supporting documents.
# - Company Name: {company_name}
# - Company Location: {company_location}
# - Geography of Focus for this analysis: {geography}

# --- Supporting Documents Provided ---
# {supporting_documents}
# --- End of Supporting Documents ---

# Based on ALL the information above (company details, geography, and ALL supporting documents) AND your own general knowledge, generate a comprehensive market intelligence report. The report should be well-structured and detailed.

# Structure the report as follows:

# 1.  **Executive Summary:** (Approx. 200-300 words)
#     *   A concise overview of the key findings, including the company's current standing, market dynamics, and primary opportunities/threats.

# 2.  **Company Overview:** (Approx. 400-500 words)
#     *   Detailed description of "{company_name}".
#     *   Include available details on: revenue, number of employees, key office locations, main business lines/divisions, core products/services, significant historical milestones, and ownership structure (e.g., public, private, VC-backed).
#     *   Synthesize information from provided documents and your general knowledge.

# 3.  **Market Overview for {geography}:** (Approx. 400-500 words)
#     *   Analyze the specific market in which "{company_name}" operates, focusing on the specified "{geography}".
#     *   Discuss current market size (if data available or estimable), key trends (e.g., technological, regulatory, consumer behavior), significant challenges, and growth opportunities within this geography.
#     *   Reference provided documents and supplement with your general market knowledge relevant to "{geography}".

# 4.  **Competitive Landscape in {geography}:** (Approx. 300-400 words)
#     *   Identify key competitors of "{company_name}" operating within "{geography}".
#     *   Briefly profile 2-3 major competitors, outlining their strengths and weaknesses relative to "{company_name}".

# 5.  **Company Positioning and SWOT Analysis for {geography}:** (Approx. 400-500 words)
#     *   Evaluate "{company_name}"'s current positioning within the market in "{geography}".
#     *   Conduct a SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) for "{company_name}" specifically in the context of "{geography}".
#         *   **Strengths:** Internal positive attributes.
#         *   **Weaknesses:** Internal negative attributes.
#         *   **Opportunities:** External factors the company can leverage.
#         *   **Threats:** External factors that could pose a risk.

# 6.  **Conclusion and Strategic Recommendations:** (Approx. 300-400 words)
#     *   Summarize the main conclusions of the report.
#     *   Provide 2-3 actionable strategic recommendations for "{company_name}" to enhance its market position or capitalize on opportunities within "{geography}".

# Ensure the report is professional, insightful, and directly addresses all sections.

# Market Intelligence Report:
# """
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
#             logger.error("MarketIntelligenceAgent: Prompt template must be created before creating the LLMChain.")
#             raise ValueError("Prompt template must be created before creating the LLMChain.")
#         return LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=self.verbose)

#     def run(self, input_data: dict) -> str:
#         if self.verbose:
#             logger.info(f"MarketIntelligenceAgent: Running with input keys: {list(input_data.keys())}")
#             logger.info(f"MarketIntelligenceAgent: Company Name from input_data: {input_data.get('company_name')}")
#             logger.info(f"MarketIntelligenceAgent: Supporting docs (preview): {str(input_data.get('supporting_documents'))[:200]}...")

#         company_name = input_data.get("company_name", "Unknown")
#         supporting_docs = input_data.get("supporting_documents", "")

#         # Validate company_name
#         if company_name.lower() == "unknown" or not is_input_valid(company_name, min_length=2, short_text_threshold=10, min_unique_chars_for_short_text=1): # Company name can be 2 chars e.g. GE, HP
#             logger.warning(f"MarketIntelligenceAgent: Company name '{company_name}' is invalid or 'Unknown'. Report generation might be impacted or fail.")
#             # Return error if no supporting docs to potentially infer from.
#             if not (supporting_docs and is_input_valid(supporting_docs, min_length=50)):
#                  return json.dumps({
#                     "error": f"Company name '{company_name}' is invalid or 'Unknown', and no substantial supporting documents were provided. Cannot generate report."
#                 })
#             # If supporting docs exist, LLM might infer or report will be generic.

#         # Validate supporting_documents if provided
#         if supporting_docs and not is_input_valid(supporting_docs, min_length=50): # Combined docs should have some substance
#             logger.warning(f"MarketIntelligenceAgent: Supporting documents appear non-meaningful or too short. Preview: {supporting_docs[:100]}...")
#             # If company name is also weak, this is a problem.
#             if company_name.lower() == "unknown" or not is_input_valid(company_name, min_length=2, short_text_threshold=10, min_unique_chars_for_short_text=1):
#                 return json.dumps({
#                     "error": "Company name is invalid/Unknown and supporting documents are also non-meaningful. Cannot generate report."
#                 })
#             # Otherwise, proceed, but report quality might be low.
#             supporting_docs_for_llm = "Note: Provided supporting documents were minimal or appeared non-meaningful."
#         else:
#             supporting_docs_for_llm = supporting_docs if supporting_docs else "No supporting documents provided."
        
#         payload = {
#             "role": self.role,
#             "goal": self.goal,
#             "company_name": company_name, # Use the validated/original name
#             "company_location": input_data.get("company_location", "Not Specified"),
#             "geography": input_data.get("geography", "Global"), 
#             "supporting_documents": supporting_docs_for_llm,
#         }
        
#         try:
#             if hasattr(self.chain, 'invoke'):
#                 response_dict = self.chain.invoke(payload)
#                 response = response_dict.get('text')
#                 if response is None: 
#                     response = str(response_dict)
#             else: 
#                  response = self.chain.run(payload)
#         except Exception as e:
#             logger.exception(f"MarketIntelligenceAgent: Error during LLM chain execution: {e}")
#             return json.dumps({"error": f"Failed to generate market report due to an internal error: {str(e)}"})

#         if self.verbose:
#             logger.info(f"MarketIntelligenceAgent response (preview): {response[:300]}...")
#         return response

#     def load_text_from_file(self, filepath: str) -> str:
#         try:
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 text = f.read()
#             if self.verbose: logger.info(f"MarketIntelligenceAgent: Loaded text from {filepath}")
#             return text.strip() # Ensure stripping
#         except FileNotFoundError:
#             logger.warning(f"MarketIntelligenceAgent: File not found: {filepath}")
#             return ""
#         # ... (other exceptions)
#         except Exception as e:
#             logger.error(f"MarketIntelligenceAgent: General error loading text file {filepath}: {e}")
#             return ""


#     def load_pdf_from_file(self, filepath: str) -> str:
#         text = ""
#         try:
#             with fitz.open(filepath) as doc:
#                 for page in doc:
#                     text += page.get_text("text") 
#             if self.verbose: logger.info(f"MarketIntelligenceAgent: Loaded PDF (via fitz) from {filepath}, length: {len(text)}")
#             return text.strip()
#         # ... (exceptions)
#         except Exception as e: 
#             logger.error(f"MarketIntelligenceAgent: Error loading PDF {filepath} with fitz: {e}")
#             return f"Warning: Error loading PDF content from {os.path.basename(filepath)} using fitz."


#     def _load_json_as_text_from_file(self, filepath: str) -> str:
#         try:
#             with open(filepath, "r", encoding="utf-8") as f:
#                 json_data = json.load(f)
#             text_content = json.dumps(json_data, indent=2)
#             if self.verbose: logger.info(f"MarketIntelligenceAgent: Loaded JSON from {filepath}, length: {len(text_content)}")
#             return text_content.strip() # Ensure stripping
#         # ... (exceptions)
#         except Exception as e:
#             logger.error(f"MarketIntelligenceAgent: Error loading JSON file {filepath}: {e}")
#             return f"Warning: Error loading JSON file {os.path.basename(filepath)}."

#     def save_report_to_file(self, report: str, filepath: str): 
#         # ... (no changes needed here for gibberish input)
#         try:
#             with open(filepath, 'w', encoding='utf-8') as f:
#                 f.write(report)
#             logger.info(f"MarketIntelligenceAgent: Report saved to {filepath}")
#         except Exception as e:
#             logger.error(f"MarketIntelligenceAgent: Error saving report to {filepath}: {e}")


#     def extract_company_details(self, company_information: str) -> dict:
#         # Use more lenient min_length for company info as it can be short phrases.
#         if not is_input_valid(company_information, min_length=5, short_text_threshold=30, min_unique_chars_for_short_text=2):
#             logger.warning(f"MarketIntelligenceAgent: company_information for extraction appears non-meaningful: {company_information[:100]}...")
#             return { "company_name": "Unknown", "company_location": "Unknown", "geography": "Unknown" }

#         extract_prompt_template = """You are an expert at extracting specific company information from unstructured text.
# Given the following text:
# --- TEXT START ---
# {company_information}
# --- TEXT END ---
# Your task is to extract the following three pieces of information:
# 1.  **Company Name:** The primary legal or commonly known name of the company.
# 2.  **Company Location:** The city and country of the company's headquarters or main office mentioned. If multiple locations, pick the most prominent or first mentioned as HQ.
# 3.  **Geography of Focus:** The primary geographical market or region the company operates in or is targeting, as suggested by the text. This might be a country, a continent, or terms like "global", "EMEA", "North America".
# Return the extracted information strictly as a JSON object with the keys: "company_name", "company_location", "geography".
# If any piece of information cannot be reliably determined from the text, use the string "Unknown" as its value.
# Provide only the JSON object in your response.
# """ # Simplified prompt for brevity, actual prompt is longer.

#         extract_prompt = PromptTemplate(
#             input_variables=["company_information"],
#             template=extract_prompt_template,
#         )
        
#         extract_chain = LLMChain(llm=self.llm, prompt=extract_prompt) 
        
#         if self.verbose:
#             logger.info(f"MarketIntelligenceAgent: Extracting company details from: {company_information[:100]}...")
        
#         if hasattr(extract_chain, 'invoke'):
#             extracted_response = extract_chain.invoke({"company_information": company_information})
#             extracted_json_str = extracted_response.get('text', str(extracted_response))
#         else:
#             extracted_json_str = extract_chain.run({"company_information": company_information})

#         try:
#             match = re.search(r"```json\s*([\s\S]*?)\s*```", extracted_json_str, re.IGNORECASE)
#             if match:
#                 json_str_to_parse = match.group(1).strip()
#             else:
#                 curly_match = re.search(r"(\{[\s\S]*\})", extracted_json_str)
#                 if curly_match:
#                     json_str_to_parse = curly_match.group(1).strip()
#                 else:
#                     json_str_to_parse = extracted_json_str.strip()
            
#             details = json.loads(json_str_to_parse)
#             final_details = {
#                 "company_name": details.get("company_name", "Unknown"),
#                 "company_location": details.get("company_location", "Unknown"),
#                 "geography": details.get("geography", "Unknown"),
#             }
#             for key in final_details: # Ensure "Unknown" if empty or not valid
#                 if not final_details[key] or not isinstance(final_details[key], str) or \
#                    (final_details[key].lower() != "unknown" and not is_input_valid(final_details[key], min_length=2, short_text_threshold=10, min_unique_chars_for_short_text=1)):
#                     final_details[key] = "Unknown"
#             if self.verbose:
#                 logger.info(f"MarketIntelligenceAgent: Extracted company details (parsed JSON): {final_details}")
#             return final_details
#         except json.JSONDecodeError:
#             logger.error(f"MarketIntelligenceAgent: Failed to decode JSON from LLM for company details. Raw output: '{extracted_json_str}'")
#             # Basic regex fallback
#             name_match = re.search(r'"company_name":\s*"([^"]*)"', extracted_json_str, re.IGNORECASE)
#             loc_match = re.search(r'"company_location":\s*"([^"]*)"', extracted_json_str, re.IGNORECASE)
#             geo_match = re.search(r'"geography":\s*"([^"]*)"', extracted_json_str, re.IGNORECASE)
            
#             return {
#                 "company_name": name_match.group(1) if name_match and name_match.group(1) else "Unknown",
#                 "company_location": loc_match.group(1) if loc_match and loc_match.group(1) else "Unknown",
#                 "geography": geo_match.group(1) if geo_match and geo_match.group(1) else "Unknown",
#             }
#         except Exception as e: 
#             logger.error(f"MarketIntelligenceAgent: Unexpected error parsing extracted company details. Raw: '{extracted_json_str}'. Error: {e}")
#             return {"company_name": "Unknown", "company_location": "Unknown", "geography": "Unknown"}

import os
# from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import re
from langchain_community.chat_models import ChatOpenAI
import fitz # PyMuPDF
import logging

load_dotenv()
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def is_input_valid(text: str, min_length: int = 1, short_text_threshold: int = 5, min_unique_chars_for_short_text: int = 1) -> bool:
    if not text or not isinstance(text, str) or not text.strip():
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
    def __init__(self, verbose: bool = False, max_tokens: int = 4000): # Keep max_tokens generous
        self.name = "Market Intelligence Analyst"
        self.role = "a highly skilled market research and company analysis expert with deep general knowledge" # Emphasize knowledge
        self.goal = "to gather information about a company and generate a detailed, well-researched company profile and market intelligence report, filling in details from its knowledge base where specific inputs are lacking."
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("MarketIntelligenceAgent: OpenAI API key not found.")
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

        # LLM for the main report generation - slightly higher temperature
        self.report_llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0.65, # Slightly increased for more inferential/descriptive filling
            model_name="gpt-3.5-turbo-0125", # Or "gpt-4-turbo-preview" / "gpt-4o" if available
            max_tokens=max_tokens,
        )

        # LLM for the finder and extractor - keep temperature low for factual retrieval
        self.utility_llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0.2,
            model_name="gpt-3.5-turbo-0125",
            max_tokens=1000 # Typically shorter responses needed
        )


        finder_template_str = """You are an AI assistant specialized in company information retrieval.
Based on your general knowledge, find the following for the company: "{company_name}".
1.  **Headquarters Location:** (City, Country. If not found or ambiguous, return "Unknown")
2.  **Primary Geography of Operation/Focus:** (e.g., USA, Europe, Global, specific countries. If not clear, return "Unknown")
Return this information strictly as a JSON object with keys "found_location" and "found_geography".
If specific information for a field cannot be reliably determined, use "Unknown" as its value for that field.
Provide only the JSON object.
"""
        self.finder_prompt = PromptTemplate(input_variables=["company_name"], template=finder_template_str)
        self.finder_chain = LLMChain(llm=self.utility_llm, prompt=self.finder_prompt, verbose=self.verbose)

        self.report_prompt_template = self._create_report_prompt()
        self.report_chain = LLMChain(llm=self.report_llm, prompt=self.report_prompt_template, verbose=self.verbose)

        if self.verbose:
            logger.info(f"{self.name} initialized. Goal: {self.goal}")

    def _create_report_prompt(self) -> PromptTemplate:
        template = """You are {role}. Your goal is {goal}.
You are tasked with generating a detailed and insightful report about "{company_name}".
Leverage all provided information AND your extensive general knowledge base to make this report as complete and informative as possible.
If specific data points (like exact revenue or founding year) are not in the provided documents or readily available public knowledge for this specific company, you should:
1.  State that precise figures are not publicly available.
2.  THEN, provide well-reasoned estimations, qualitative descriptions, or industry-typical information. For example, instead of "[X-Y] million", you might say "As a private company in the [industry] sector, its revenue is likely substantial, though specific figures are not disclosed. Companies of this nature in [geography] typically see revenues in the range of..."
3.  For history, if the exact founding year is unknown, describe the likely period of its establishment based on its industry and offerings, or focus on known developments.
4.  **Do NOT leave placeholders like "[year]", "[details]", "[X-Y]". Fill them with your best inferential knowledge or qualitative descriptions.** Your primary goal is a readable, informative report, not a template with blanks.

Company Details Provided for Analysis:
- Company Name: {company_name}
- Company Location (if specified by user or found): {company_location}
- Primary Geography for this Analysis: {geography}

Supporting Documents Synopsis (if provided):
{supporting_documents}

User's Specific Areas of Interest (if any):
{areas_of_interest}
--- End of Input Data ---

--- Important Instructions for Report Generation ---
- If "Company Location" is "Not Specified", state this fact in "Facts & Figures". Conduct analysis based on "{geography}" and your general knowledge of the company's operational spread.
- If "Primary Geography for this Analysis" is "Global", ensure "Market and Competitive Environment" and "Challenges" adopt a global perspective. If a specific geography is provided, focus analysis there.
- If "Company Name" is generic (e.g., "A Company (Name Not Specified)"), clearly state this limitation. Focus on the general industry implied by documents/interests, or state that detailed company-specific analysis is limited.
- **Avoid literal "Unknown" placeholders.** Adapt language for general, global, or "not specified" information.
- If "User's Specific Areas of Interest" are provided and not "None provided", thoughtfully weave insights into relevant sections.
--- End of Important Instructions ---

Generate a comprehensive company profile and market intelligence report structured as follows:

**1. History:** (Approx. 200-300 words)
    *   Describe the founding context (e.g., "likely founded in the early 2010s to address the growing need for..."), significant historical milestones, and key evolutionary phases of "{company_name}". Use your knowledge to elaborate.

**2. Facts & Figures:** (Approx. 200-300 words)
    *   Provide available data or reasoned estimations/qualitative descriptions for:
        *   Revenue: (If exact figures are unknown, describe its likely scale, e.g., "While specific revenue figures for the private company {company_name} are not publicly disclosed, it is positioned as a [small/medium/large] player in the [industry] market. Similar companies in this space generate revenues in the range of...")
        *   Number of Employees: (e.g., "Estimated to have between X and Y employees, typical for a company of its type and market focus.")
        *   Key Office Locations / Global Presence: (If "{company_location}" is specific, state it as the headquarters. If "Not Specified", describe the company's general known global or regional presence, e.g., "While a specific headquarters for this analysis was not determined, {company_name} likely serves customers primarily in {geography} and may have a distributed team or key operational hubs in major cities within this region.")
        *   Ownership Structure: (e.g., "Believed to be a privately held company.", "Operates as a subsidiary of... if known.", "Publicly traded under ticker...") Use your knowledge.

**3. Business Units & Offerings:** (Approx. 300-400 words)
    *   Describe the main business lines, divisions, or segments of "{company_name}".
    *   Detail its core products, services, or solutions. Be specific based on your knowledge of such companies.

**4. Market and Competitive Environment (focused on {geography}):** (Approx. 400-500 words)
    *   Analyze the primary market(s) "{company_name}" operates in, with a focus on "{geography}".
    *   Discuss key market trends (e.g., technological, regulatory, consumer behavior, economic factors), elaborating with your knowledge.
    *   Identify 2-3 major competitors to "{company_name}" operating within "{geography}". Briefly outline their main offerings or strengths in comparison to "{company_name}". Use your knowledge to identify relevant competitors.

**5. Challenges (within {geography}):** (Approx. 200-300 words)
    *   Identify and discuss 2-3 significant challenges "{company_name}" likely faces in its market(s) within "{geography}". These could be market-specific, competitive, technological, regulatory, or internal. Elaborate based on your understanding of the industry and geography.

Ensure the report is professional, insightful, well-structured, and addresses all sections by actively using your knowledge base to fill in details and provide context.

Company Profile and Market Intelligence Report for {company_name}:
"""
        return PromptTemplate(
            input_variables=[
                "role", "goal", "company_name", "company_location",
                "geography", "supporting_documents", "areas_of_interest"
            ],
            template=template,
        )

    def _find_missing_company_info_with_llm(self, company_name: str) -> dict:
        # This method remains the same.
        if not company_name or not is_input_valid(company_name, min_length=2) \
           or company_name.lower() == "unknown" or company_name == "A Company (Name Not Specified)":
            if self.verbose: logger.info(f"MarketIntelligenceAgent (Finder): Skipping LLM find for generic/invalid company name: '{company_name}'")
            return {"found_location": "Unknown", "found_geography": "Unknown"}

        payload = {"company_name": company_name.strip()}
        extracted_json_str = ""
        try:
            if self.verbose: logger.info(f"MarketIntelligenceAgent (Finder): Querying LLM for location/geography of '{company_name}'")
            
            if hasattr(self.finder_chain, 'invoke'):
                response_dict = self.finder_chain.invoke(payload)
                extracted_json_str = response_dict.get('text', str(response_dict))
            else:
                extracted_json_str = self.finder_chain.run(payload)

            match_json_block = re.search(r"```json\s*([\s\S]*?)\s*```", extracted_json_str, re.IGNORECASE)
            if match_json_block:
                json_str_to_parse = match_json_block.group(1).strip()
            else:
                try:
                    start_index = extracted_json_str.index("{")
                    end_index = extracted_json_str.rindex("}") + 1
                    json_str_to_parse = extracted_json_str[start_index:end_index].strip()
                except ValueError:
                    logger.error(f"MarketIntelligenceAgent (Finder): Could not find JSON object markers in LLM response: '{extracted_json_str}'")
                    return {"found_location": "Unknown", "found_geography": "Unknown"}
            
            details = json.loads(json_str_to_parse)
            found_loc = details.get("found_location", "Unknown")
            found_geo = details.get("found_geography", "Unknown")

            if not isinstance(found_loc, str): found_loc = "Unknown"
            if not isinstance(found_geo, str): found_geo = "Unknown"

            if self.verbose: logger.info(f"MarketIntelligenceAgent (Finder): LLM returned - Location: '{found_loc}', Geography: '{found_geo}' for '{company_name}'")
            return {"found_location": found_loc.strip(), "found_geography": found_geo.strip()}
        except json.JSONDecodeError:
            logger.error(f"MarketIntelligenceAgent (Finder): Failed to decode JSON. Raw output: '{extracted_json_str}'")
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent (Finder): Error during LLM execution or parsing: {e}. Raw output: '{extracted_json_str}'")
        return {"found_location": "Unknown", "found_geography": "Unknown"}

    def run(self, input_data: dict) -> str:
        # This method's logic for determining company_name, location, geography, docs, areas_of_interest
        # remains largely the same as the previous version, as it correctly prepares these for the payload.
        # The key change is the enhanced prompt and potentially the report_llm temperature.

        if self.verbose: logger.info(f"MarketIntelligenceAgent: Received run request with input_data keys: {list(input_data.keys())}")

        company_name_from_api = input_data.get("company_name", "A Company (Name Not Specified)")
        if not company_name_from_api or not is_input_valid(company_name_from_api, min_length=2):
            company_name_for_payload = "A Company (Name Not Specified)"
            logger.warning(f"MarketIntelligenceAgent: Company name from API ('{company_name_from_api}') is invalid or missing, defaulted to '{company_name_for_payload}'.")
        else:
            company_name_for_payload = company_name_from_api.strip()

        supporting_docs_text_from_api = input_data.get("supporting_documents", "")
        if supporting_docs_text_from_api and isinstance(supporting_docs_text_from_api, str) and \
           is_input_valid(supporting_docs_text_from_api, min_length=50, short_text_threshold=100, min_unique_chars_for_short_text=10):
            supporting_docs_for_llm = supporting_docs_text_from_api.strip()
        elif supporting_docs_text_from_api:
            supporting_docs_for_llm = "Note: Provided supporting documents were minimal, appeared non-meaningful, or could not be fully processed."
            logger.warning(f"MarketIntelligenceAgent: Supporting documents text from API seems weak. Using note for LLM.")
        else:
            supporting_docs_for_llm = "No supporting documents provided."

        if company_name_for_payload == "A Company (Name Not Specified)" and supporting_docs_for_llm == "No supporting documents provided.":
            logger.error("MarketIntelligenceAgent: Cannot generate report. Company name is generic and no supporting documents were effectively provided.")
            return "Error: Cannot generate report. Company name is generic/unknown and no substantial supporting documents were provided."

        raw_location_from_api = input_data.get("company_location", "Unknown")
        raw_geography_from_api = input_data.get("geography", "Unknown")
        
        areas_of_interest_input = input_data.get("areas_of_interest", "")
        areas_of_interest_for_llm = areas_of_interest_input.strip() if areas_of_interest_input and isinstance(areas_of_interest_input, str) and areas_of_interest_input.strip() else "None provided."
        if areas_of_interest_for_llm != "None provided." and self.verbose:
            logger.info(f"MarketIntelligenceAgent: Areas of interest provided: '{areas_of_interest_for_llm}'")

        final_location = "Not Specified"
        final_geography = "Global"

        if raw_location_from_api and isinstance(raw_location_from_api, str) and \
           raw_location_from_api.strip().lower() != "unknown" and \
           is_input_valid(raw_location_from_api.strip(), min_length=2):
            final_location = raw_location_from_api.strip()
        
        if raw_geography_from_api and isinstance(raw_geography_from_api, str) and \
           raw_geography_from_api.strip().lower() != "unknown" and \
           is_input_valid(raw_geography_from_api.strip(), min_length=2):
            final_geography = raw_geography_from_api.strip()

        if company_name_for_payload != "A Company (Name Not Specified)":
            needs_location_search = (final_location == "Not Specified")
            needs_geography_search = (final_geography == "Global" and (raw_geography_from_api is None or raw_geography_from_api.strip().lower() == "unknown" or raw_geography_from_api.strip().lower() == "global")) or \
                                     (raw_geography_from_api and raw_geography_from_api.strip().lower() == "unknown")

            if needs_location_search or needs_geography_search:
                if self.verbose: logger.info(f"MarketIntelligenceAgent: Attempting to find/confirm details for '{company_name_for_payload}' via internal LLM knowledge.")
                found_details = self._find_missing_company_info_with_llm(company_name_for_payload)

                if needs_location_search and found_details.get("found_location","Unknown").lower() != "unknown":
                    if is_input_valid(found_details["found_location"], min_length=2):
                        final_location = found_details["found_location"]
                        if self.verbose: logger.info(f"MarketIntelligenceAgent: Using LLM-found location: '{final_location}'")
                
                if needs_geography_search and found_details.get("found_geography","Unknown").lower() != "unknown":
                    if is_input_valid(found_details["found_geography"], min_length=2):
                        final_geography = found_details["found_geography"]
                        if self.verbose: logger.info(f"MarketIntelligenceAgent: Using LLM-found geography: '{final_geography}'")
        
        if final_location.lower() == "unknown" or not is_input_valid(final_location): final_location = "Not Specified"
        if final_geography.lower() == "unknown" or not is_input_valid(final_geography): final_geography = "Global"

        payload = {
            "role": self.role,
            "goal": self.goal,
            "company_name": company_name_for_payload,
            "company_location": final_location,
            "geography": final_geography,
            "supporting_documents": supporting_docs_for_llm,
            "areas_of_interest": areas_of_interest_for_llm,
        }

        if self.verbose:
            logger.info(f"MarketIntelligenceAgent: Final payload for LLM report generation: {payload}")

        try:
            if hasattr(self.report_chain, 'invoke'):
                response_dict = self.report_chain.invoke(payload)
                response_text = response_dict.get('text')
                if response_text is None: response_text = str(response_dict)
            else:
                 response_text = self.report_chain.run(payload)
            
            final_report = response_text.strip()
            if company_name_for_payload == "A Company (Name Not Specified)" and "A Company (Name Not Specified)" not in final_report[:500]:
                final_report = f"Note: This report addresses a company for which a specific name was not provided or determined.\n\n{final_report}"

        except Exception as e:
            logger.exception(f"MarketIntelligenceAgent: Error during LLM report chain execution: {e}")
            return f"Error: Failed to generate report due to an internal LLM or processing error: {str(e)}"

        if self.verbose: logger.info(f"MarketIntelligenceAgent: Report generated successfully (preview: {final_report[:400]}...)")
        return final_report

    # --- File Loading Methods & Extractor (remain the same as previous good version) ---
    def load_text_from_file(self, filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            if self.verbose: logger.info(f"MarketIntelligenceAgent: Loaded text from {filepath}")
            return text.strip()
        except FileNotFoundError:
            logger.warning(f"MarketIntelligenceAgent: Text file not found: {filepath}")
            return ""
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent: Error loading text file {filepath}: {e}")
            return f"Warning: Error loading text file {os.path.basename(filepath)}."

    def load_pdf_from_file(self, filepath: str) -> str:
        text = ""
        try:
            with fitz.open(filepath) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text("text")
            if self.verbose: logger.info(f"MarketIntelligenceAgent: Loaded PDF content from {filepath}, length: {len(text)}")
            return text.strip()
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent: Error loading PDF {filepath} with fitz: {e}")
            return f"Warning: Error loading PDF content from {os.path.basename(filepath)}. Content may be missing or incomplete."

    def _load_json_as_text_from_file(self, filepath: str) -> str:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            text_content = json.dumps(json_data, indent=2)
            if self.verbose: logger.info(f"MarketIntelligenceAgent: Loaded JSON as text from {filepath}, length: {len(text_content)}")
            return text_content.strip()
        except FileNotFoundError:
            logger.warning(f"MarketIntelligenceAgent: JSON file not found: {filepath}")
            return ""
        except json.JSONDecodeError:
            logger.error(f"MarketIntelligenceAgent: Error decoding JSON file {filepath}")
            return f"Warning: Error decoding JSON file {os.path.basename(filepath)}. It may be malformed."
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent: Error loading JSON file {filepath}: {e}")
            return f"Warning: Error loading JSON file {os.path.basename(filepath)}."

    def save_report_to_file(self, report: str, filepath: str):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"MarketIntelligenceAgent: Report successfully saved to {filepath}")
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent: Error saving report to {filepath}: {e}")

    def extract_company_details(self, company_information_text: str) -> dict:
        if not company_information_text or not isinstance(company_information_text, str) or len(company_information_text.strip()) < 3:
            logger.warning(f"MarketIntelligenceAgent (Extractor): Input text for extraction ('{company_information_text[:50]}...') is too short or invalid.")
            return {"company_name": "Unknown", "company_location": "Unknown", "geography": "Unknown"}

        extract_prompt_template_str = """You are an expert at extracting specific company information from unstructured text.
Given the following text:
--- TEXT START ---
{company_information}
--- TEXT END ---
Your task is to extract:
1.  **Company Name:** The primary legal or commonly known name of the company. If multiple names seem present, choose the most complete or official one. If no clear company name, return "Unknown".
2.  **Company Location:** The city and country of the company's headquarters or main office if explicitly mentioned. If multiple locations, pick the most prominent or first mentioned as HQ. If none, return "Unknown".
3.  **Geography of Focus:** The primary geographical market or region the company operates in or is targeting, as suggested by the text (e.g., a specific country, a continent like Europe, or terms like "global", "EMEA", "North America"). If none, return "Unknown".
Return this information strictly as a JSON object with keys: "company_name", "company_location", "geography".
Provide only the JSON object in your response.
"""
        extract_prompt = PromptTemplate(input_variables=["company_information"], template=extract_prompt_template_str)
        # Use utility_llm for extraction
        extract_chain = LLMChain(llm=self.utility_llm, prompt=extract_prompt, verbose=self.verbose)


        if self.verbose: logger.info(f"MarketIntelligenceAgent (Extractor): Attempting to extract company details from text (preview): {company_information_text[:150]}...")
        
        extracted_json_str = ""
        try:
            if hasattr(extract_chain, 'invoke'):
                response_dict = extract_chain.invoke({"company_information": company_information_text})
                extracted_json_str = response_dict.get('text', str(response_dict))
            else:
                extracted_json_str = extract_chain.run({"company_information": company_information_text})
            
            match_json_block = re.search(r"```json\s*([\s\S]*?)\s*```", extracted_json_str, re.IGNORECASE)
            if match_json_block:
                json_str_to_parse = match_json_block.group(1).strip()
            else:
                try:
                    start_index = extracted_json_str.index("{")
                    end_index = extracted_json_str.rindex("}") + 1
                    json_str_to_parse = extracted_json_str[start_index:end_index].strip()
                except ValueError:
                    logger.error(f"MarketIntelligenceAgent (Extractor): Could not find JSON object markers in LLM response: '{extracted_json_str}'")
                    return {"company_name": "Unknown", "company_location": "Unknown", "geography": "Unknown"}

            details = json.loads(json_str_to_parse)
            extracted_data = {
                "company_name": details.get("company_name", "Unknown"),
                "company_location": details.get("company_location", "Unknown"),
                "geography": details.get("geography", "Unknown"),
            }
            for key, value in extracted_data.items():
                if not isinstance(value, str): 
                    extracted_data[key] = "Unknown"
                else:
                    stripped_value = value.strip()
                    if stripped_value.lower() != "unknown" and not is_input_valid(stripped_value, min_length=2):
                        if self.verbose: logger.info(f"Extractor: Invalid value '{stripped_value}' for {key}, treating as 'Unknown'.")
                        extracted_data[key] = "Unknown"
                    else:
                        extracted_data[key] = stripped_value
            
            if self.verbose: logger.info(f"MarketIntelligenceAgent (Extractor): Successfully extracted details: {extracted_data}")
            return extracted_data
        except json.JSONDecodeError:
            logger.error(f"MarketIntelligenceAgent (Extractor): Failed to decode JSON from LLM. Raw output: '{extracted_json_str}'")
        except Exception as e:
            logger.error(f"MarketIntelligenceAgent (Extractor): Unexpected error during extraction: {e}. Raw output: '{extracted_json_str}'")
        return {"company_name": "Unknown", "company_location": "Unknown", "geography": "Unknown"}