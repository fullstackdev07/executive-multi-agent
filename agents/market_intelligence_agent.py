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
import tempfile
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from utils.agent_output_formatter import format_agent_output

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

# Helper function to check for API input validity (more lenient)
def is_api_input_valid(text: str, field_name: str, min_length: int = 3, short_text_threshold: int = 20, min_unique_chars_for_short_text: int = 1) -> bool:
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
    def __init__(self, verbose: bool = False, max_tokens: int = 4000):  # Increased max_tokens
        self.name = "Market Intelligence Analyst"
        self.role = "a highly skilled market research analyst with access to premium online databases" #Added premium online databases access
        self.goal = "to gather comprehensive and precise information about a company and generate a detailed market intelligence report" #emphasized precision
        self.verbose = verbose
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("MarketIntelligenceAgent: OpenAI API key not found.")
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

        self.max_tokens = max_tokens  # Store max_tokens as an instance attribute
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=0.3, # Reduced temperature
            model_name="gpt-4-1106-preview",  # Use GPT-4 for better results, or specify a different model
            max_tokens=self.max_tokens,
        )

        self.prompt_template = self._create_prompt()
        self.chain = self._create_chain()
        if self.verbose:
            logger.info("MarketIntelligenceAgent initialized.")

    def _create_prompt(self) -> PromptTemplate:
        template = """You are {role}. Your goal is {goal}.

You have been provided with limited information about a company and supporting documents. Your priority is to produce ACCURATE and DETAILED information in the report, even if you need to use your own knowledge and research capabilities.

- Company Name: {company_name}
- Company Location: {company_location}
- Geography of Focus for this analysis: {geography}

--- Supporting Documents Provided ---
{supporting_documents}
--- End of Supporting Documents ---

Based on the provided information and, CRITICALLY, your own access to online databases and research tools, generate a concise and insightful market intelligence report. DO NOT rely solely on the provided documents. Actively research the company online to fill in any gaps in the provided information. If the provided location and geography are not specific, use your research to determine them.  AVOID making vague or uncertain statements.

The report should mimic the example output provided below in terms of structure and level of detail. Focus on accuracy, brevity and actionable insights.

**Example Output Format:**

**Market Research Report for: [Company Name]**

**Company Overview**
- [Company Name] is a company based in [Location], founded in [Year].
- It provides [brief description of products/services].
- The company is a [market leader/major player] in [specific industry/market].
- It serves [number] customers, including [types of customers].
- [Mention any significant recent events, like acquisitions, funding rounds].

**Market Position**
- [Company Name] is a [market leader/major player/rising competitor] in [specific market].
- It has a [strong/moderate/weak] market presence compared to its competitors.
- The company recently [describe recent actions or events affecting its market position, e.g., product launches, partnerships].

**Competitors**
- The company has [number] significant active competitors.
- Major competitors include: [List 2-3 key competitors with one-line descriptions, including their strengths/weaknesses].

**Recent Developments**
- A significant recent development for [Company Name] is [describe a key event or change].
- This [event/change] was [financed by/caused by/related to] [relevant entities or factors].

**Strategic Outlook**
- [Company Name], along with its [stakeholders], has a clear strategic vision to [describe the company's goals, based on available information].
- This is likely to be achieved through [describe key strategies, e.g., expansion into new markets, new product development].
- The company continues to focus on its core market of [describe target market, based on available information].

--- End of Example ---

Now, generate the Market Intelligence Report based on the information you have, augmented by your own online research:

Market Research Report for: {company_name}
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

    def run(self, input_data: dict) -> dict:
        if self.verbose:
            logger.info(f"MarketIntelligenceAgent: Running with input keys: {list(input_data.keys())}")
            logger.info(f"MarketIntelligenceAgent: Company Name from input_data: {input_data.get('company_name')}")
            logger.info(f"MarketIntelligenceAgent: Supporting docs (preview): {str(input_data.get('supporting_documents'))[:200]}...")

        company_name = input_data.get("company_name", "Unknown")
        company_location = input_data.get("company_location", "Unknown")
        geography = input_data.get("geography", "Unknown")
        supporting_docs = input_data.get("supporting_documents", "")
        used_files = input_data.get("used_files", []) if "used_files" in input_data else []

        # Validate company_name (Crucial)
        if company_name.lower() == "unknown" or not is_input_valid(company_name, min_length=2, short_text_threshold=10, min_unique_chars_for_short_text=1):
            logger.warning(f"MarketIntelligenceAgent: Company name '{company_name}' is invalid or 'Unknown'. Report generation might be impacted or fail.")
            if not (supporting_docs and is_input_valid(supporting_docs, min_length=50)):
                return format_agent_output(
                    title="Market Intelligence Report Error",
                    sections=[],
                    summary="Error: Company name is invalid and no supporting documents were provided. Cannot generate a meaningful report.",
                    next_steps=["Provide a valid company name or upload supporting documents."],
                    used_files=used_files,
                    meta={"agent": "Market Intelligence"}
                )

        if supporting_docs and not is_input_valid(supporting_docs, min_length=50):
            logger.warning(f"MarketIntelligenceAgent: Supporting documents appear non-meaningful or too short. Preview: {supporting_docs[:100]}...")
            supporting_docs_for_llm = "Note: Provided supporting documents were minimal or appeared non-meaningful. Use your research tools to find additional information."
        else:
            supporting_docs_for_llm = supporting_docs if supporting_docs else "No supporting documents provided. Use your research tools to find the necessary information."

        payload = {
            "role": self.role,
            "goal": self.goal,
            "company_name": company_name,
            "company_location": company_location,
            "geography": geography,
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
            return format_agent_output(
                title="Market Intelligence Report Error",
                sections=[],
                summary=f"Error: Failed to generate market report due to an internal error: {str(e)}",
                next_steps=["Try again with more detailed input or files."],
                used_files=used_files,
                meta={"agent": "Market Intelligence"}
            )

        # Split the response into sections based on markdown headers
        sections = []
        if response:
            for part in response.split("\n**"):
                if part.strip():
                    lines = part.strip().split("\n", 1)
                    header = lines[0].replace("**", "").strip()
                    content = lines[1].strip() if len(lines) > 1 else ""
                    sections.append({"header": header, "content": content})
        title = f"Market Intelligence Report for {company_name}"
        summary = f"This market intelligence report was generated for {company_name} using your input and the following files: {', '.join(used_files) if used_files else 'None'}."
        next_steps = [
            "Generate a job description for this company",
            "Create a client persona based on this market context",
            "Get feedback on this report"
        ]
        meta = {"agent": "Market Intelligence"}
        return format_agent_output(
            title=title,
            sections=sections,
            summary=summary,
            next_steps=next_steps,
            used_files=used_files,
            meta=meta
        )

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

app = FastAPI()

@app.post("/market_intelligence")
async def get_market_intelligence(
    company_information: str = Form(""),
    supporting_documents: list[UploadFile] = File([])
):
    endpoint_name = "/market_intelligence"
    # Validate company_information - crucial for this endpoint
    if company_information and not is_api_input_valid(company_information, "company_information", min_length=3, short_text_threshold=20, min_unique_chars_for_short_text=1): # Company names can be very short (e.g. "GE")
        # If supporting_documents are also empty, then definitely an issue.
        if not supporting_documents:
            raise HTTPException(status_code=400, detail="Provided 'company_information' is not meaningful, and no supporting documents were uploaded.")
        logger.warning(f"{endpoint_name}: 'company_information' seems weak, will rely heavily on documents.")
        # Allow to proceed if documents are present. Agent will handle bad company_information.

    logger.info(f"{endpoint_name}: Received company_information: {company_information[:100]}...")
    temp_file_paths, loaded_documents_content = [], []
    try:
        agent = MarketIntelligenceAgent(verbose=True)
        for doc_file in supporting_documents:
            logger.info(f"{endpoint_name}: Processing file: {doc_file.filename}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(doc_file.filename)[-1]) as tmp:
                shutil.copyfileobj(doc_file.file, tmp); tmp_path = tmp.name
            temp_file_paths.append(tmp_path)
            try:
                text_content = ""
                fname_lower = doc_file.filename.lower()
                if fname_lower.endswith(".pdf"): text_content = agent.load_pdf_from_file(tmp_path)
                elif fname_lower.endswith(".txt"): text_content = agent.load_text_from_file(tmp_path)
                elif fname_lower.endswith(".json"): text_content = agent._load_json_as_text_from_file(tmp_path)
                else: logger.warning(f"{endpoint_name}: Unsupported file: {doc_file.filename}"); continue
                if text_content and text_content.strip(): loaded_documents_content.append(text_content)
                else: logger.warning(f"{endpoint_name}: No content from {doc_file.filename}")
            except Exception as e: logger.error(f"{endpoint_name}: Error processing {doc_file.filename}: {e}")

        all_docs_text = "\n\n---DOC_SEP---\n\n".join(filter(None, loaded_documents_content))

        # Remove extract_company_details and rely on the Agent's online research
        # and internal prompting.
        # Setting 'Unknown' initially allows for agent research.
        market_input = {
            "company_name": company_information if company_information.strip() else "Unknown",
            "company_location": "Research online to determine the company location",
            "geography": "Research online to determine the company's primary geography",
            "supporting_documents": all_docs_text,
        }
        market_report = agent.run(market_input)
        if market_report.startswith("Error:"): # Check if agent itself returned an error
            raise HTTPException(status_code=400, detail=market_report) # Propagate agent error
        return {"market_report": market_report}
    except HTTPException as http_exc: raise http_exc
    except Exception as e: logger.exception(f"{endpoint_name}: Error: {e}"); raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in temp_file_paths:
            try: os.remove(path)
            except Exception as ex: logger.warning(f"{endpoint_name}: Cleanup error {path}: {ex}")