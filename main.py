# from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter
# import logging, os, tempfile, json, io, shutil, uuid, fitz
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Optional, List
# from agents.client_representative_agent import ClientRepresentativeAgent
# from agents.client_representative_creator_agent import ClientRepresentativeCreatorAgent
# from agents.interview_report_creator_agent import InterviewReportCreatorAgent
# from agents.job_description_writer_agent import JobDescriptionWriterAgent
# from agents.market_intelligence_agent import MarketIntelligenceAgent
# from orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# app = FastAPI()
 
# origins = [
#     "*", 
#     "http://localhost:3000", "http://127.0.0.1:3000",
#     "https://executive-multi-agent-frontend.vercel.app", 
# ]

# app.add_middleware(
#     CORSMiddleware, allow_origins=origins, allow_credentials=True,
#     allow_methods=["*"], allow_headers=["*"],
# )

# # Helper function to check for basic input validity (can be in a utils.py)
# def is_api_input_valid(text: Optional[str], field_name: str, min_length: int = 10, short_text_threshold: int = 20, min_unique_chars_for_short_text: int = 3) -> bool:
#     if text is None: # Allow optional fields to be None
#         return True
#     if not text.strip(): # If provided but empty string
#         # Depending on whether an empty string is acceptable for this optional field
#         logger.warning(f"API Validation: Optional field '{field_name}' is an empty string.")
#         return True # Or False if empty string is not allowed even if optional
        
#     stripped_text = text.strip()
#     if len(stripped_text) < min_length:
#         logger.warning(f"API Validation: Field '{field_name}' ('{stripped_text[:30]}...') is too short (min_length: {min_length}).")
#         return False
#     if len(stripped_text) <= short_text_threshold:
#         alnum_text_part = ''.join(filter(str.isalnum, stripped_text))
#         if not alnum_text_part and len(stripped_text) > 0:
#              logger.warning(f"API Validation: Field '{field_name}' ('{stripped_text[:30]}...') consists mainly of symbols.")
#              return False
#         elif alnum_text_part and len(set(alnum_text_part.lower())) < min_unique_chars_for_short_text:
#             logger.warning(f"API Validation: Field '{field_name}' ('{stripped_text[:30]}...') lacks character diversity for its length.")
#             return False
#     return True


# @app.get("/")
# async def root():
#     return {"message": "Executive Multi-Agent Backend is running."}

# @app.post("/market_intelligence")
# async def get_market_intelligence(
#     company_information: str = Form(""),
#     supporting_documents: list[UploadFile] = File([])
# ):
#     endpoint_name = "/market_intelligence"
#     # Validate company_information - crucial for this endpoint
#     if company_information and not is_api_input_valid(company_information, "company_information", min_length=3, short_text_threshold=20, min_unique_chars_for_short_text=1): # Company names can be very short (e.g. "GE")
#         # If supporting_documents are also empty, then definitely an issue.
#         if not supporting_documents:
#             raise HTTPException(status_code=400, detail="Provided 'company_information' is not meaningful, and no supporting documents were uploaded.")
#         logger.warning(f"{endpoint_name}: 'company_information' seems weak, will rely heavily on documents.")
#         # Allow to proceed if documents are present. Agent will handle bad company_information.

#     logger.info(f"{endpoint_name}: Received company_information: {company_information[:100]}...")
#     temp_file_paths, loaded_documents_content = [], []
#     try:
#         agent = MarketIntelligenceAgent(verbose=True)
#         for doc_file in supporting_documents:
#             logger.info(f"{endpoint_name}: Processing file: {doc_file.filename}")
#             with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(doc_file.filename)[-1]) as tmp:
#                 shutil.copyfileobj(doc_file.file, tmp); tmp_path = tmp.name
#             temp_file_paths.append(tmp_path)
#             try:
#                 text_content = ""
#                 fname_lower = doc_file.filename.lower()
#                 if fname_lower.endswith(".pdf"): text_content = agent.load_pdf_from_file(tmp_path)
#                 elif fname_lower.endswith(".txt"): text_content = agent.load_text_from_file(tmp_path)
#                 elif fname_lower.endswith(".json"): text_content = agent._load_json_as_text_from_file(tmp_path)
#                 else: logger.warning(f"{endpoint_name}: Unsupported file: {doc_file.filename}"); continue
#                 if text_content and text_content.strip(): loaded_documents_content.append(text_content)
#                 else: logger.warning(f"{endpoint_name}: No content from {doc_file.filename}")
#             except Exception as e: logger.error(f"{endpoint_name}: Error processing {doc_file.filename}: {e}")
        
#         all_docs_text = "\n\n---DOC_SEP---\n\n".join(filter(None, loaded_documents_content))
        
#         # Agent's extract_company_details will handle bad company_information string further
#         company_details = agent.extract_company_details(company_information if company_information.strip() else all_docs_text)

#         market_input = {
#             "company_name": company_details.get('company_name', 'Unknown'),
#             "company_location": company_details.get('company_location', 'Unknown'),
#             "geography": company_details.get('geography', 'Unknown'),
#             "supporting_documents": all_docs_text,
#         }
#         market_report = agent.run(market_input)
#         if market_report.startswith("Error:"): # Check if agent itself returned an error
#             raise HTTPException(status_code=400, detail=market_report) # Propagate agent error
#         return {"market_report": market_report}
#     except HTTPException as http_exc: raise http_exc
#     except Exception as e: logger.exception(f"{endpoint_name}: Error: {e}"); raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         for path in temp_file_paths:
#             try: os.remove(path)
#             except Exception as ex: logger.warning(f"{endpoint_name}: Cleanup error {path}: {ex}")

# @app.post("/job_description/")
# async def create_job_description(
#     manual_input: str = Form(""),
#     files: List[UploadFile] = File([])
# ):
#     endpoint_name = "/job_description/"
#     if manual_input and not is_api_input_valid(manual_input, "manual_input", min_length=20):
#         if not files: # If no files to compensate for bad manual_input
#              raise HTTPException(status_code=400, detail="Provided 'manual_input' is not meaningful, and no files were uploaded.")
#         logger.warning(f"{endpoint_name}: 'manual_input' seems weak, agent will be notified or rely on files.")
#         # Agent's internal logic will handle this specific case for `manual_input`.

#     logger.info(f"{endpoint_name}: Manual_input: {manual_input[:100]}..., Files: {len(files)}")
#     temp_paths = []
#     try:
#         agent = JobDescriptionWriterAgent(verbose=True)
#         for file_obj in files:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.filename)[-1]) as tmp:
#                 shutil.copyfileobj(file_obj.file, tmp); temp_paths.append(tmp.name)
        
#         job_description = agent.run(manual_input=manual_input, file_paths=temp_paths)
#         if job_description.startswith("Error:"):
#             raise HTTPException(status_code=400, detail=job_description)
#         return {"job_description": job_description}
#     except HTTPException as http_exc: raise http_exc
#     except Exception as e: logger.exception(f"{endpoint_name}: Error: {e}"); raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         for path in temp_paths:
#             try: os.remove(path)
#             except Exception as ex: logger.warning(f"{endpoint_name}: Cleanup error {path}: {ex}")

# @app.post("/client_creator/")
# async def create_client_persona_prompt( 
#     client_description: Optional[str] = Form(None),
#     transcript_files: Optional[List[UploadFile]] = File(None)
# ):
#     endpoint_name = "/client_creator/"
#     # Validate client_description only if it's provided
#     if client_description and not is_api_input_valid(client_description, "client_description", min_length=20):
#         if not transcript_files: # If no files to compensate
#             raise HTTPException(status_code=400, detail="Provided 'client_description' is not meaningful, and no transcript files were uploaded.")
#         logger.warning(f"{endpoint_name}: 'client_description' seems weak, agent will be notified or rely on files.")
#         # Agent handles empty or weak description if files are present.

#     logger.info(f"{endpoint_name}: Client_desc: {bool(client_description)}, Files: {len(transcript_files) if transcript_files else 0}")
#     temp_paths = []
#     try:
#         if not client_description and not transcript_files:
#             raise HTTPException(status_code=400, detail="Provide client_description or transcript_files.")
#         if transcript_files:
#             for file_obj in transcript_files:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.filename)[-1]) as tmp:
#                     shutil.copyfileobj(file_obj.file, tmp); temp_paths.append(tmp.name)
        
#         agent = ClientRepresentativeCreatorAgent(verbose=True)
#         prompt = agent.run(client_description=client_description or "", transcript_file_paths=temp_paths)
#         if prompt.startswith("Error:"):
#             raise HTTPException(status_code=400, detail=prompt)
#         return {"generated_prompt": prompt}
#     except HTTPException as http_exc: raise http_exc
#     except Exception as e: logger.exception(f"{endpoint_name}: Error: {e}"); raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         for path in temp_paths:
#             try: os.remove(path)
#             except Exception as ex: logger.warning(f"{endpoint_name}: Cleanup error {path}: {ex}")

# @app.post("/client_feedback/")
# async def get_client_feedback( 
#     user_input: str = Form(""), 
#     files: List[UploadFile] = File(default=[])
# ):
#     endpoint_name = "/client_feedback/"
#     if user_input and not is_api_input_valid(user_input, "user_input", min_length=20): # User input is the main document to review + persona
#         if not files:
#             raise HTTPException(status_code=400, detail="Provided 'user_input' is not meaningful, and no files were uploaded for context.")
#         logger.warning(f"{endpoint_name}: 'user_input' seems weak, agent will attempt with files.")
#         # Agent's `run` will handle this.

#     logger.info(f"{endpoint_name}: User_input: {bool(user_input)}, Files: {len(files)}")
#     temp_paths = []
#     try:
#         if not user_input.strip() and not files:
#             raise HTTPException(status_code=400, detail="No input (user_input or files) provided.")
#         for file_obj in files:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.filename)[-1]) as tmp:
#                 shutil.copyfileobj(file_obj.file, tmp); temp_paths.append(tmp.name)
        
#         agent = ClientRepresentativeAgent(verbose=True)
#         response = agent.run(input_statement=user_input, transcript_file_paths=temp_paths)
#         if response.startswith("Error:"):
#             raise HTTPException(status_code=400, detail=response)
#         return {"client_representative_feedback": response}
#     except HTTPException as http_exc: raise http_exc
#     except Exception as e: logger.exception(f"{endpoint_name}: Error: {e}"); raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         for path in temp_paths:
#             try: os.remove(path)
#             except Exception as ex: logger.warning(f"{endpoint_name}: Cleanup error {path}: {ex}")

# @app.post("/interview_report")
# async def create_interview_report(
#     input_text: Optional[str] = Form(None),
#     files: Optional[List[UploadFile]] = File(None)
# ):
#     endpoint_name = "/interview_report"
#     if input_text and not is_api_input_valid(input_text, "input_text", min_length=20):
#         if not files:
#             raise HTTPException(status_code=400, detail="Provided 'input_text' is not meaningful, and no files were uploaded.")
#         logger.warning(f"{endpoint_name}: 'input_text' seems weak, agent will primarily use files if available.")
#         # Agent's `run` logic handles empty/weak input_text if files are present.

#     logger.info(f"{endpoint_name}: Input_text: {bool(input_text)}, Files: {len(files) if files else 0}")
#     temp_paths_cleanup, agent_paths = [], []
#     try:
#         if not (input_text and input_text.strip()) and not files:
#             raise HTTPException(status_code=400, detail="Provide input_text or file attachments.")
#         if files:
#             for up_file in files:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up_file.filename)[-1]) as tmp:
#                     shutil.copyfileobj(up_file.file, tmp)
#                     temp_paths_cleanup.append(tmp.name); agent_paths.append(tmp.name)
        
#         agent = InterviewReportCreatorAgent(verbose=True)
#         report = agent.run(input_text=input_text or "", attachment_paths=agent_paths)
#         if report.startswith("Error:"):
#             raise HTTPException(status_code=400, detail=report)
#         return {"interview_report": report}
#     except HTTPException as http_exc: raise http_exc
#     except Exception as e: logger.exception(f"{endpoint_name}: Error: {e}"); raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         for path in temp_paths_cleanup:
#             try: os.remove(path)
#             except Exception as ex: logger.warning(f"{endpoint_name}: Cleanup error {path}: {ex}")

# @app.post("/multi_agent_run/")
# async def multi_agent_pipeline(
#     company_info: str = Form(...), 
#     files: List[UploadFile] = File([]) 
# ):
#     endpoint_name = "/multi_agent_run/"
#     if not is_api_input_valid(company_info, "company_info", min_length=5): # Company info can be a short query or name
#          raise HTTPException(status_code=400, detail="Provided 'company_info' is not meaningful or too short.")

#     logger.info(f"{endpoint_name}: Company_info: {bool(company_info)}, Files: {len(files)}")
#     temp_paths = [] 
#     try:
#         for f_obj in files:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f_obj.filename)[-1]) as tmp:
#                 shutil.copyfileobj(f_obj.file, tmp); temp_paths.append(tmp.name)
        
#         orchestrator = MultiAgentOrchestrator(verbose=True)
#         result = orchestrator.run_pipeline(company_info, temp_paths) 
#         # Orchestrator now checks for "Error:" prefixes and can return them in results
#         # Check if a top-level error was returned by orchestrator due to early pipeline failure
#         if isinstance(result, dict) and any(key.endswith("_error") for key in result.keys()):
#             # Find first error message to report
#             error_message = "Pipeline encountered an error. Check individual agent outputs."
#             for key, value in result.items():
#                 if key.endswith("_error") and isinstance(value, str):
#                     error_message = value
#                     break
#             logger.error(f"{endpoint_name}: Orchestrator reported an error: {error_message}")
#             # Depending on how you want to signal this, you might raise HTTPException
#             # or let the client parse the `results` dict. For now, let client parse.
#             # raise HTTPException(status_code=400, detail=error_message) 

#         return result
#     except HTTPException as http_exc: raise http_exc
#     except Exception as e: logger.exception(f"{endpoint_name}: Error: {e}"); raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         for path in temp_paths:
#             try: os.remove(path)
#             except Exception as ex: logger.warning(f"{endpoint_name}: Cleanup error {path}: {ex}")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter
import logging, os, tempfile, json, io, shutil, uuid, fitz
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from agents.client_representative_agent import ClientRepresentativeAgent
from agents.client_representative_creator_agent import ClientRepresentativeCreatorAgent
from agents.interview_report_creator_agent import InterviewReportCreatorAgent
from agents.job_description_writer_agent import JobDescriptionWriterAgent
from agents.market_intelligence_agent import MarketIntelligenceAgent
from orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
 
origins = [
    "*", 
    "http://localhost:3000", "http://127.0.0.1:3000",
    "https://executive-multi-agent-frontend.vercel.app", 
]

app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Helper function to check for basic input validity (can be in a utils.py)
def is_api_input_valid(text: Optional[str], field_name: str, min_length: int = 10, short_text_threshold: int = 20, min_unique_chars_for_short_text: int = 3) -> bool:
    if text is None: # Allow optional fields to be None
        return True
    if not text.strip(): # If provided but empty string
        # Depending on whether an empty string is acceptable for this optional field
        logger.warning(f"API Validation: Optional field '{field_name}' is an empty string.")
        return True # Or False if empty string is not allowed even if optional
        
    stripped_text = text.strip()
    if len(stripped_text) < min_length:
        logger.warning(f"API Validation: Field '{field_name}' ('{stripped_text[:30]}...') is too short (min_length: {min_length}).")
        return False
    if len(stripped_text) <= short_text_threshold:
        alnum_text_part = ''.join(filter(str.isalnum, stripped_text))
        if not alnum_text_part and len(stripped_text) > 0:
             logger.warning(f"API Validation: Field '{field_name}' ('{stripped_text[:30]}...') consists mainly of symbols.")
             return False
        elif alnum_text_part and len(set(alnum_text_part.lower())) < min_unique_chars_for_short_text:
            logger.warning(f"API Validation: Field '{field_name}' ('{stripped_text[:30]}...') lacks character diversity for its length.")
            return False
    return True


@app.get("/")
async def root():
    return {"message": "Executive Multi-Agent Backend is running."}

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
        
        # Agent's extract_company_details will handle bad company_information string further
        company_details = agent.extract_company_details(company_information if company_information.strip() else all_docs_text)

        market_input = {
            "company_name": company_details.get('company_name', 'Unknown'),
            "company_location": company_details.get('company_location', 'Unknown'),
            "geography": company_details.get('geography', 'Unknown'),
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

@app.post("/job_description/")
async def create_job_description(
    manual_input: str = Form(""),
    files: List[UploadFile] = File([])
):
    endpoint_name = "/job_description/"
    if manual_input and not is_api_input_valid(manual_input, "manual_input", min_length=20):
        if not files: # If no files to compensate for bad manual_input
             raise HTTPException(status_code=400, detail="Provided 'manual_input' is not meaningful, and no files were uploaded.")
        logger.warning(f"{endpoint_name}: 'manual_input' seems weak, agent will be notified or rely on files.")
        # Agent's internal logic will handle this specific case for `manual_input`.

    logger.info(f"{endpoint_name}: Manual_input: {manual_input[:100]}..., Files: {len(files)}")
    temp_paths = []
    try:
        agent = JobDescriptionWriterAgent(verbose=True)
        for file_obj in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.filename)[-1]) as tmp:
                shutil.copyfileobj(file_obj.file, tmp); temp_paths.append(tmp.name)
        
        job_description = agent.run(manual_input=manual_input, file_paths=temp_paths)
        if job_description.startswith("Error:"):
            raise HTTPException(status_code=400, detail=job_description)
        return {"job_description": job_description}
    except HTTPException as http_exc: raise http_exc
    except Exception as e: logger.exception(f"{endpoint_name}: Error: {e}"); raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in temp_paths:
            try: os.remove(path)
            except Exception as ex: logger.warning(f"{endpoint_name}: Cleanup error {path}: {ex}")

@app.post("/client_creator/")
async def create_client_persona_prompt( 
    client_description: Optional[str] = Form(None),
    transcript_files: Optional[List[UploadFile]] = File(None)
):
    endpoint_name = "/client_creator/"
    # Validate client_description only if it's provided
    if client_description and not is_api_input_valid(client_description, "client_description", min_length=20):
        if not transcript_files: # If no files to compensate
            raise HTTPException(status_code=400, detail="Provided 'client_description' is not meaningful, and no transcript files were uploaded.")
        logger.warning(f"{endpoint_name}: 'client_description' seems weak, agent will be notified or rely on files.")
        # Agent handles empty or weak description if files are present.

    logger.info(f"{endpoint_name}: Client_desc: {bool(client_description)}, Files: {len(transcript_files) if transcript_files else 0}")
    temp_paths = []
    try:
        if not client_description and not transcript_files:
            raise HTTPException(status_code=400, detail="Provide client_description or transcript_files.")
        if transcript_files:
            for file_obj in transcript_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.filename)[-1]) as tmp:
                    shutil.copyfileobj(file_obj.file, tmp); temp_paths.append(tmp.name)
        
        agent = ClientRepresentativeCreatorAgent(verbose=True)
        prompt = agent.run(client_description=client_description or "", transcript_file_paths=temp_paths)
        if prompt.startswith("Error:"):
            raise HTTPException(status_code=400, detail=prompt)
        return {"generated_prompt": prompt}
    except HTTPException as http_exc: raise http_exc
    except Exception as e: logger.exception(f"{endpoint_name}: Error: {e}"); raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in temp_paths:
            try: os.remove(path)
            except Exception as ex: logger.warning(f"{endpoint_name}: Cleanup error {path}: {ex}")

@app.post("/client_feedback/")
async def get_client_feedback( 
    user_input: str = Form(""), 
    files: List[UploadFile] = File(default=[])
):
    endpoint_name = "/client_feedback/"
    if user_input and not is_api_input_valid(user_input, "user_input", min_length=20): # User input is the main document to review + persona
        if not files:
            raise HTTPException(status_code=400, detail="Provided 'user_input' is not meaningful, and no files were uploaded for context.")
        logger.warning(f"{endpoint_name}: 'user_input' seems weak, agent will attempt with files.")
        # Agent's `run` will handle this.

    logger.info(f"{endpoint_name}: User_input: {bool(user_input)}, Files: {len(files)}")
    temp_paths = []
    try:
        if not user_input.strip() and not files:
            raise HTTPException(status_code=400, detail="No input (user_input or files) provided.")
        for file_obj in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.filename)[-1]) as tmp:
                shutil.copyfileobj(file_obj.file, tmp); temp_paths.append(tmp.name)
        
        agent = ClientRepresentativeAgent(verbose=True)
        response = agent.run(input_statement=user_input, transcript_file_paths=temp_paths)
        if response.startswith("Error:"):
            raise HTTPException(status_code=400, detail=response)
        return {"client_representative_feedback": response}
    except HTTPException as http_exc: raise http_exc
    except Exception as e: logger.exception(f"{endpoint_name}: Error: {e}"); raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in temp_paths:
            try: os.remove(path)
            except Exception as ex: logger.warning(f"{endpoint_name}: Cleanup error {path}: {ex}")

@app.post("/interview_report")
async def create_interview_report(
    input_text: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    endpoint_name = "/interview_report"
    if input_text and not is_api_input_valid(input_text, "input_text", min_length=20):
        if not files:
            raise HTTPException(status_code=400, detail="Provided 'input_text' is not meaningful, and no files were uploaded.")
        logger.warning(f"{endpoint_name}: 'input_text' seems weak, agent will primarily use files if available.")
        # Agent's `run` logic handles empty/weak input_text if files are present.

    logger.info(f"{endpoint_name}: Input_text: {bool(input_text)}, Files: {len(files) if files else 0}")
    temp_paths_cleanup, agent_paths = [], []
    try:
        if not (input_text and input_text.strip()) and not files:
            raise HTTPException(status_code=400, detail="Provide input_text or file attachments.")
        if files:
            for up_file in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up_file.filename)[-1]) as tmp:
                    shutil.copyfileobj(up_file.file, tmp)
                    temp_paths_cleanup.append(tmp.name); agent_paths.append(tmp.name)
        
        agent = InterviewReportCreatorAgent(verbose=True)
        report = agent.run(input_text=input_text or "", attachment_paths=agent_paths)
        if report.startswith("Error:"):
            raise HTTPException(status_code=400, detail=report)
        return {"interview_report": report}
    except HTTPException as http_exc: raise http_exc
    except Exception as e: logger.exception(f"{endpoint_name}: Error: {e}"); raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in temp_paths_cleanup:
            try: os.remove(path)
            except Exception as ex: logger.warning(f"{endpoint_name}: Cleanup error {path}: {ex}")

@app.post("/multi_agent_run/")
async def multi_agent_pipeline(
    company_info: str = Form(...), 
    files: List[UploadFile] = File([]) 
):
    endpoint_name = "/multi_agent_run/"
    if not is_api_input_valid(company_info, "company_info", min_length=5): # Company info can be a short query or name
         raise HTTPException(status_code=400, detail="Provided 'company_info' is not meaningful or too short.")

    logger.info(f"{endpoint_name}: Company_info: {bool(company_info)}, Files: {len(files)}")
    temp_paths = [] 
    try:
        for f_obj in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f_obj.filename)[-1]) as tmp:
                shutil.copyfileobj(f_obj.file, tmp); temp_paths.append(tmp.name)
        
        orchestrator = MultiAgentOrchestrator(verbose=True)
        result = orchestrator.run_pipeline(company_info, temp_paths) 
        # Orchestrator now checks for "Error:" prefixes and can return them in results
        # Check if a top-level error was returned by orchestrator due to early pipeline failure
        if isinstance(result, dict) and any(key.endswith("_error") for key in result.keys()):
            # Find first error message to report
            error_message = "Pipeline encountered an error. Check individual agent outputs."
            for key, value in result.items():
                if key.endswith("_error") and isinstance(value, str):
                    error_message = value
                    break
            logger.error(f"{endpoint_name}: Orchestrator reported an error: {error_message}")
            # Depending on how you want to signal this, you might raise HTTPException
            # or let the client parse the `results` dict. For now, let client parse.
            # raise HTTPException(status_code=400, detail=error_message) 

        return result
    except HTTPException as http_exc: raise http_exc
    except Exception as e: logger.exception(f"{endpoint_name}: Error: {e}"); raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in temp_paths:
            try: os.remove(path)
            except Exception as ex: logger.warning(f"{endpoint_name}: Cleanup error {path}: {ex}")

@app.post("/interactive_jd/")
async def interactive_jd_creation(
    message: str = Form(...),
    state: str = Optional[Form(None)],
    files: List[UploadFile] = Optional[File(None)]
):
    """Single endpoint for interactive JD creation process"""
    endpoint_name = "/interactive_jd/"
    temp_paths = []

    try:
        # Process any uploaded files
        for file_obj in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.filename)[-1]) as tmp:
                shutil.copyfileobj(file_obj.file, tmp)
                temp_paths.append(tmp.name)

        # Initialize orchestrator
        orchestrator = MultiAgentOrchestrator(verbose=True)

        if state == "start":
            # Starting new JD creation process
            response = orchestrator.start_jd_creation(message)
        else:
            # Continue existing process
            response = orchestrator.process_jd_input(message, state, temp_paths)

        return response

    except Exception as e:
        logger.exception(f"{endpoint_name}: Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except Exception as ex:
                logger.warning(f"{endpoint_name}: Cleanup error {path}: {ex}")