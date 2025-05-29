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

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()
# router = APIRouter()
 
# #CORS Configuration
# origins = [
#     "*", #USE AT YOUR OWN RISK
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
#     "https://executive-multi-agent-frontend.vercel.app", # Add the Vercel URL,
#     "https://bb45-2405-201-4021-112e-a09c-f089-f179-f075.ngrok-free.app"  # Replace with your ngrok URL (e.g., "https://your-ngrok-url.ngrok-free.app")
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# @app.post("/market_intelligence")
# async def get_market_intelligence(
#     company_information: str = Form(""),
#     supporting_documents: list[UploadFile] = File([])
# ):
#     logger.info(f"Received company_information: {company_information}")
#     logger.info(f"Number of supporting documents: {len(supporting_documents)}")
    
#     temp_file_paths = []
#     loaded_documents_content = []

#     try:
#         agent = MarketIntelligenceAgent()

#         for doc_file in supporting_documents:
#             logger.info(f"Processing file: {doc_file.filename} (Content Type: {doc_file.content_type})")
#             filename_lower = doc_file.filename.lower()
            
#             with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(doc_file.filename)[-1]) as tmp:
#                 shutil.copyfileobj(doc_file.file, tmp)
#                 tmp_path = tmp.name
#             temp_file_paths.append(tmp_path) # For cleanup

#             try:
#                 if filename_lower.endswith(".pdf"):
#                     text = agent.load_pdf_from_file(tmp_path)
#                     loaded_documents_content.append(text)
#                 elif filename_lower.endswith(".txt"):
#                     text = agent.load_text_from_file(tmp_path) # Assuming this method exists or is added
#                     loaded_documents_content.append(text)
#                 elif filename_lower.endswith(".json"):
#                     # Replicate JSON loading logic if MarketIntelligenceAgent needs specific handling
#                     # or use a generic text loader if it expects a string dump of JSON.
#                     # For consistency with its own method, let's use/add one for JSON.
#                     if hasattr(agent, '_load_json_as_text_from_file'):
#                         text = agent._load_json_as_text_from_file(tmp_path)
#                     else: # Fallback to simple read if method not present (though we'll add it)
#                         with open(tmp_path, "r", encoding="utf-8") as f:
#                             json_data = json.load(f)
#                         text = json.dumps(json_data, indent=2)
#                     loaded_documents_content.append(text)
#                 else:
#                     logger.warning(f"Unsupported file format for market intelligence: {doc_file.filename}")
#             except Exception as e:
#                 logger.error(f"Could not process file {doc_file.filename} for market intelligence: {e}")

#         all_documents_text = "\n\n---SEPARATOR---\n\n".join(filter(None, loaded_documents_content))

#         company_details = agent.extract_company_details(company_information)

#         market_input = {
#             "company_name": company_details.get('company_name', ''),
#             "company_location": company_details.get('company_location', ''),
#             "geography": company_details.get('geography', ''),
#             "supporting_documents": all_documents_text,
#             "role": agent.role,
#             "goal": agent.goal
#         }

#         market_report = agent.run(market_input)

#         # Optionally save the report server-side
#         # report_filename = f"market_report_{uuid.uuid4()}.txt"
#         # with open(report_filename, "w", encoding="utf-8") as f:
#         #     f.write(market_report)
#         # logger.info(f"Market intelligence report saved to {report_filename}")

#         return {"market_report": market_report}

#     except HTTPException as http_exc:
#         raise http_exc
#     except Exception as e:
#         logger.exception(f"Error generating market intelligence report: {e}")
#         raise HTTPException(status_code=500, detail=f"Error generating market intelligence report: {str(e)}")
#     finally:
#         for path in temp_file_paths:
#             try:
#                 if os.path.exists(path):
#                     os.remove(path)
#             except Exception as cleanup_error:
#                 logger.warning(f"Could not delete temp file {path}: {cleanup_error}")

# @app.post("/job_description/")
# async def create_job_description(
#     manual_input: str = Form(""),
#     files: List[UploadFile] = File([])
# ):
#     logger.info(f"Received manual_input for JD: {manual_input[:100]}...")
#     logger.info(f"Number of uploaded files for JD: {len(files)}")

#     temp_file_paths_for_jd = []
#     try:
#         agent = JobDescriptionWriterAgent(verbose=True)

#         for file_obj in files:
#             suffix = os.path.splitext(file_obj.filename)[-1]
#             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                 shutil.copyfileobj(file_obj.file, tmp) # Use shutil.copyfileobj for UploadFile
#                 tmp_path = tmp.name
#             temp_file_paths_for_jd.append(tmp_path)
#             logger.info(f"JD Agent: Saved uploaded file to temp path: {tmp_path}")

#         # The JobDescriptionWriterAgent's run method expects file_paths
#         job_description = agent.run(
#             manual_input=manual_input,
#             file_paths=temp_file_paths_for_jd
#         )

#         logger.info("Job description generated successfully.")
#         return {"job_description": job_description}

#     except HTTPException as http_exc:
#         raise http_exc
#     except Exception as e:
#         logger.exception("Error generating job description")
#         raise HTTPException(status_code=500, detail=f"Error generating JD: {str(e)}")
#     finally:
#         for path in temp_file_paths_for_jd:
#             try:
#                 if os.path.exists(path):
#                     os.remove(path)
#                     logger.info(f"JD Agent: Cleaned up temp file: {path}")
#             except Exception as cleanup_error:
#                 logger.warning(f"JD Agent: Could not delete temp file {path}: {cleanup_error}")


# @app.post("/client_creator/")
# async def get_client_feedback( # This endpoint seems to be for creating a client *prompt*, not getting feedback
#     client_description: Optional[str] = Form(None, description="Free-form client description (persona, values, priorities, tone, etc.)"),
#     transcript_files: Optional[List[UploadFile]] = File(None, description="Optional transcript files (PDF, TXT, JSON)")
# ):
#     logger.info("Client Creator: Received request.")
#     temp_file_paths = []

#     try:
#         if not client_description and not transcript_files:
#             raise HTTPException(
#                 status_code=400,
#                 detail="You must provide either a client_description or transcript_files (or both)."
#             )

#         if transcript_files:
#             for file_obj in transcript_files:
#                 suffix = os.path.splitext(file_obj.filename)[-1]
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                     shutil.copyfileobj(file_obj.file, tmp)
#                     temp_file_paths.append(tmp.name)
#                     logger.info(f"Client Creator: Saved transcript file to temp path: {tmp.name}")

#         description_text = client_description.strip() if client_description else "Not explicitly described. Please infer from communication tone."

#         agent = ClientRepresentativeCreatorAgent(verbose=True)
#         generated_prompt = agent.run(
#             client_description=description_text,
#             transcript_file_paths=temp_file_paths if temp_file_paths else None # Agent expects list of paths
#         )
#         logger.info("Client Creator: Prompt generated successfully.")
#         return {"generated_prompt": generated_prompt}

#     except HTTPException as http_exc:
#         raise http_exc
#     except Exception as e:
#         logger.exception(f"Error generating client representative prompt: {e}")
#         raise HTTPException(status_code=500, detail=f"An error occurred while generating the prompt: {str(e)}")
#     finally:
#         for path in temp_file_paths:
#             try:
#                 if os.path.exists(path):
#                     os.remove(path)
#                     logger.info(f"Client Creator: Cleaned up temp file: {path}")
#             except Exception as cleanup_error:
#                 logger.warning(f"Client Creator: Could not delete temp file {path}: {cleanup_error}")

# @app.post("/client_feedback/")
# async def create_client_characteristics( # Renamed from create_client_characteristics for clarity, as it generates feedback
#     user_input: str = Form(""),
#     files: List[UploadFile] = File(default=[])
# ):
#     logger.info("Client Feedback: Received request.")
#     temp_file_paths = []

#     try:
#         if not user_input.strip() and not files:
#             raise HTTPException(status_code=400, detail="No input (user_input or files) provided for client feedback.")

#         for file_obj in files:
#             suffix = os.path.splitext(file_obj.filename)[-1]
#             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                 shutil.copyfileobj(file_obj.file, tmp)
#                 temp_file_paths.append(tmp.name)
#                 logger.info(f"Client Feedback: Saved uploaded file to temp path: {tmp.name}")

#         agent = ClientRepresentativeAgent(verbose=True)
#         response = agent.run(
#             input_statement=user_input,
#             transcript_file_paths=temp_file_paths # Agent expects list of paths
#         )
#         logger.info("Client Feedback: Feedback generated successfully.")
#         return {"client_representative_feedback": response}

#     except HTTPException as http_exception:
#         raise http_exception
#     except Exception as e:
#         logger.exception("Error processing client feedback (characteristics)")
#         raise HTTPException(status_code=500, detail=f"Internal Server Error during client feedback: {str(e)}")
#     finally:
#         for path in temp_file_paths:
#             try:
#                 if os.path.exists(path):
#                     os.remove(path)
#                     logger.info(f"Client Feedback: Cleaned up temp file: {path}")
#             except Exception as cleanup_error:
#                 logger.warning(f"Client Feedback: Could not delete temp file {path}: {cleanup_error}")


# @app.post("/interview_report")
# async def create_interview_report(
#     input_text: Optional[str] = Form(
#         None,
#         description="Optional freeform input from user (e.g. consultant assessment or structured sections with delimiters like ---JOB SPEC---)."
#     ),
#     files: Optional[List[UploadFile]] = File(
#         None,
#         description="Optional supporting files (job spec, CV, transcript, etc.) in .pdf, .txt, or .json format."
#     )
# ):
#     logger.info("Interview Report: Received request.")
#     temp_file_paths = []
#     try:
#         if not input_text and not files:
#             raise HTTPException(
#                 status_code=400,
#                 detail="You must provide either input_text, file attachments, or both for interview report."
#             )

#         attachment_paths_for_agent = []
#         if files:
#             for upload_file in files:
#                 suffix = os.path.splitext(upload_file.filename)[-1]
#                 # Create temp file and ensure it's written to before getting its name for the agent
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file_obj:
#                     shutil.copyfileobj(upload_file.file, tmp_file_obj)
#                     # tmp_file_obj is closed upon exiting 'with', but file still exists due to delete=False
#                     temp_file_paths.append(tmp_file_obj.name) # Add to list for cleanup
#                     attachment_paths_for_agent.append(tmp_file_obj.name) # Add to list for agent
#                 logger.info(f"Interview Report: Saved uploaded file to temp path: {temp_file_paths[-1]}")


#         agent = InterviewReportCreatorAgent(verbose=True)
#         report = agent.run(
#             input_text=input_text or "",
#             attachment_paths=attachment_paths_for_agent if attachment_paths_for_agent else None
#         )
#         logger.info("Interview Report: Report generated successfully.")
#         return {"interview_report": report}

#     except HTTPException as http_exc:
#         raise http_exc
#     except Exception as e:
#         logger.exception(f"Error creating interview report: {e}")
#         raise HTTPException(status_code=500, detail=f"An internal error occurred while generating the interview report: {str(e)}")
#     finally:
#         for path in temp_file_paths:
#             try:
#                 if os.path.exists(path):
#                     os.remove(path)
#                     logger.info(f"Interview Report: Cleaned up temp file: {path}")
#             except Exception as cleanup_error:
#                 logger.warning(f"Interview Report: Could not delete temp file {path}: {cleanup_error}")

# @app.post("/multi_agent_run/")
# async def multi_agent_pipeline(
#     company_info: str = Form(...),
#     files: List[UploadFile] = File([])
# ):
#     logger.info("Multi-Agent Pipeline: Received request.")
#     temp_file_paths = [] # Initialize for the finally block
#     try:
#         # Save uploaded files to temporary paths
#         for f in files:
#             suffix = os.path.splitext(f.filename)[-1]
#             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file_obj:
#                 shutil.copyfileobj(f.file, tmp_file_obj)
#                 temp_file_paths.append(tmp_file_obj.name)
#             logger.info(f"Multi-Agent Pipeline: Saved uploaded file to temp path: {temp_file_paths[-1]}")

#         orchestrator = MultiAgentOrchestrator(verbose=True)
#         result = orchestrator.run_pipeline(company_info, temp_file_paths) # Pass list of paths

#         logger.info("Multi-Agent Pipeline: Pipeline executed successfully.")
#         return result

#     except HTTPException as http_exc:
#         raise http_exc
#     except Exception as e:
#         logger.exception("Error running multi-agent pipeline")
#         raise HTTPException(status_code=500, detail=f"Error in multi-agent pipeline: {str(e)}")
#     finally:
#         for path in temp_file_paths:
#             try:
#                 if os.path.exists(path):
#                     os.remove(path)
#                     logger.info(f"Multi-Agent Pipeline: Cleaned up temp file: {path}")
#             except Exception as cleanup_error:
#                 logger.warning(f"Multi-Agent Pipeline: Could not delete temp file {path}: {cleanup_error}")

# # app.include_router(router) # If you are using APIRouter for organization, uncomment and define routes on 'router'
# # If not, @app.post decorator is fine.

# # The CORS middleware was added twice. Removed one. It's correctly placed after FastAPI app initialization.

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