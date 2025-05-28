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
#     print(f"Received company_information: {company_information}")
#     print(f"Number of supporting documents: {len(supporting_documents)}")
#     for doc in supporting_documents:
#         print(f"Received file: {doc.filename} (Content Type: {doc.content_type})")

#     try:

#         # Initialize the agent
#         agent = MarketIntelligenceAgent()

#         loaded_documents = []

#         for doc in supporting_documents:
#             filename = doc.filename.lower()

#             # Save the uploaded file to a temporary location
#             with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(doc.filename)[-1]) as tmp:
#                 tmp.write(await doc.read())
#                 tmp_path = tmp.name

#             try:
#                 if filename.endswith(".pdf"):
#                     text = agent.load_pdf_from_file(tmp_path)
#                     loaded_documents.append(text)

#                 elif filename.endswith(".txt"):
#                     with open(tmp_path, "r", encoding="utf-8") as f:
#                         text = f.read()
#                     loaded_documents.append(text)

#                 elif filename.endswith(".json"):
#                     with open(tmp_path, "r", encoding="utf-8") as f:
#                         json_data = json.load(f)
#                     text = json.dumps(json_data, indent=2)  # Pretty print for LLM readability
#                     loaded_documents.append(text)

#                 else:
#                     logger.warning(f"Unsupported file format: {doc.filename}")

#             except Exception as e:
#                 logger.warning(f"Could not process file {doc.filename}: {e}")

#         # Combine all loaded documents into one string
#         all_documents_text = "\n".join(loaded_documents)

#         # Extract company details
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

#         with open("market_report.txt", "w", encoding="utf-8") as f:
#             f.write(market_report)

#         logger.info("Market intelligence report saved to market_report.txt")

#         return {"market_report": market_report}

#     except Exception as e:
#         logger.exception(f"Error generating market intelligence report: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/job_description/")
# async def create_job_description(
#     manual_input: str = Form(""),
#     files: List[UploadFile] = File([])
# ):
#     logger.info(f"Received manual_input: {manual_input[:100]}...")
#     logger.info(f"Number of uploaded files: {len(files)}")

#     try:
#         agent = JobDescriptionWriterAgent(verbose=True)
#         loaded_documents = []

#         for file in files:
#             filename = file.filename.lower()
#             suffix = os.path.splitext(filename)[-1]

#             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                 tmp.write(await file.read())
#                 tmp_path = tmp.name

#             try:
#                 label = f"\n--- File: {filename} ---\n"

#                 if filename.endswith(".pdf"):
#                     text = agent._extract_text_from_pdf(tmp_path)
#                 elif filename.endswith(".txt"):
#                     with open(tmp_path, "r", encoding="utf-8") as f:
#                         text = f.read()
#                 elif filename.endswith(".json"):
#                     text = agent._extract_transcript_from_json(tmp_path)
#                 else:
#                     logger.warning(f"Unsupported file format: {filename}")
#                     continue

#                 loaded_documents.append(label + text)

#             except Exception as e:
#                 logger.warning(f"Could not process file {filename}: {e}")

#         # Combine all text segments
#         combined_text = "\n".join(loaded_documents).strip()

#         job_description = agent.chain.run({
#             "manual_input": manual_input,
#             "file_text": combined_text
#         })

#         logger.info("Job description generated successfully.")
#         return {"job_description": job_description}

#     except Exception as e:
#         logger.exception("Error generating job description")
#         raise HTTPException(status_code=500, detail=f"Error generating JD: {str(e)}")

# @app.post("/client_creator/")
# async def get_client_feedback(
#     client_description: Optional[str] = Form(None, description="Free-form client description (persona, values, priorities, tone, etc.)"),
#     transcript_files: Optional[List[UploadFile]] = File(None, description="Optional transcript files (PDF, TXT, JSON)")
# ):
#     temp_file_paths = []

#     try:
#         # Validate input: at least one input
#         if not client_description and not transcript_files:
#             raise HTTPException(
#                 status_code=400,
#                 detail="You must provide either a client_description or transcript_files (or both)."
#             )

#         # Save uploaded transcript files
#         if transcript_files:
#             for file in transcript_files:
#                 contents = await file.read()
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
#                     tmp.write(contents)
#                     temp_file_paths.append(tmp.name)

#         # Use fallback text if client_description is missing
#         description_text = client_description.strip() if client_description else "Not explicitly described. Please infer from communication tone."

#         # Run the agent
#         agent = ClientRepresentativeCreatorAgent(verbose=True)
#         generated_prompt = agent.run(
#             client_description=description_text,
#             transcript_file_paths=temp_file_paths if temp_file_paths else None
#         )

#         return {"generated_prompt": generated_prompt}

#     except Exception as e:
#         logger.exception(f"Error generating client representative prompt: {e}")
#         raise HTTPException(status_code=500, detail="An error occurred while generating the prompt.")

#     finally:
#         # Clean up temp files
#         for path in temp_file_paths:
#             try:
#                 if os.path.exists(path):
#                     os.remove(path)
#             except Exception as cleanup_error:
#                 logger.warning(f"Could not delete temp file {path}: {cleanup_error}")

# @app.post("/client_feedback/")
# async def create_client_characteristics(
#     user_input: str = Form(""),
#     files: List[UploadFile] = File(default=[])
# ):
#     extracted_texts = []
#     filepaths = []

#     try:
#         #Save the files, and extract the paths
#         for file in files:
#             try:
#                 contents = await file.read()
#                 #Save to temporary file
#                 temp_file_path = f"temp_{file.filename}" #You need to implement this
#                 with open(temp_file_path, "wb") as f:
#                     f.write(contents)

#                 filepaths.append(temp_file_path) #This is what we will send to the agent

#             except Exception as file_error:
#                 logger.warning(f"Could not process file {file.filename}: {file_error}")
#                 raise HTTPException(status_code=400, detail=f"Could not process file {file.filename}: {file_error}")

#         # Validate that at least one source of input is present
#         if not user_input.strip() and not filepaths:
#             raise HTTPException(status_code=400, detail="No input or files provided.")

#         # Run the agent with combined input
#         agent = ClientRepresentativeAgent(verbose=True)
#         response = agent.run(input_statement=user_input, transcript_file_paths=filepaths)

#         #Delete all created files
#         for path in filepaths:
#             try:
#                 os.remove(path)
#             except:
#                 logger.warning(f"Cannot delete file {path}")

#         return {"client_representative_feedback": response}

#     except HTTPException as http_exception:
#         raise http_exception

#     except Exception as e:
#         logger.exception("Error processing client characteristics")
#         raise HTTPException(status_code=500, detail="Internal Server Error")

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
#     """
#     Generate a structured interview report (~300 words) from candidate info.
#     You can provide either or both:
#     - input_text: manual input (consultant assessment or structured sections)
#     - files: uploaded CV, job spec, interview transcript, etc.
#     """
#     temp_file_paths = []
#     print(f"Received input_text: {input_text}")
#     print(f"Received files: {files}")
#     try:
#         if not input_text and not files:
#             raise HTTPException(
#                 status_code=400,
#                 detail="You must provide either input_text, file attachments, or both."
#             )

#         attachment_paths = []

#         if files:
#             for upload_file in files:
#                 content_bytes = await upload_file.read()
#                 suffix = os.path.splitext(upload_file.filename)[-1]
#                 tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
#                 tmp_file.write(content_bytes)
#                 tmp_file.close()
#                 temp_file_paths.append(tmp_file.name)
#                 attachment_paths.append(tmp_file.name)

#         # Run the report generator
#         agent = InterviewReportCreatorAgent(verbose=True)
#         report = agent.run(
#             input_text=input_text or "",
#             attachment_paths=attachment_paths if attachment_paths else None
#         )

#         return {"interview_report": report}

#     except Exception as e:
#         logger.exception(f"Error creating interview report: {e}")
#         raise HTTPException(status_code=500, detail="An internal error occurred while generating the report.")

# @app.post("/multi_agent_run/")
# async def multi_agent_pipeline(
#     company_info: str = Form(...),
#     files: List[UploadFile] = File([])
# ):
#     try:
#         tmp_paths = []
#         for f in files:
#             temp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.filename)[1])
#             temp.write(await f.read())
#             temp.close()
#             tmp_paths.append(temp.name)

#         orchestrator = MultiAgentOrchestrator(verbose=True)
#         result = orchestrator.run_pipeline(company_info, tmp_paths)

#         for p in tmp_paths:
#             os.remove(p)

#         return result

#     except Exception as e:
#         logger.exception("Error running multi-agent pipeline")
#         raise HTTPException(status_code=500, detail=str(e))

#     finally:
#         for path in temp_file_paths:
#             try:
#                 os.remove(path)
#             except Exception as cleanup_error:
#                 logger.warning(f"Could not delete temp file {path}: {cleanup_error}")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
router = APIRouter()
 
#CORS Configuration
origins = [
    "*", #USE AT YOUR OWN RISK
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://executive-multi-agent-frontend.vercel.app", # Add the Vercel URL,
    "https://bb45-2405-201-4021-112e-a09c-f089-f179-f075.ngrok-free.app"  # Replace with your ngrok URL (e.g., "https://your-ngrok-url.ngrok-free.app")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/market_intelligence")
async def get_market_intelligence(
    company_information: str = Form(""),
    supporting_documents: list[UploadFile] = File([])
):
    logger.info(f"Received company_information: {company_information}")
    logger.info(f"Number of supporting documents: {len(supporting_documents)}")
    
    temp_file_paths = []
    loaded_documents_content = []

    try:
        agent = MarketIntelligenceAgent()

        for doc_file in supporting_documents:
            logger.info(f"Processing file: {doc_file.filename} (Content Type: {doc_file.content_type})")
            filename_lower = doc_file.filename.lower()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(doc_file.filename)[-1]) as tmp:
                shutil.copyfileobj(doc_file.file, tmp)
                tmp_path = tmp.name
            temp_file_paths.append(tmp_path) # For cleanup

            try:
                if filename_lower.endswith(".pdf"):
                    text = agent.load_pdf_from_file(tmp_path)
                    loaded_documents_content.append(text)
                elif filename_lower.endswith(".txt"):
                    text = agent.load_text_from_file(tmp_path) # Assuming this method exists or is added
                    loaded_documents_content.append(text)
                elif filename_lower.endswith(".json"):
                    # Replicate JSON loading logic if MarketIntelligenceAgent needs specific handling
                    # or use a generic text loader if it expects a string dump of JSON.
                    # For consistency with its own method, let's use/add one for JSON.
                    if hasattr(agent, '_load_json_as_text_from_file'):
                        text = agent._load_json_as_text_from_file(tmp_path)
                    else: # Fallback to simple read if method not present (though we'll add it)
                        with open(tmp_path, "r", encoding="utf-8") as f:
                            json_data = json.load(f)
                        text = json.dumps(json_data, indent=2)
                    loaded_documents_content.append(text)
                else:
                    logger.warning(f"Unsupported file format for market intelligence: {doc_file.filename}")
            except Exception as e:
                logger.error(f"Could not process file {doc_file.filename} for market intelligence: {e}")

        all_documents_text = "\n\n---SEPARATOR---\n\n".join(filter(None, loaded_documents_content))

        company_details = agent.extract_company_details(company_information)

        market_input = {
            "company_name": company_details.get('company_name', ''),
            "company_location": company_details.get('company_location', ''),
            "geography": company_details.get('geography', ''),
            "supporting_documents": all_documents_text,
            "role": agent.role,
            "goal": agent.goal
        }

        market_report = agent.run(market_input)

        # Optionally save the report server-side
        # report_filename = f"market_report_{uuid.uuid4()}.txt"
        # with open(report_filename, "w", encoding="utf-8") as f:
        #     f.write(market_report)
        # logger.info(f"Market intelligence report saved to {report_filename}")

        return {"market_report": market_report}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Error generating market intelligence report: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating market intelligence report: {str(e)}")
    finally:
        for path in temp_file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as cleanup_error:
                logger.warning(f"Could not delete temp file {path}: {cleanup_error}")

@app.post("/job_description/")
async def create_job_description(
    manual_input: str = Form(""),
    files: List[UploadFile] = File([])
):
    logger.info(f"Received manual_input for JD: {manual_input[:100]}...")
    logger.info(f"Number of uploaded files for JD: {len(files)}")

    temp_file_paths_for_jd = []
    try:
        agent = JobDescriptionWriterAgent(verbose=True)

        for file_obj in files:
            suffix = os.path.splitext(file_obj.filename)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file_obj.file, tmp) # Use shutil.copyfileobj for UploadFile
                tmp_path = tmp.name
            temp_file_paths_for_jd.append(tmp_path)
            logger.info(f"JD Agent: Saved uploaded file to temp path: {tmp_path}")

        # The JobDescriptionWriterAgent's run method expects file_paths
        job_description = agent.run(
            manual_input=manual_input,
            file_paths=temp_file_paths_for_jd
        )

        logger.info("Job description generated successfully.")
        return {"job_description": job_description}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception("Error generating job description")
        raise HTTPException(status_code=500, detail=f"Error generating JD: {str(e)}")
    finally:
        for path in temp_file_paths_for_jd:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"JD Agent: Cleaned up temp file: {path}")
            except Exception as cleanup_error:
                logger.warning(f"JD Agent: Could not delete temp file {path}: {cleanup_error}")


@app.post("/client_creator/")
async def get_client_feedback( # This endpoint seems to be for creating a client *prompt*, not getting feedback
    client_description: Optional[str] = Form(None, description="Free-form client description (persona, values, priorities, tone, etc.)"),
    transcript_files: Optional[List[UploadFile]] = File(None, description="Optional transcript files (PDF, TXT, JSON)")
):
    logger.info("Client Creator: Received request.")
    temp_file_paths = []

    try:
        if not client_description and not transcript_files:
            raise HTTPException(
                status_code=400,
                detail="You must provide either a client_description or transcript_files (or both)."
            )

        if transcript_files:
            for file_obj in transcript_files:
                suffix = os.path.splitext(file_obj.filename)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    shutil.copyfileobj(file_obj.file, tmp)
                    temp_file_paths.append(tmp.name)
                    logger.info(f"Client Creator: Saved transcript file to temp path: {tmp.name}")

        description_text = client_description.strip() if client_description else "Not explicitly described. Please infer from communication tone."

        agent = ClientRepresentativeCreatorAgent(verbose=True)
        generated_prompt = agent.run(
            client_description=description_text,
            transcript_file_paths=temp_file_paths if temp_file_paths else None # Agent expects list of paths
        )
        logger.info("Client Creator: Prompt generated successfully.")
        return {"generated_prompt": generated_prompt}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Error generating client representative prompt: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the prompt: {str(e)}")
    finally:
        for path in temp_file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Client Creator: Cleaned up temp file: {path}")
            except Exception as cleanup_error:
                logger.warning(f"Client Creator: Could not delete temp file {path}: {cleanup_error}")

@app.post("/client_feedback/")
async def create_client_characteristics( # Renamed from create_client_characteristics for clarity, as it generates feedback
    user_input: str = Form(""),
    files: List[UploadFile] = File(default=[])
):
    logger.info("Client Feedback: Received request.")
    temp_file_paths = []

    try:
        if not user_input.strip() and not files:
            raise HTTPException(status_code=400, detail="No input (user_input or files) provided for client feedback.")

        for file_obj in files:
            suffix = os.path.splitext(file_obj.filename)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file_obj.file, tmp)
                temp_file_paths.append(tmp.name)
                logger.info(f"Client Feedback: Saved uploaded file to temp path: {tmp.name}")

        agent = ClientRepresentativeAgent(verbose=True)
        response = agent.run(
            input_statement=user_input,
            transcript_file_paths=temp_file_paths # Agent expects list of paths
        )
        logger.info("Client Feedback: Feedback generated successfully.")
        return {"client_representative_feedback": response}

    except HTTPException as http_exception:
        raise http_exception
    except Exception as e:
        logger.exception("Error processing client feedback (characteristics)")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during client feedback: {str(e)}")
    finally:
        for path in temp_file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Client Feedback: Cleaned up temp file: {path}")
            except Exception as cleanup_error:
                logger.warning(f"Client Feedback: Could not delete temp file {path}: {cleanup_error}")


@app.post("/interview_report")
async def create_interview_report(
    input_text: Optional[str] = Form(
        None,
        description="Optional freeform input from user (e.g. consultant assessment or structured sections with delimiters like ---JOB SPEC---)."
    ),
    files: Optional[List[UploadFile]] = File(
        None,
        description="Optional supporting files (job spec, CV, transcript, etc.) in .pdf, .txt, or .json format."
    )
):
    logger.info("Interview Report: Received request.")
    temp_file_paths = []
    try:
        if not input_text and not files:
            raise HTTPException(
                status_code=400,
                detail="You must provide either input_text, file attachments, or both for interview report."
            )

        attachment_paths_for_agent = []
        if files:
            for upload_file in files:
                suffix = os.path.splitext(upload_file.filename)[-1]
                # Create temp file and ensure it's written to before getting its name for the agent
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file_obj:
                    shutil.copyfileobj(upload_file.file, tmp_file_obj)
                    # tmp_file_obj is closed upon exiting 'with', but file still exists due to delete=False
                    temp_file_paths.append(tmp_file_obj.name) # Add to list for cleanup
                    attachment_paths_for_agent.append(tmp_file_obj.name) # Add to list for agent
                logger.info(f"Interview Report: Saved uploaded file to temp path: {temp_file_paths[-1]}")


        agent = InterviewReportCreatorAgent(verbose=True)
        report = agent.run(
            input_text=input_text or "",
            attachment_paths=attachment_paths_for_agent if attachment_paths_for_agent else None
        )
        logger.info("Interview Report: Report generated successfully.")
        return {"interview_report": report}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Error creating interview report: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred while generating the interview report: {str(e)}")
    finally:
        for path in temp_file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Interview Report: Cleaned up temp file: {path}")
            except Exception as cleanup_error:
                logger.warning(f"Interview Report: Could not delete temp file {path}: {cleanup_error}")

@app.post("/multi_agent_run/")
async def multi_agent_pipeline(
    company_info: str = Form(...),
    files: List[UploadFile] = File([])
):
    logger.info("Multi-Agent Pipeline: Received request.")
    temp_file_paths = [] # Initialize for the finally block
    try:
        # Save uploaded files to temporary paths
        for f in files:
            suffix = os.path.splitext(f.filename)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file_obj:
                shutil.copyfileobj(f.file, tmp_file_obj)
                temp_file_paths.append(tmp_file_obj.name)
            logger.info(f"Multi-Agent Pipeline: Saved uploaded file to temp path: {temp_file_paths[-1]}")

        orchestrator = MultiAgentOrchestrator(verbose=True)
        result = orchestrator.run_pipeline(company_info, temp_file_paths) # Pass list of paths

        logger.info("Multi-Agent Pipeline: Pipeline executed successfully.")
        return result

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception("Error running multi-agent pipeline")
        raise HTTPException(status_code=500, detail=f"Error in multi-agent pipeline: {str(e)}")
    finally:
        for path in temp_file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Multi-Agent Pipeline: Cleaned up temp file: {path}")
            except Exception as cleanup_error:
                logger.warning(f"Multi-Agent Pipeline: Could not delete temp file {path}: {cleanup_error}")

# app.include_router(router) # If you are using APIRouter for organization, uncomment and define routes on 'router'
# If not, @app.post decorator is fine.

# The CORS middleware was added twice. Removed one. It's correctly placed after FastAPI app initialization.