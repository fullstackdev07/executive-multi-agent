from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter
import logging
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import tempfile
from typing import Optional, List
from agents.client_representative_agent import ClientRepresentativeAgent
from agents.client_representative_creator_agent import ClientRepresentativeCreatorAgent
from agents.interview_report_creator_agent import InterviewReportCreatorAgent
from agents.job_description_writer_agent import JobDescriptionWriterAgent
from agents.market_intelligence_agent import MarketIntelligenceAgent
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
import shutil
import os
import uuid
import tempfile

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
    print(f"Received company_information: {company_information}")
    print(f"Number of supporting documents: {len(supporting_documents)}")
    for doc in supporting_documents:
        print(f"Received file: {doc.filename} (Content Type: {doc.content_type})")

    try:

        # Initialize the agent
        agent = MarketIntelligenceAgent()

        loaded_documents = []

        for doc in supporting_documents:
            filename = doc.filename.lower()

            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(doc.filename)[-1]) as tmp:
                tmp.write(await doc.read())
                tmp_path = tmp.name

            try:
                if filename.endswith(".pdf"):
                    text = agent.load_pdf_from_file(tmp_path)
                    loaded_documents.append(text)

                elif filename.endswith(".txt"):
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    loaded_documents.append(text)

                elif filename.endswith(".json"):
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                    text = json.dumps(json_data, indent=2)  # Pretty print for LLM readability
                    loaded_documents.append(text)

                else:
                    logger.warning(f"Unsupported file format: {doc.filename}")

            except Exception as e:
                logger.warning(f"Could not process file {doc.filename}: {e}")

        # Combine all loaded documents into one string
        all_documents_text = "\n".join(loaded_documents)

        # Extract company details
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

        with open("market_report.txt", "w", encoding="utf-8") as f:
            f.write(market_report)

        logger.info("Market intelligence report saved to market_report.txt")

        return {"market_report": market_report}

    except Exception as e:
        logger.exception(f"Error generating market intelligence report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/job_description/")
async def create_job_description(
    manual_input: str = Form(""),
    files: List[UploadFile] = File([])
):
    logger.info(f"Received manual_input: {manual_input[:100]}...")
    logger.info(f"Number of uploaded files: {len(files)}")

    try:
        agent = JobDescriptionWriterAgent(verbose=True)
        loaded_documents = []

        for file in files:
            filename = file.filename.lower()
            suffix = os.path.splitext(filename)[-1]

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            try:
                label = f"\n--- File: {filename} ---\n"

                if filename.endswith(".pdf"):
                    text = agent._extract_text_from_pdf(tmp_path)
                elif filename.endswith(".txt"):
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        text = f.read()
                elif filename.endswith(".json"):
                    text = agent._extract_transcript_from_json(tmp_path)
                else:
                    logger.warning(f"Unsupported file format: {filename}")
                    continue

                loaded_documents.append(label + text)

            except Exception as e:
                logger.warning(f"Could not process file {filename}: {e}")

        # Combine all text segments
        combined_text = "\n".join(loaded_documents).strip()

        job_description = agent.chain.run({
            "manual_input": manual_input,
            "file_text": combined_text
        })

        logger.info("Job description generated successfully.")
        return {"job_description": job_description}

    except Exception as e:
        logger.exception("Error generating job description")
        raise HTTPException(status_code=500, detail=f"Error generating JD: {str(e)}")

@app.post("/client_creator")
async def get_client_feedback(
    client_description: Optional[str] = Form(None, description="Free-form client description (persona, values, priorities, tone, etc.)"),
    transcript_files: Optional[List[UploadFile]] = File(None, description="Optional transcript files (PDF, TXT, JSON)")
):
    temp_file_paths = []

    try:
        # Validate input: at least one input
        if not client_description and not transcript_files:
            raise HTTPException(
                status_code=400,
                detail="You must provide either a client_description or transcript_files (or both)."
            )

        # Save uploaded transcript files
        if transcript_files:
            for file in transcript_files:
                contents = await file.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
                    tmp.write(contents)
                    temp_file_paths.append(tmp.name)

        # Use fallback text if client_description is missing
        description_text = client_description.strip() if client_description else "Not explicitly described. Please infer from communication tone."

        # Run the agent
        agent = ClientRepresentativeCreatorAgent(verbose=True)
        generated_prompt = agent.run(
            client_description=description_text,
            transcript_file_paths=temp_file_paths if temp_file_paths else None
        )

        return {"generated_prompt": generated_prompt}

    except Exception as e:
        logger.exception(f"Error generating client representative prompt: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the prompt.")

    finally:
        # Clean up temp files
        for path in temp_file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as cleanup_error:
                logger.warning(f"Could not delete temp file {path}: {cleanup_error}")


@app.post("/client_feedback/")
async def create_client_characteristics(
    user_input: str = Form(""),
    files: List[UploadFile] = File(default=[])
):
    try:
        file_contents = {}
        for file in files:
            try:
                content = await file.read()
                file_contents[file.filename] = content
            except Exception as e:
                logger.warning(f"Could not read {file.filename}: {e}")

        temp_paths = []
        for filename, content in file_contents.items():
            ext = filename.split('.')[-1].lower()
            if ext in ["txt", "pdf", "json"]:
                path = f"{filename}"
                with open(path, "wb") as f:
                    f.write(content)
                temp_paths.append(path)

        agent = ClientRepresentativeAgent(verbose=True)

        # Pass files if they exist, else None
        if temp_paths:
            response = agent.run(input_statement=user_input.strip(), files=temp_paths)
        else:
            response = agent.run(input_statement=user_input.strip())

        return {"client_representative_feedback": response}

    except Exception as e:
        logger.exception(f"Error processing client characteristics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interview_report/")
async def create_interview_report(
    manual_input: Optional[str] = Form(None, description="Structured input with ---JOB SPEC---, ---CANDIDATE CV---, ---INTERVIEW TRANSCRIPT---"),
    files: Optional[List[UploadFile]] = File(None, description="Job spec, CV, or transcript files (.pdf, .txt, .json)")
):
    temp_file_paths = []

    try:
        # Validate input presence
        if not manual_input and not files:
            raise HTTPException(
                status_code=400,
                detail="You must provide either manual_input or file attachments, or both."
            )

        attachment_paths = []

        if files:
            for upload_file in files:
                content_bytes = await upload_file.read()
                suffix = os.path.splitext(upload_file.filename)[-1]
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp_file.write(content_bytes)
                tmp_file.close()
                temp_file_paths.append(tmp_file.name)
                attachment_paths.append(tmp_file.name)

        # Initialize and run the agent
        agent = InterviewReportCreatorAgent(verbose=True)
        report = agent.run(
            manual_input=manual_input or "",
            attachment_paths=attachment_paths if attachment_paths else None
        )

        return {"interview_report": report}

    except Exception as e:
        logger.exception(f"Error creating interview report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for path in temp_file_paths:
            try:
                os.remove(path)
            except Exception as cleanup_error:
                logger.warning(f"Could not delete temp file {path}: {cleanup_error}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
