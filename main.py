from fastapi import FastAPI, File, UploadFile, HTTPException
import logging
from fastapi.middleware.cors import CORSMiddleware
import json
# Import the agent modules
from agents.client_representative_agent import ClientRepresentativeAgent
from agents.client_representative_creator_agent import ClientRepresentativeCreatorAgent
from agents.interview_report_creator_agent import InterviewReportCreatorAgent
from agents.job_description_writer_agent import JobDescriptionWriterAgent
from agents.market_intelligence_agent import MarketIntelligenceAgent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

#CORS Configuration
origins = [
    "*", #USE AT YOUR OWN RISK
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://executive-multi-agent-frontend.vercel.app", # Add the Vercel URL,
    "https://45c9-2405-201-4021-112e-1c1-1dcb-3765-8318.ngrok-free.app"  # Replace with your ngrok URL (e.g., "https://your-ngrok-url.ngrok-free.app")
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
async def get_market_intelligence(conversation_file: UploadFile = File(...)):
    print("Market Intelligence")
    try:
        # Read the content directly from the file
        content = await conversation_file.read()
        conversation_text = content.decode("utf-8") #Decode file

        # Run the agent
        agent = MarketIntelligenceAgent()
        market_report = agent.run(conversation_text)

        # Print the output
        print(f"Market Intelligence Report: {market_report}")

        # Save the output to a file
        output_file_path = "market_report.txt"  # Adjust the filename as needed
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(market_report)
        logger.info(f"Market intelligence report saved to {output_file_path}")

        # Return result
        return {"market_report": market_report}

    except Exception as e:
        logger.exception(f"Error generating market intelligence report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/job_description/")
async def create_job_description(market_report_file: UploadFile = File(...), transcript_file: UploadFile = File(...)):
    try:
        # Read the market report from the file
        market_report_content = await market_report_file.read()
        market_intelligence = market_report_content.decode("utf-8")

        # Read the transcript content from the file
        transcript_content = await transcript_file.read()
        call_transcript = transcript_content.decode("utf-8")

        # Run the agent with the market report and transcript
        agent = JobDescriptionWriterAgent()

        # Extract details from the transcript
        job_details = agent.extract_job_details(call_transcript)
        job_title = job_details.get('job_title', 'Unknown Job Title')
        client_name = job_details.get('client_name', 'Unknown Client')
        job_spec = job_details.get('job_spec', 'No Job Specification')
        additional_documents = job_details.get('additional_documents', 'No additional documents')

        jd_input = {
            "job_title": job_title,
            "client_name": client_name,
            "market_intelligence": market_intelligence,
            "call_transcript": call_transcript,
            "job_spec": job_spec,
            "additional_documents": additional_documents,
            "role": agent.role,
            "goal": agent.goal
        }

        job_description = agent.run(jd_input)

        return {"job_description": job_description}

    except Exception as e:
        logger.exception(f"Error creating job description: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/client_feedback")
async def get_client_feedback(
    job_description_file: UploadFile = File(...),
    transcript_file: UploadFile = File(...)
):
    try:
        # Read the job description from the file
        job_description_content = await job_description_file.read()
        job_description = job_description_content.decode("utf-8")

        # Read and parse the transcript JSON file
        transcript_content = await transcript_file.read()
        try:
            transcript_data = json.loads(transcript_content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in transcript_file: {e}")

        # Convert the transcript json data into string
        transcript = json.dumps(transcript_data)

        agent = ClientRepresentativeAgent()
        client_info = agent.extract_client_information(transcript)

        client_feedback = agent.run(
            client_info["client_name"],
            client_info["client_title"],
            client_info["client_characteristics"],
            job_description
        )
        return {"client_feedback": client_feedback}
    except Exception as e:
        logger.exception(f"Error getting client feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interview_report/")
async def create_interview_report(
    job_description_file: UploadFile = File(...),
    candidate_cv_file: UploadFile = File(...),
    interview_transcript_file: UploadFile = File(...)
):
    try:
        # Read the content of each file
        job_description_content = await job_description_file.read()
        job_description = job_description_content.decode("utf-8")

        candidate_cv_content = await candidate_cv_file.read()
        candidate_cv = candidate_cv_content.decode("utf-8")

        interview_transcript_content = await interview_transcript_file.read()
        interview_transcript = interview_transcript_content.decode("utf-8")

        # Create the interview report using the agent
        agent = InterviewReportCreatorAgent()
        report = agent.run(
            job_description,
            candidate_cv,
            interview_transcript,
            "" # leave the assessor value empty
        )
        return {"interview_report": report}

    except Exception as e:
        logger.exception(f"Error creating interview report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/client_characteristics/")
async def create_client_characteristics(user_input_file: UploadFile = File(...)):
    try:
        # Read the content from the file
        content = await user_input_file.read()
        try:
            json_data = json.loads(content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")

        # Extract the user input from the JSON
        # You'll need to adapt this based on your expected JSON structure
        user_input = json_data.get("user_input")

        if user_input is None:
            raise HTTPException(status_code=400, detail="Missing 'user_input' key in the JSON")

        agent = ClientRepresentativeCreatorAgent()
        response = agent.run(user_input)  # Pass user input

        return {"response": response, "conversation_history": agent.get_conversation_history()}
    except Exception as e:
        logger.exception(f"Error creating client characteristics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)