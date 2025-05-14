from agents.market_intelligence_agent import MarketIntelligenceAgent
from agents.job_description_writer_agent import JobDescriptionWriterAgent
from agents.client_representative_agent import ClientRepresentativeAgent
import os
from dotenv import load_dotenv
import json

load_dotenv()

def main():
    market_agent = MarketIntelligenceAgent(verbose=True)
    conversation_filepath = 'conversation.txt'

    market_conversation = market_agent.load_conversation_from_file(conversation_filepath)

    market_input = {
        "conversation": market_conversation,
        "role": market_agent.role,
        "goal": market_agent.goal
    }
    market_report = market_agent.run(market_input)
    print("\nMarket Intelligence Report:\n", market_report)

    jd_agent = JobDescriptionWriterAgent(verbose=True)
    call_transcript_filepath = 'transcript.json' 
    call_transcript = jd_agent.load_call_transcript_from_json(call_transcript_filepath)

    job_details = jd_agent.extract_job_details(call_transcript)
    job_title = job_details.get('job_title', 'Unknown Job Title')
    client_name = job_details.get('client_name', 'Unknown Client')
    job_spec = job_details.get('job_spec', 'No Job Specification')
    additional_documents = job_details.get('additional_documents', 'No additional documents')

    jd_input = {
        "job_title": job_title,
        "client_name": client_name,
        "market_intelligence": market_report, 
        "call_transcript": call_transcript,
        "job_spec": job_spec,
        "additional_documents": additional_documents,
        "role": jd_agent.role,
        "goal": jd_agent.goal
    }
    job_description = jd_agent.run(jd_input)
    print("\nJob Description:\n", job_description)

    client_rep_agent = ClientRepresentativeAgent(verbose=True)
    conversation_filepath = 'transcript.json'
    f = open(conversation_filepath)
    conversation = json.load(f)
    client_info = client_rep_agent.extract_client_information(str(conversation))

    client_rep_input = {
        "client_name": client_info["client_name"],
        "client_title": client_info["client_title"],
        "client_characteristics": client_info["client_characteristics"],
        "document": job_description, 
        "role": client_rep_agent.role,
        "goal": client_rep_agent.goal
    }
    client_feedback = client_rep_agent.run(client_rep_input)
    print("\nClient Feedback:\n", client_feedback)

if __name__ == "__main__":
    main()