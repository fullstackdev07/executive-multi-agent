# from agents.client_representative_agent import ClientRepresentativeAgent
# from agents.client_representative_creator_agent import ClientRepresentativeCreatorAgent
# from agents.interview_report_creator_agent import InterviewReportCreatorAgent
# from agents.job_description_writer_agent import JobDescriptionWriterAgent
# from agents.market_intelligence_agent import MarketIntelligenceAgent

# class MultiAgentOrchestrator:
#     def __init__(self, verbose=False):
#         self.verbose = verbose
#         self.market_agent = MarketIntelligenceAgent(verbose=verbose)
#         self.client_prompt_agent = ClientRepresentativeCreatorAgent(verbose=verbose)
#         self.jd_agent = JobDescriptionWriterAgent(verbose=verbose)
#         self.interview_agent = InterviewReportCreatorAgent(verbose=verbose)
#         self.client_feedback_agent = ClientRepresentativeAgent(verbose=verbose)

#     def run_pipeline(self, company_info: str, files: list) -> dict:
#         results = {}

#         company_details = self.market_agent.extract_company_details(company_info)
#         market_input = {
#             "company_name": company_details.get("company_name"),
#             "company_location": company_details.get("company_location"),
#             "geography": company_details.get("geography"),
#             "supporting_documents": "\n".join([self.market_agent.load_text_from_file(f) for f in files]),
#             "role": self.market_agent.role,
#             "goal": self.market_agent.goal
#         }
#         market_report = self.market_agent.run(market_input)
#         results["market_report"] = market_report

#         client_prompt = self.client_prompt_agent.run(
#             client_description=market_report,
#             transcript_file_paths=files
#         )
#         results["client_prompt"] = client_prompt

#         jd = self.jd_agent.run(manual_input=client_prompt, file_paths=files)
#         results["job_description"] = jd

#         report = self.interview_agent.run(input_text=jd, attachment_paths=files)
#         results["interview_report"] = report

#         feedback = self.client_feedback_agent.run(
#             input_statement=report,
#             transcript_file_paths=files
#         )
#         results["client_feedback"] = feedback

#         return results

from agents.client_representative_agent import ClientRepresentativeAgent
from agents.client_representative_creator_agent import ClientRepresentativeCreatorAgent
from agents.interview_report_creator_agent import InterviewReportCreatorAgent
from agents.job_description_writer_agent import JobDescriptionWriterAgent
from agents.market_intelligence_agent import MarketIntelligenceAgent
import logging, os

logger = logging.getLogger(__name__)

class MultiAgentOrchestrator:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.market_agent = MarketIntelligenceAgent(verbose=verbose)
        self.client_prompt_agent = ClientRepresentativeCreatorAgent(verbose=verbose)
        self.jd_agent = JobDescriptionWriterAgent(verbose=verbose)
        self.interview_agent = InterviewReportCreatorAgent(verbose=verbose)
        self.client_feedback_agent = ClientRepresentativeAgent(verbose=verbose)
        if self.verbose:
            logger.info("MultiAgentOrchestrator initialized.")

    def run_pipeline(self, company_info: str, file_paths: list) -> dict:
        results = {}
        if self.verbose:
            logger.info(f"Orchestrator: Starting pipeline with company_info and {len(file_paths)} files.")

        # 1. Market Intelligence Agent
        if self.verbose: logger.info("Orchestrator: Running MarketIntelligenceAgent...")
        company_details = self.market_agent.extract_company_details(company_info)
        
        loaded_docs_texts_market = []
        for f_path in file_paths:
            filename_lower = f_path.lower()
            content = None
            try:
                if filename_lower.endswith(".pdf"):
                    content = self.market_agent.load_pdf_from_file(f_path)
                elif filename_lower.endswith(".txt"):
                    content = self.market_agent.load_text_from_file(f_path)
                elif filename_lower.endswith(".json"):
                    content = self.market_agent._load_json_as_text_from_file(f_path) # Use the new method
                else:
                    if self.verbose:
                        logger.warning(f"Orchestrator (MarketAgent): Unsupported file type for {f_path}, skipping.")
                
                if content:
                    loaded_docs_texts_market.append(content)
            except Exception as e:
                logger.error(f"Orchestrator (MarketAgent): Error loading file {f_path}: {e}")

        all_documents_text_for_market = "\n\n---SEPARATOR---\n\n".join(filter(None, loaded_docs_texts_market))

        market_input = {
            "company_name": company_details.get("company_name", "Unknown"),
            "company_location": company_details.get("company_location", "Unknown"),
            "geography": company_details.get("geography", "Unknown"),
            "supporting_documents": all_documents_text_for_market,
            "role": self.market_agent.role,
            "goal": self.market_agent.goal
        }
        market_report = self.market_agent.run(market_input)
        results["market_report"] = market_report
        if self.verbose: logger.info("Orchestrator: MarketIntelligenceAgent finished.")

        # 2. Client Representative Prompt Creator Agent
        if self.verbose: logger.info("Orchestrator: Running ClientRepresentativeCreatorAgent...")
        # It can use the market report as a description and original files for tone.
        client_prompt = self.client_prompt_agent.run(
            client_description=market_report, # Using market_report as initial client description
            transcript_file_paths=file_paths # Original files can help infer tone
        )
        results["client_prompt"] = client_prompt
        if self.verbose: logger.info("Orchestrator: ClientRepresentativeCreatorAgent finished.")

        # 3. Job Description Writer Agent
        if self.verbose: logger.info("Orchestrator: Running JobDescriptionWriterAgent...")
        # Uses the generated client prompt as manual input and original files as supporting docs
        jd = self.jd_agent.run(
            manual_input=client_prompt, # Or could be a combination, e.g., market_report + client_prompt
            file_paths=file_paths
        )
        results["job_description"] = jd
        if self.verbose: logger.info("Orchestrator: JobDescriptionWriterAgent finished.")

        # 4. Interview Report Creator Agent
        if self.verbose: logger.info("Orchestrator: Running InterviewReportCreatorAgent...")
        # Uses the JD as input text, and original files might contain CVs/transcripts
        # This assumes 'file_paths' might contain CV, interview notes etc.
        # The 'input_text' for interview agent could be the JD, or a placeholder if files are primary.
        report = self.interview_agent.run(
            input_text=f"Job Description:\n{jd}\n\nOther relevant information might be in attached files.", # Pass JD to it
            attachment_paths=file_paths
        )
        results["interview_report"] = report
        if self.verbose: logger.info("Orchestrator: InterviewReportCreatorAgent finished.")

        # 5. Client Representative Feedback Agent
        if self.verbose: logger.info("Orchestrator: Running ClientRepresentativeAgent for feedback...")
        # Gets feedback on the generated interview report, using original files for context.
        # The input_statement for client feedback agent is the Interview Report.
        # The client's characteristics (persona, tone) will be inferred by this agent from the market_report or client_prompt
        # and/or the original files.
        # Let's use the client_prompt (which embodies client persona) + interview_report to get feedback
        feedback_input_statement = f"---CLIENT PERSONA GUIDANCE---\n{client_prompt}\n\n---DOCUMENT TO REVIEW---\n{report}"

        feedback = self.client_feedback_agent.run(
            input_statement=feedback_input_statement, # The report to be reviewed, prefixed with persona
            transcript_file_paths=file_paths # Files to help infer client persona if not fully in input_statement
        )
        results["client_feedback"] = feedback
        if self.verbose: logger.info("Orchestrator: ClientRepresentativeAgent for feedback finished.")
        
        if self.verbose: logger.info("Orchestrator: Pipeline completed.")
        return results