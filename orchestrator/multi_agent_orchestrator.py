# from agents.client_representative_agent import ClientRepresentativeAgent
# from agents.client_representative_creator_agent import ClientRepresentativeCreatorAgent
# from agents.interview_report_creator_agent import InterviewReportCreatorAgent
# from agents.job_description_writer_agent import JobDescriptionWriterAgent
# from agents.market_intelligence_agent import MarketIntelligenceAgent
# import logging, os

# logger = logging.getLogger(__name__)

# class MultiAgentOrchestrator:
#     def __init__(self, verbose=False):
#         self.verbose = verbose
#         self.market_agent = MarketIntelligenceAgent(verbose=verbose)
#         self.client_prompt_agent = ClientRepresentativeCreatorAgent(verbose=verbose)
#         self.jd_agent = JobDescriptionWriterAgent(verbose=verbose)
#         self.interview_agent = InterviewReportCreatorAgent(verbose=verbose)
#         self.client_feedback_agent = ClientRepresentativeAgent(verbose=verbose)
#         if self.verbose:
#             logger.info("MultiAgentOrchestrator initialized.")

#     def run_pipeline(self, company_info: str, file_paths: list) -> dict:
#         results = {}
#         if self.verbose:
#             logger.info(f"Orchestrator: Starting pipeline with company_info and {len(file_paths)} files.")

#         # 1. Market Intelligence Agent
#         if self.verbose: logger.info("Orchestrator: Running MarketIntelligenceAgent...")
#         company_details = self.market_agent.extract_company_details(company_info)
        
#         loaded_docs_texts_market = []
#         for f_path in file_paths:
#             filename_lower = f_path.lower()
#             content = None
#             try:
#                 if filename_lower.endswith(".pdf"):
#                     content = self.market_agent.load_pdf_from_file(f_path)
#                 elif filename_lower.endswith(".txt"):
#                     content = self.market_agent.load_text_from_file(f_path)
#                 elif filename_lower.endswith(".json"):
#                     content = self.market_agent._load_json_as_text_from_file(f_path) # Use the new method
#                 else:
#                     if self.verbose:
#                         logger.warning(f"Orchestrator (MarketAgent): Unsupported file type for {f_path}, skipping.")
                
#                 if content:
#                     loaded_docs_texts_market.append(content)
#             except Exception as e:
#                 logger.error(f"Orchestrator (MarketAgent): Error loading file {f_path}: {e}")

#         all_documents_text_for_market = "\n\n---SEPARATOR---\n\n".join(filter(None, loaded_docs_texts_market))

#         market_input = {
#             "company_name": company_details.get("company_name", "Unknown"),
#             "company_location": company_details.get("company_location", "Unknown"),
#             "geography": company_details.get("geography", "Unknown"),
#             "supporting_documents": all_documents_text_for_market,
#             "role": self.market_agent.role,
#             "goal": self.market_agent.goal
#         }
#         market_report = self.market_agent.run(market_input)
#         results["market_report"] = market_report
#         if self.verbose: logger.info("Orchestrator: MarketIntelligenceAgent finished.")

#         # 2. Client Representative Prompt Creator Agent
#         if self.verbose: logger.info("Orchestrator: Running ClientRepresentativeCreatorAgent...")
#         # It can use the market report as a description and original files for tone.
#         client_prompt = self.client_prompt_agent.run(
#             client_description=market_report, # Using market_report as initial client description
#             transcript_file_paths=file_paths # Original files can help infer tone
#         )
#         results["client_prompt"] = client_prompt
#         if self.verbose: logger.info("Orchestrator: ClientRepresentativeCreatorAgent finished.")

#         # 3. Job Description Writer Agent
#         if self.verbose: logger.info("Orchestrator: Running JobDescriptionWriterAgent...")
#         # Uses the generated client prompt as manual input and original files as supporting docs
#         jd = self.jd_agent.run(
#             manual_input=client_prompt, # Or could be a combination, e.g., market_report + client_prompt
#             file_paths=file_paths
#         )
#         results["job_description"] = jd
#         if self.verbose: logger.info("Orchestrator: JobDescriptionWriterAgent finished.")

#         # 4. Interview Report Creator Agent
#         if self.verbose: logger.info("Orchestrator: Running InterviewReportCreatorAgent...")
#         # Uses the JD as input text, and original files might contain CVs/transcripts
#         # This assumes 'file_paths' might contain CV, interview notes etc.
#         # The 'input_text' for interview agent could be the JD, or a placeholder if files are primary.
#         report = self.interview_agent.run(
#             input_text=f"Job Description:\n{jd}\n\nOther relevant information might be in attached files.", # Pass JD to it
#             attachment_paths=file_paths
#         )
#         results["interview_report"] = report
#         if self.verbose: logger.info("Orchestrator: InterviewReportCreatorAgent finished.")

#         # 5. Client Representative Feedback Agent
#         if self.verbose: logger.info("Orchestrator: Running ClientRepresentativeAgent for feedback...")
#         # Gets feedback on the generated interview report, using original files for context.
#         # The input_statement for client feedback agent is the Interview Report.
#         # The client's characteristics (persona, tone) will be inferred by this agent from the market_report or client_prompt
#         # and/or the original files.
#         # Let's use the client_prompt (which embodies client persona) + interview_report to get feedback
#         feedback_input_statement = f"---CLIENT PERSONA GUIDANCE---\n{client_prompt}\n\n---DOCUMENT TO REVIEW---\n{report}"

#         feedback = self.client_feedback_agent.run(
#             input_statement=feedback_input_statement, # The report to be reviewed, prefixed with persona
#             transcript_file_paths=file_paths # Files to help infer client persona if not fully in input_statement
#         )
#         results["client_feedback"] = feedback
#         if self.verbose: logger.info("Orchestrator: ClientRepresentativeAgent for feedback finished.")
        
#         if self.verbose: logger.info("Orchestrator: Pipeline completed.")
#         return results

from agents.client_representative_agent import ClientRepresentativeAgent
from agents.client_representative_creator_agent import ClientRepresentativeCreatorAgent
from agents.interview_report_creator_agent import InterviewReportCreatorAgent
from agents.job_description_writer_agent import JobDescriptionWriterAgent
from agents.market_intelligence_agent import MarketIntelligenceAgent
import logging, os

logger = logging.getLogger(__name__)

# Helper function (can be in utils.py) - assuming agents use their own internal or imported version
def is_orchestrator_input_valid(text: str, min_length: int = 10) -> bool:
    if not text or not text.strip() or len(text.strip()) < min_length:
        return False
    # Add more sophisticated checks if needed for orchestrator-level inputs
    return True

class MultiAgentOrchestrator:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.market_agent = MarketIntelligenceAgent(verbose=verbose)
        self.client_prompt_agent = ClientRepresentativeCreatorAgent(verbose=verbose)
        self.jd_agent = JobDescriptionWriterAgent(verbose=verbose)
        self.interview_agent = InterviewReportCreatorAgent(verbose=verbose)
        self.client_feedback_agent = ClientRepresentativeAgent(verbose=verbose)
        if self.verbose:
            logger.info("MultiAgentOrchestrator initialized with all sub-agents.")

    def _read_content_from_file_paths(self, agent_file_loader_methods: dict, file_paths: list) -> str:
        loaded_texts = []
        if not file_paths: return ""
        for f_path in file_paths:
            if not os.path.exists(f_path):
                logger.warning(f"Orchestrator: File {f_path} does not exist, skipping.")
                continue
            filename_lower = f_path.lower()
            ext = os.path.splitext(filename_lower)[-1]
            content = None
            try:
                if ext in agent_file_loader_methods:
                    loader_method = agent_file_loader_methods[ext]
                    content = loader_method(f_path) # Agent methods should strip
                elif self.verbose: 
                    logger.warning(f"Orchestrator: No specific loader for {ext} for current agent step, file {f_path} might not be used by this agent's direct loading.")
                
                if content and content.strip(): # Ensure content is not just whitespace
                    base_name = os.path.basename(f_path)
                    # Agents should add their own headers if needed, orchestrator just combines
                    loaded_texts.append(content) 
            except Exception as e:
                logger.error(f"Orchestrator: Error loading file {f_path} for an agent: {e}")
        
        return "\n\n---FILE_SEPARATOR_ORCHESTRATOR---\n\n".join(filter(None, loaded_texts)).strip()


    def run_pipeline(self, company_info: str, file_paths: list) -> dict:
        results = {}
        if self.verbose:
            logger.info(f"Orchestrator: Starting pipeline with company_info (len {len(company_info)}) and {len(file_paths)} files: {file_paths}")

        if not is_orchestrator_input_valid(company_info, min_length=3): # Company info query can be short
            logger.error("Orchestrator: Initial 'company_info' is invalid.")
            results["pipeline_error"] = "Error: Initial 'company_info' is insufficient or non-meaningful."
            return results

        # Step 1: Market Intelligence Agent
        if self.verbose: logger.info("Orchestrator: === Step 1: MarketIntelligenceAgent ===")
        company_details_from_info = self.market_agent.extract_company_details(company_info) # Agent validates internally
        if company_details_from_info.get("company_name", "Unknown").lower() == "unknown" and not file_paths :
             logger.warning("Orchestrator: Could not extract company name from 'company_info' and no files provided for Market Agent.")
             # Allow agent to run and potentially return its own error or generic report.

        market_agent_loaders = {
            ".pdf": self.market_agent.load_pdf_from_file,
            ".txt": self.market_agent.load_text_from_file,
            ".json": self.market_agent._load_json_as_text_from_file
        }
        all_docs_text_market = self._read_content_from_file_paths(market_agent_loaders, file_paths)
        
        market_input_payload = {
            "company_name": company_details_from_info.get("company_name", "Unknown"),
            "company_location": company_details_from_info.get("company_location", "Unknown"),
            "geography": company_details_from_info.get("geography", "Unknown"),
            "supporting_documents": all_docs_text_market,
        }
        market_report = self.market_agent.run(market_input_payload)
        results["market_report"] = market_report
        if market_report.startswith("Error:"):
            logger.error(f"Orchestrator: MarketIntelligenceAgent failed: {market_report}")
            results["market_report_error"] = market_report # Store specific error
            return results # Stop pipeline
        if self.verbose: logger.info(f"Orchestrator: MarketIntelligenceAgent finished. Report length: {len(market_report)}")

        # Step 2: Client Representative Prompt Creator Agent
        if self.verbose: logger.info("Orchestrator: === Step 2: ClientRepresentativeCreatorAgent ===")
        client_prompt = self.client_prompt_agent.run(client_description=market_report, transcript_file_paths=file_paths)
        results["client_prompt"] = client_prompt
        if client_prompt.startswith("Error:"):
            logger.error(f"Orchestrator: ClientRepresentativeCreatorAgent failed: {client_prompt}")
            results["client_prompt_error"] = client_prompt
            return results
        if self.verbose: logger.info(f"Orchestrator: ClientRepresentativeCreatorAgent finished. Prompt length: {len(client_prompt)}")

        # Step 3: Job Description Writer Agent
        if self.verbose: logger.info("Orchestrator: === Step 3: JobDescriptionWriterAgent ===")
        jd = self.jd_agent.run(manual_input=client_prompt, file_paths=file_paths)
        results["job_description"] = jd
        if jd.startswith("Error:"):
            logger.error(f"Orchestrator: JobDescriptionWriterAgent failed: {jd}")
            results["job_description_error"] = jd
            return results
        if self.verbose: logger.info(f"Orchestrator: JobDescriptionWriterAgent finished. JD length: {len(jd)}")

        # Step 4: Interview Report Creator Agent
        if self.verbose: logger.info("Orchestrator: === Step 4: InterviewReportCreatorAgent ===")
        interview_input_text = f"---JOB SPEC---\n{jd}\n\n---CLIENT CONTEXT---\n{client_prompt}\n\n---CONSULTANT ASSESSMENT---\n[Consultant should provide notes here if any, or rely on information from attached files for candidate details, interview transcripts etc.]"
        report = self.interview_agent.run(input_text=interview_input_text, attachment_paths=file_paths)
        results["interview_report"] = report
        if report.startswith("Error:"):
            logger.error(f"Orchestrator: InterviewReportCreatorAgent failed: {report}")
            results["interview_report_error"] = report
            return results
        if self.verbose: logger.info(f"Orchestrator: InterviewReportCreatorAgent finished. Report length: {len(report)}")

        # Step 5: Client Representative Feedback Agent
        if self.verbose: logger.info("Orchestrator: === Step 5: ClientRepresentativeAgent for feedback ===")
        feedback_input_statement = f"---CLIENT PERSONA GUIDANCE---\n{client_prompt}\n\n---DOCUMENT TO REVIEW---\n{report}"
        feedback = self.client_feedback_agent.run(input_statement=feedback_input_statement, transcript_file_paths=file_paths)
        results["client_feedback"] = feedback
        if feedback.startswith("Error:"): # This agent might return an error string
            logger.error(f"Orchestrator: ClientRepresentativeAgent (feedback) failed: {feedback}")
            results["client_feedback_error"] = feedback
            # Not necessarily stopping the whole pipeline for a feedback error, but good to note.
            # For consistency, let's also return here if it fails.
            return results 
        if self.verbose: logger.info(f"Orchestrator: ClientRepresentativeAgent for feedback finished. Feedback length: {len(feedback)}")
        
        if self.verbose: logger.info("Orchestrator: Pipeline completed successfully.")
        return results