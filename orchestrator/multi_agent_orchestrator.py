# import os
# import json
# import logging
# # Assuming your agents are in the 'agents' directory and importable
# from agents.market_intelligence_agent import MarketIntelligenceAgent
# from agents.client_representative_creator_agent import ClientRepresentativeCreatorAgent
# from agents.job_description_writer_agent import JobDescriptionWriterAgent
# from agents.interview_report_creator_agent import InterviewReportCreatorAgent
# from agents.client_representative_agent import ClientRepresentativeAgent

# logger = logging.getLogger(__name__)

# # Helper function for basic input validation (can be moved to a utils.py)
# def is_orchestrator_input_valid(text: str, min_length: int = 1, allow_empty_ok: bool = False) -> bool:
#     if allow_empty_ok and (text is None or not text.strip()):
#         return True # Empty is fine if allowed
#     if not text or not text.strip() or len(text.strip()) < min_length:
#         return False
#     # Add more sophisticated checks if needed
#     return True


# class MultiAgentOrchestrator:
#     def __init__(self, verbose=False):
#         self.verbose = verbose
#         # self.market_agent = MarketIntelligenceAgent(verbose=verbose)   #REMOVE INSTANCE VARIABLE
#         # self.client_prompt_agent = ClientRepresentativeCreatorAgent(verbose=verbose)
#         # self.jd_agent = JobDescriptionWriterAgent(verbose=verbose)
#         # self.interview_agent = InterviewReportCreatorAgent(verbose=verbose)
#         # self.client_feedback_agent = ClientRepresentativeAgent(verbose=verbose)
#         if self.verbose:
#             logger.info("MultiAgentOrchestrator initialized with all sub-agents.")

#     # --- Helper for loading files (generic enough for most agents) ---
#     def _load_files_for_agent_input(self, agent_instance, file_paths: list) -> str:
#         """
#         Loads files and concatenates their text content.
#         Agents are expected to have their own file reading methods if they need specific parsing.
#         This helper is more for preparing a combined text block of 'supporting documents'
#         if an agent expects that instead of handling individual file paths.
#         However, most of your agents now handle file_paths directly.
#         This method might be more useful if you want to create a single string from files
#         BEFORE passing to an agent that doesn't take file_paths.
#         For now, let's assume agents called by individual runners will use their own loaders via file_paths.
#         """
#         all_texts = []
#         if not file_paths:
#             return ""
            
#         for path in file_paths:
#             if not os.path.exists(path):
#                 logger.warning(f"Orchestrator (_load_files_for_agent_input): File not found {path}, skipping.")
#                 continue
            
#             content = ""
#             ext = os.path.splitext(path)[-1].lower()
            
#             # Try to use agent-specific methods if they exist and are relevant for general text combining
#             # This is a bit tricky because agent loaders are specific.
#             # For simplicity, this helper will do a generic read for now.
#             # The individual agent runner methods should pass `file_paths` to the agent.
#             try:
#                 with open(path, "r", encoding='utf-8', errors='ignore') as f: # errors='ignore' for problematic files
#                     content = f.read()
#                 logger.info(f"Orchestrator (_load_files_for_agent_input): Loaded {path} with generic text read.")
#             except Exception as e:
#                 logger.warning(f"Orchestrator (_load_files_for_agent_input): Could not load {path} generically: {e}")
            
#             if content and content.strip():
#                 all_texts.append(f"--- Content from file: {os.path.basename(path)} ---\n{content.strip()}")
#         return "\n\n".join(all_texts).strip()

#     def run_market_intelligence(self, company_query: str, file_paths: list = None) -> str:
#         """
#         Runs the MarketIntelligenceAgent.
#         Returns the report string (which might be a JSON string containing an error or the report).
#         """
#         if self.verbose: logger.info(f"Orchestrator: Running Market Intelligence for query: '{company_query}'")
        
#         # The MarketIntelligenceAgent's run method expects "company_name_query"
#         # and "supporting_documents" (as a single string).
#         # It handles its own file reading if file_paths were passed to its internal loaders,
#         # but here we prepare 'supporting_documents' string if needed.
#         # However, the MarketIntelligenceAgent's run method now calls _pre_analyze_input which takes file text.
#         # For direct call, we'll need to ensure the agent's internal file loading is used if file_paths are given
#         # or a combined text is passed. Let's simplify: pass paths and let agent handle or pass combined text.

#         # Option 1: Agent handles paths (if its run method adapted) - Preferred
#         # Option 2: Orchestrator combines text.
#         # Your Market Agent's `run` takes `supporting_documents` (a string).
#         # Its `_pre_analyze_input` also takes `supporting_documents_text`.
        
#         supporting_docs_str = self._load_files_for_agent_input(self, file_paths or [])

#         market_input = {
#             "company_name_query": company_query, 
#             "supporting_documents": supporting_docs_str,
#         }
#         report_output = MarketIntelligenceAgent.run(market_input) # This is a string (report or JSON error)
        
#         # Standardize error checking
#         try:
#             # Check if it's a JSON string with an "error" key
#             data = json.loads(report_output)
#             if isinstance(data, dict) and "error" in data:
#                 logger.error(f"Orchestrator: Market Intelligence failed: {data['error']}")
#                 return report_output # Return the JSON error string
#         except json.JSONDecodeError:
#             # Not a JSON error, check for "Error:" prefix
#             if report_output.startswith("Error:"):
#                  logger.error(f"Orchestrator: Market Intelligence failed: {report_output}")
#         return report_output


#     def run_client_persona_creation(self, client_description_text: str, transcript_file_paths: list = None) -> str:
#         if self.verbose: logger.info(f"Orchestrator: Running Client Persona Creation. Description: '{client_description_text[:100]}...'")
#         persona_prompt = ClientRepresentativeCreatorAgent.run(
#             client_description=client_description_text,
#             transcript_file_paths=transcript_file_paths or [] # Agent handles paths
#         )
#         if persona_prompt.startswith("Error:"): logger.error(f"Orchestrator: Client Persona Creation failed: {persona_prompt}")
#         return persona_prompt

#     def run_jd_writing(self, manual_input_for_jd: str, supporting_file_paths: list = None) -> str:
#         if self.verbose: logger.info(f"Orchestrator: Running JD Writing. Manual Input: '{manual_input_for_jd[:100]}...'")
#         jd = JobDescriptionWriterAgent.run(
#             manual_input=manual_input_for_jd,
#             file_paths=supporting_file_paths or [] # Agent handles paths
#         )
#         if jd.startswith("Error:"): logger.error(f"Orchestrator: JD Writing failed: {jd}")
#         return jd
        
#     def run_interview_report_creation(self, structured_input_text: str, attachment_file_paths: list = None) -> str:
#         if self.verbose: logger.info(f"Orchestrator: Running Interview Report Creation. Input Text: '{structured_input_text[:100]}...'")
#         report = InterviewReportCreatorAgent.run(
#             input_text=structured_input_text,
#             attachment_paths=attachment_file_paths or [] # Agent handles paths
#         )
#         if report.startswith("Error:"): logger.error(f"Orchestrator: Interview Report Creation failed: {report}")
#         return report

#     def run_client_feedback_generation(self, statement_for_feedback: str, context_file_paths: list = None) -> str:
#         if self.verbose: logger.info(f"Orchestrator: Running Client Feedback Generation. Statement: '{statement_for_feedback[:100]}...'")
#         feedback = ClientRepresentativeAgent.run(
#             input_statement=statement_for_feedback,
#             transcript_file_paths=context_file_paths or [] # Agent handles paths
#         )
#         if feedback.startswith("Error:"): logger.error(f"Orchestrator: Client Feedback Generation failed: {feedback}")
#         return feedback

#     # --- Chained Pipeline Methods ---

#     def create_jd_from_market_report(self, market_report_text: str, jd_role_request: str, additional_jd_files: list = None) -> dict:
#         results = {}
#         if self.verbose: logger.info(f"Orchestrator: JD from Market Report for role: '{jd_role_request}'")

#         is_report_valid = is_orchestrator_input_valid(market_report_text, min_length=50) and \
#                           not market_report_text.startswith("Error:") and \
#                           not ("error" in market_report_text.lower() and len(market_report_text) < 200 and market_report_text.startswith("{"))
                          
#         if not is_report_valid:
#             results["job_description_error"] = "Error: Invalid or empty market report provided to create JD."
#             logger.error(results["job_description_error"] + f" (Report snippet: {market_report_text[:100]})")
#             return results
#         if not is_orchestrator_input_valid(jd_role_request, min_length=3):
#             results["job_description_error"] = "Error: Role for JD not specified or too short."
#             logger.error(results["job_description_error"])
#             return results

#         manual_input = f"""Instruction: Create a Job Description for the role: "{jd_role_request}".
# Use the following Market Intelligence Report as the primary source of context for the company, its strategy, and market position.
# Ensure the JD reflects the insights from this report.

# --- MARKET INTELLIGENCE REPORT START ---
# {market_report_text}
# --- MARKET INTELLIGENCE REPORT END ---
# """
#         jd = self.run_jd_writing(manual_input_for_jd=manual_input, supporting_file_paths=additional_jd_files)
#         results["job_description"] = jd
#         if jd.startswith("Error:"): results["job_description_error"] = jd
#         return results

#     def create_client_persona_from_transcripts(self, persona_request: str, transcript_files: list) -> dict:
#         results = {}
#         if self.verbose: logger.info(f"Orchestrator: Client Persona from Transcripts. Request: '{persona_request}'")

#         if not transcript_files: # transcript_files must be provided for this method
#             results["client_persona_error"] = "Error: No transcript files provided for persona creation."
#             logger.error(results["client_persona_error"])
#             return results
#         if not is_orchestrator_input_valid(persona_request, min_length=3, allow_empty_ok=False): # Request can be short but not empty
#             results["client_persona_error"] = "Error: Persona creation request is too short or invalid."
#             logger.error(results["client_persona_error"])
#             return results
            
#         persona_prompt_str = self.run_client_persona_creation(
#             client_description_text=persona_request,
#             transcript_file_paths=transcript_files
#         )
#         results["client_persona_prompt"] = persona_prompt_str
#         if persona_prompt_str.startswith("Error:"): results["client_persona_error"] = persona_prompt_str
#         return results

#     def get_feedback_on_document(self, client_persona_guidance: str, document_to_review: str, context_files: list = None) -> dict:
#         results = {}
#         if self.verbose: logger.info(f"Orchestrator: Feedback on Document. Persona: '{client_persona_guidance[:100]}...'. Doc: '{document_to_review[:100]}...'")

#         if not is_orchestrator_input_valid(client_persona_guidance, min_length=10): # Persona guidance should have some substance
#             results["client_feedback_error"] = "Error: Client persona guidance is missing or too short."
#             logger.error(results["client_feedback_error"])
#             return results
#         if not is_orchestrator_input_valid(document_to_review, min_length=20): # Document should be substantial
#             results["client_feedback_error"] = "Error: Document to review is missing or too short."
#             logger.error(results["client_feedback_error"])
#             return results

#         input_statement_for_feedback = f"""
# ---CLIENT PERSONA GUIDANCE---
# {client_persona_guidance}
# ---END CLIENT PERSONA GUIDANCE---

# ---DOCUMENT TO REVIEW---
# {document_to_review}
# ---END DOCUMENT TO REVIEW---
# """
#         feedback_text = self.run_client_feedback_generation(
#             statement_for_feedback=input_statement_for_feedback,
#             context_file_paths=context_files
#         )
#         results["client_feedback"] = feedback_text
#         if feedback_text.startswith("Error:"): results["client_feedback_error"] = feedback_text
#         return results


#     def full_pipeline_from_company_query(self, company_query: str, initial_files: list = None, specific_jd_role_request:str = None) -> dict:
#         pipeline_results = {"steps_completed": []} # Changed key for clarity
#         if self.verbose: logger.info(f"Orchestrator: Starting FULL pipeline for query: '{company_query}'")
        
#         if not is_orchestrator_input_valid(company_query, min_length=1): # Company query can be very short
#             pipeline_results["pipeline_error"] = "Error: Initial company query is invalid or empty."
#             logger.error(pipeline_results["pipeline_error"])
#             return pipeline_results
            
#         initial_files = initial_files or []

#         # 1. Market Intelligence
#         market_report_output_str = self.run_market_intelligence(company_query, initial_files)
#         pipeline_results["market_report_raw_output"] = market_report_output_str # Store raw output
        
#         market_report_text = ""
#         try: # Try to parse if it's a JSON error, otherwise assume it's report text
#             data = json.loads(market_report_output_str)
#             if isinstance(data, dict) and "error" in data:
#                 logger.error(f"Full Pipeline - Market Intel Error: {data['error']}")
#                 pipeline_results["market_report_error"] = data['error']
#                 return pipeline_results 
#             # If it was JSON but not an error structure, treat as text (though unlikely for this agent)
#             market_report_text = market_report_output_str 
#         except json.JSONDecodeError:
#             market_report_text = market_report_output_str # Assume it's the report text or a non-JSON error string

#         if market_report_text.startswith("Error:"):
#             logger.error(f"Full Pipeline - Market Intel Error (string): {market_report_text}")
#             pipeline_results["market_report_error"] = market_report_text
#             return pipeline_results
        
#         pipeline_results["market_report"] = market_report_text
#         pipeline_results["steps_completed"].append("Market Report Generated")
#         if self.verbose: logger.info("Full Pipeline: Market Report Generated.")

#         # 2. Client Persona Creation
#         persona_input_desc = f"Persona based on the following company/market context:\n{market_report_text}"
#         client_persona_prompt = self.run_client_persona_creation(persona_input_desc, initial_files)
#         if client_persona_prompt.startswith("Error:"):
#             logger.error(f"Full Pipeline - Client Persona Error: {client_persona_prompt}")
#             pipeline_results["client_persona_error"] = client_persona_prompt
#             return pipeline_results
#         pipeline_results["client_persona_prompt"] = client_persona_prompt
#         pipeline_results["steps_completed"].append("Client Persona Prompt Created")
#         if self.verbose: logger.info("Full Pipeline: Client Persona Prompt Created.")

#         # 3. Job Description
#         jd_manual_input = ""
#         if specific_jd_role_request and specific_jd_role_request.strip():
#             jd_manual_input = f"Specific Role Request for JD: {specific_jd_role_request}\n\nClient Persona Context:\n{client_persona_prompt}"
#         else:
#              jd_manual_input = f"""Primary Context for JD (Client Persona derived from Market Report):
# {client_persona_prompt}

# Supporting Context (Original Market Report - use this to infer appropriate roles if not obvious from persona):
# --- MARKET REPORT START ---
# {market_report_text}
# --- MARKET REPORT END ---

# Instruction: Generate a suitable job description. If the client persona implies a certain type of role or need, focus on that. If it's more general, use the market report to identify a strategically relevant role.
# """
#         job_description = self.run_jd_writing(jd_manual_input, initial_files)
#         if job_description.startswith("Error:"):
#             logger.error(f"Full Pipeline - JD Error: {job_description}")
#             pipeline_results["job_description_error"] = job_description
#             return pipeline_results
#         pipeline_results["job_description"] = job_description
#         pipeline_results["steps_completed"].append("Job Description Generated")
#         if self.verbose: logger.info("Full Pipeline: Job Description Generated.")

#         # 4. Interview Report Template/Guide
#         interview_report_input = f"""
# ---CONTEXT: JOB DESCRIPTION---
# {job_description}
# ---END CONTEXT: JOB DESCRIPTION---
# ---CONTEXT: CLIENT PERSONA (HIRING MANAGER'S PERSPECTIVE)---
# {client_persona_prompt}
# ---END CONTEXT: CLIENT PERSONA---
# Instruction: Based on the job description and the hiring client's persona, create a template or a set of key questions and assessment criteria for an interview report for this role. This is NOT a report for a specific candidate yet, but a guide for what should be in such a report.
# """
#         interview_report_template = self.run_interview_report_creation(interview_report_input, []) # No candidate files
#         if interview_report_template.startswith("Error:"):
#             logger.warning(f"Full Pipeline - Interview Report Template Warning (non-fatal): {interview_report_template}")
#             pipeline_results["interview_report_template_error"] = interview_report_template # Log error but continue
#         pipeline_results["interview_report_template"] = interview_report_template
#         pipeline_results["steps_completed"].append("Interview Report Template/Guide Generated")
#         if self.verbose: logger.info("Full Pipeline: Interview Report Template Generated.")

#         # 5. Client Feedback on JD
#         feedback_doc_to_review = job_description
#         feedback_statement = f"""
# ---CLIENT PERSONA GUIDANCE---
# {client_persona_prompt}
# ---END CLIENT PERSONA GUIDANCE---
# ---DOCUMENT TO REVIEW (Job Description)---
# {feedback_doc_to_review}
# ---END DOCUMENT TO REVIEW---
# """
#         client_feedback = self.run_client_feedback_generation(feedback_statement, initial_files)
#         if client_feedback.startswith("Error:"):
#             logger.warning(f"Full Pipeline - Client Feedback Warning (non-fatal): {client_feedback}")
#             pipeline_results["client_feedback_on_jd_error"] = client_feedback # Log error but continue
#         pipeline_results["client_feedback_on_jd"] = client_feedback
#         pipeline_results["steps_completed"].append("Client Feedback on JD Generated")
#         if self.verbose: logger.info("Full Pipeline: Client Feedback on JD Generated.")
        
#         logger.info("Full Pipeline: Completed.")
#         return pipeline_results

import os
import json
import logging
# Assuming your agents are in the 'agents' directory and importable
from agents.market_intelligence_agent import MarketIntelligenceAgent
from agents.client_representative_creator_agent import ClientRepresentativeCreatorAgent
from agents.job_description_writer_agent import JobDescriptionWriterAgent
from agents.interview_report_creator_agent import InterviewReportCreatorAgent
from agents.client_representative_agent import ClientRepresentativeAgent

logger = logging.getLogger(__name__)

# Helper function for basic input validation (can be moved to a utils.py)
def is_orchestrator_input_valid(text: str, min_length: int = 1, allow_empty_ok: bool = False) -> bool:
    if allow_empty_ok and (text is None or not text.strip()):
        return True # Empty is fine if allowed
    if not text or not text.strip() or len(text.strip()) < min_length:
        return False
    # Add more sophisticated checks if needed
    return True


class MultiAgentOrchestrator:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.market_agent = MarketIntelligenceAgent(verbose=verbose)
        self.client_prompt_agent = ClientRepresentativeCreatorAgent(verbose=verbose)
        self.jd_agent = JobDescriptionWriterAgent(verbose=verbose)
        self.interview_agent = InterviewReportCreatorAgent(verbose=verbose)
        self.client_feedback_agent = ClientRepresentativeAgent(verbose=verbose)
        self.conversation_history = []
        if self.verbose:
            logger.info("MultiAgentOrchestrator initialized with all sub-agents.")

    # --- Helper for loading files (generic enough for most agents) ---
    def _load_files_for_agent_input(self, agent_instance, file_paths: list) -> str:
        """
        Loads files and concatenates their text content.
        Agents are expected to have their own file reading methods if they need specific parsing.
        This helper is more for preparing a combined text block of 'supporting documents'
        if an agent expects that instead of handling individual file paths.
        However, most of your agents now handle file_paths directly.
        This method might be more useful if you want to create a single string from files
        BEFORE passing to an agent that doesn't take file_paths.
        For now, let's assume agents called by individual runners will use their own loaders via file_paths.
        """
        all_texts = []
        if not file_paths:
            return ""
            
        for path in file_paths:
            if not os.path.exists(path):
                logger.warning(f"Orchestrator (_load_files_for_agent_input): File not found {path}, skipping.")
                continue
            
            content = ""
            ext = os.path.splitext(path)[-1].lower()
            
            # Try to use agent-specific methods if they exist and are relevant for general text combining
            # This is a bit tricky because agent loaders are specific.
            # For simplicity, this helper will do a generic read for now.
            # The individual agent runner methods should pass `file_paths` to the agent.
            try:
                with open(path, "r", encoding='utf-8', errors='ignore') as f: # errors='ignore' for problematic files
                    content = f.read()
                logger.info(f"Orchestrator (_load_files_for_agent_input): Loaded {path} with generic text read.")
            except Exception as e:
                logger.warning(f"Orchestrator (_load_files_for_agent_input): Could not load {path} generically: {e}")
            
            if content and content.strip():
                all_texts.append(f"--- Content from file: {os.path.basename(path)} ---\n{content.strip()}")
        return "\n\n".join(all_texts).strip()

    # --- Individual Agent Runners (Wrappers for direct agent calls) ---

    def run_market_intelligence(self, company_query: str, file_paths: list = None) -> str:
        """
        Runs the MarketIntelligenceAgent.
        Returns the report string (which might be a JSON string containing an error or the report).
        """
        if self.verbose: logger.info(f"Orchestrator: Running Market Intelligence for query: '{company_query}'")
        
        # The MarketIntelligenceAgent's run method expects "company_name_query"
        # and "supporting_documents" (as a single string).
        # It handles its own file reading if file_paths were passed to its internal loaders,
        # but here we prepare 'supporting_documents' string if needed.
        # However, the MarketIntelligenceAgent's run method now calls _pre_analyze_input which takes file text.
        # For direct call, we'll need to ensure the agent's internal file loading is used if file_paths are given
        # or a combined text is passed. Let's simplify: pass paths and let agent handle or pass combined text.

        # Option 1: Agent handles paths (if its run method adapted) - Preferred
        # Option 2: Orchestrator combines text.
        # Your Market Agent's `run` takes `supporting_documents` (a string).
        # Its `_pre_analyze_input` also takes `supporting_documents_text`.
        
        supporting_docs_str = self._load_files_for_agent_input(self.market_agent, file_paths or [])

        market_input = {
            "company_name_query": company_query, 
            "supporting_documents": supporting_docs_str,
        }
        report_output = self.market_agent.run(market_input) # This is a string (report or JSON error)
        
        # Standardize error checking
        try:
            # Check if it's a JSON string with an "error" key
            data = json.loads(report_output)
            if isinstance(data, dict) and "error" in data:
                logger.error(f"Orchestrator: Market Intelligence failed: {data['error']}")
                return report_output # Return the JSON error string
        except json.JSONDecodeError:
            # Not a JSON error, check for "Error:" prefix
            if report_output.startswith("Error:"):
                 logger.error(f"Orchestrator: Market Intelligence failed: {report_output}")
        return report_output


    def run_client_persona_creation(self, client_description_text: str, transcript_file_paths: list = None) -> str:
        if self.verbose: logger.info(f"Orchestrator: Running Client Persona Creation. Description: '{client_description_text[:100]}...'")
        persona_prompt = self.client_prompt_agent.run(
            client_description=client_description_text,
            transcript_file_paths=transcript_file_paths or [] # Agent handles paths
        )
        if persona_prompt.startswith("Error:"): logger.error(f"Orchestrator: Client Persona Creation failed: {persona_prompt}")
        return persona_prompt

    def run_jd_writing(self, manual_input_for_jd: str, supporting_file_paths: list = None) -> str:
        if self.verbose: logger.info(f"Orchestrator: Running JD Writing. Manual Input: '{manual_input_for_jd[:100]}...'")
        jd = self.jd_agent.run(
            manual_input=manual_input_for_jd,
            file_paths=supporting_file_paths or [] # Agent handles paths
        )
        if jd.startswith("Error:"): logger.error(f"Orchestrator: JD Writing failed: {jd}")
        return jd
        
    def run_interview_report_creation(self, structured_input_text: str, attachment_file_paths: list = None) -> str:
        if self.verbose: logger.info(f"Orchestrator: Running Interview Report Creation. Input Text: '{structured_input_text[:100]}...'")
        report = self.interview_agent.run(
            input_text=structured_input_text,
            attachment_paths=attachment_file_paths or [] # Agent handles paths
        )
        if report.startswith("Error:"): logger.error(f"Orchestrator: Interview Report Creation failed: {report}")
        return report

    def run_client_feedback_generation(self, statement_for_feedback: str, context_file_paths: list = None) -> str:
        if self.verbose: logger.info(f"Orchestrator: Running Client Feedback Generation. Statement: '{statement_for_feedback[:100]}...'")
        feedback = self.client_feedback_agent.run(
            input_statement=statement_for_feedback,
            transcript_file_paths=context_file_paths or [] # Agent handles paths
        )
        if feedback.startswith("Error:"): logger.error(f"Orchestrator: Client Feedback Generation failed: {feedback}")
        return feedback

    # --- Chained Pipeline Methods ---

    def create_jd_from_market_report(self, market_report_text: str, jd_role_request: str, additional_jd_files: list = None) -> dict:
        results = {}
        if self.verbose: logger.info(f"Orchestrator: JD from Market Report for role: '{jd_role_request}'")

        is_report_valid = is_orchestrator_input_valid(market_report_text, min_length=50) and \
                          not market_report_text.startswith("Error:") and \
                          not ("error" in market_report_text.lower() and len(market_report_text) < 200 and market_report_text.startswith("{"))
                          
        if not is_report_valid:
            results["job_description_error"] = "Error: Invalid or empty market report provided to create JD."
            logger.error(results["job_description_error"] + f" (Report snippet: {market_report_text[:100]})")
            return results
        if not is_orchestrator_input_valid(jd_role_request, min_length=3):
            results["job_description_error"] = "Error: Role for JD not specified or too short."
            logger.error(results["job_description_error"])
            return results

        manual_input = f"""Instruction: Create a Job Description for the role: "{jd_role_request}".
Use the following Market Intelligence Report as the primary source of context for the company, its strategy, and market position.
Ensure the JD reflects the insights from this report.

--- MARKET INTELLIGENCE REPORT START ---
{market_report_text}
--- MARKET INTELLIGENCE REPORT END ---
"""
        jd = self.run_jd_writing(manual_input_for_jd=manual_input, supporting_file_paths=additional_jd_files)
        results["job_description"] = jd
        if jd.startswith("Error:"): results["job_description_error"] = jd
        return results

    def create_client_persona_from_transcripts(self, persona_request: str, transcript_files: list) -> dict:
        results = {}
        if self.verbose: logger.info(f"Orchestrator: Client Persona from Transcripts. Request: '{persona_request}'")

        if not transcript_files: # transcript_files must be provided for this method
            results["client_persona_error"] = "Error: No transcript files provided for persona creation."
            logger.error(results["client_persona_error"])
            return results
        if not is_orchestrator_input_valid(persona_request, min_length=3, allow_empty_ok=False): # Request can be short but not empty
            results["client_persona_error"] = "Error: Persona creation request is too short or invalid."
            logger.error(results["client_persona_error"])
            return results
            
        persona_prompt_str = self.run_client_persona_creation(
            client_description_text=persona_request,
            transcript_file_paths=transcript_files
        )
        results["client_persona_prompt"] = persona_prompt_str
        if persona_prompt_str.startswith("Error:"): results["client_persona_error"] = persona_prompt_str
        return results

    def get_feedback_on_document(self, client_persona_guidance: str, document_to_review: str, context_files: list = None) -> dict:
        results = {}
        if self.verbose: logger.info(f"Orchestrator: Feedback on Document. Persona: '{client_persona_guidance[:100]}...'. Doc: '{document_to_review[:100]}...'")

        if not is_orchestrator_input_valid(client_persona_guidance, min_length=10): # Persona guidance should have some substance
            results["client_feedback_error"] = "Error: Client persona guidance is missing or too short."
            logger.error(results["client_feedback_error"])
            return results
        if not is_orchestrator_input_valid(document_to_review, min_length=20): # Document should be substantial
            results["client_feedback_error"] = "Error: Document to review is missing or too short."
            logger.error(results["client_feedback_error"])
            return results

        input_statement_for_feedback = f"""
---CLIENT PERSONA GUIDANCE---
{client_persona_guidance}
---END CLIENT PERSONA GUIDANCE---

---DOCUMENT TO REVIEW---
{document_to_review}
---END DOCUMENT TO REVIEW---
"""
        feedback_text = self.run_client_feedback_generation(
            statement_for_feedback=input_statement_for_feedback,
            context_file_paths=context_files
        )
        results["client_feedback"] = feedback_text
        if feedback_text.startswith("Error:"): results["client_feedback_error"] = feedback_text
        return results


    def full_pipeline_from_company_query(self, company_query: str, initial_files: list = None, specific_jd_role_request:str = None) -> dict:
        pipeline_results = {"steps_completed": []} # Changed key for clarity
        if self.verbose: logger.info(f"Orchestrator: Starting FULL pipeline for query: '{company_query}'")
        
        if not is_orchestrator_input_valid(company_query, min_length=1): # Company query can be very short
            pipeline_results["pipeline_error"] = "Error: Initial company query is invalid or empty."
            logger.error(pipeline_results["pipeline_error"])
            return pipeline_results
            
        initial_files = initial_files or []

        # 1. Market Intelligence
        market_report_output_str = self.run_market_intelligence(company_query, initial_files)
        pipeline_results["market_report_raw_output"] = market_report_output_str # Store raw output
        
        market_report_text = ""
        try: # Try to parse if it's a JSON error, otherwise assume it's report text
            data = json.loads(market_report_output_str)
            if isinstance(data, dict) and "error" in data:
                logger.error(f"Full Pipeline - Market Intel Error: {data['error']}")
                pipeline_results["market_report_error"] = data['error']
                return pipeline_results 
            # If it was JSON but not an error structure, treat as text (though unlikely for this agent)
            market_report_text = market_report_output_str 
        except json.JSONDecodeError:
            market_report_text = market_report_output_str # Assume it's the report text or a non-JSON error string

        if market_report_text.startswith("Error:"):
            logger.error(f"Full Pipeline - Market Intel Error (string): {market_report_text}")
            pipeline_results["market_report_error"] = market_report_text
            return pipeline_results
        
        pipeline_results["market_report"] = market_report_text
        pipeline_results["steps_completed"].append("Market Report Generated")
        if self.verbose: logger.info("Full Pipeline: Market Report Generated.")

        # 2. Client Persona Creation
        persona_input_desc = f"Persona based on the following company/market context:\n{market_report_text}"
        client_persona_prompt = self.run_client_persona_creation(persona_input_desc, initial_files)
        if client_persona_prompt.startswith("Error:"):
            logger.error(f"Full Pipeline - Client Persona Error: {client_persona_prompt}")
            pipeline_results["client_persona_error"] = client_persona_prompt
            return pipeline_results
        pipeline_results["client_persona_prompt"] = client_persona_prompt
        pipeline_results["steps_completed"].append("Client Persona Prompt Created")
        if self.verbose: logger.info("Full Pipeline: Client Persona Prompt Created.")

        # 3. Job Description
        jd_manual_input = ""
        if specific_jd_role_request and specific_jd_role_request.strip():
            jd_manual_input = f"Specific Role Request for JD: {specific_jd_role_request}\n\nClient Persona Context:\n{client_persona_prompt}"
        else:
             jd_manual_input = f"""Primary Context for JD (Client Persona derived from Market Report):
{client_persona_prompt}

Supporting Context (Original Market Report - use this to infer appropriate roles if not obvious from persona):
--- MARKET REPORT START ---
{market_report_text}
--- MARKET REPORT END ---

Instruction: Generate a suitable job description. If the client persona implies a certain type of role or need, focus on that. If it's more general, use the market report to identify a strategically relevant role.
"""
        job_description = self.run_jd_writing(jd_manual_input, initial_files)
        if job_description.startswith("Error:"):
            logger.error(f"Full Pipeline - JD Error: {job_description}")
            pipeline_results["job_description_error"] = job_description
            return pipeline_results
        pipeline_results["job_description"] = job_description
        pipeline_results["steps_completed"].append("Job Description Generated")
        if self.verbose: logger.info("Full Pipeline: Job Description Generated.")

        # 4. Interview Report Template/Guide
        interview_report_input = f"""
---CONTEXT: JOB DESCRIPTION---
{job_description}
---END CONTEXT: JOB DESCRIPTION---
---CONTEXT: CLIENT PERSONA (HIRING MANAGER'S PERSPECTIVE)---
{client_persona_prompt}
---END CONTEXT: CLIENT PERSONA---
Instruction: Based on the job description and the hiring client's persona, create a template or a set of key questions and assessment criteria for an interview report for this role. This is NOT a report for a specific candidate yet, but a guide for what should be in such a report.
"""
        interview_report_template = self.run_interview_report_creation(interview_report_input, []) # No candidate files
        if interview_report_template.startswith("Error:"):
            logger.warning(f"Full Pipeline - Interview Report Template Warning (non-fatal): {interview_report_template}")
            pipeline_results["interview_report_template_error"] = interview_report_template # Log error but continue
        pipeline_results["interview_report_template"] = interview_report_template
        pipeline_results["steps_completed"].append("Interview Report Template/Guide Generated")
        if self.verbose: logger.info("Full Pipeline: Interview Report Template Generated.")

        # 5. Client Feedback on JD
        feedback_doc_to_review = job_description
        feedback_statement = f"""
---CLIENT PERSONA GUIDANCE---
{client_persona_prompt}
---END CLIENT PERSONA GUIDANCE---
---DOCUMENT TO REVIEW (Job Description)---
{feedback_doc_to_review}
---END DOCUMENT TO REVIEW---
"""
        client_feedback = self.run_client_feedback_generation(feedback_statement, initial_files)
        if client_feedback.startswith("Error:"):
            logger.warning(f"Full Pipeline - Client Feedback Warning (non-fatal): {client_feedback}")
            pipeline_results["client_feedback_on_jd_error"] = client_feedback # Log error but continue
        pipeline_results["client_feedback_on_jd"] = client_feedback
        pipeline_results["steps_completed"].append("Client Feedback on JD Generated")
        if self.verbose: logger.info("Full Pipeline: Client Feedback on JD Generated.")
        
        logger.info("Full Pipeline: Completed.")
        return pipeline_results

    def start_jd_creation(self, initial_input: str = None) -> dict:
        """Start the JD creation process by gathering initial information"""
        if initial_input:
            self.conversation_history.append({"role": "user", "content": initial_input})
        
        # Initial questions to gather information
        questions = [
            "What is the role you're hiring for?",
            "What industry is the company in?",
            "What are the key responsibilities for this role?",
            "What are the required qualifications?"
        ]
        
        return {
            "type": "questions",
            "questions": questions,
            "state": "gathering_info"
        }

    def process_jd_input(self, user_response: str, current_state: str, files: list = None) -> dict:
        """Process user input based on the current state of JD creation"""
        self.conversation_history.append({"role": "user", "content": user_response})
        
        if current_state == "gathering_info":
            # Check if we have enough information to proceed
            combined_input = " ".join([msg["content"] for msg in self.conversation_history if msg["role"] == "user"])
            
            if len(self.conversation_history) < 8:  # Need more information
                follow_up_questions = [
                    "What is the expected experience level?",
                    "What are the desired soft skills?",
                    "What is the company culture like?",
                    "What are the growth opportunities in this role?"
                ]
                return {
                    "type": "questions",
                    "questions": follow_up_questions,
                    "state": "gathering_info"
                }
            else:
                # Get market intelligence
                market_input = {
                    "company_name": "Extracted from conversation",  # You'd extract this properly
                    "supporting_documents": combined_input
                }
                market_report = self.market_agent.run(market_input)
                
                # Create initial JD
                jd = self.jd_agent.run(combined_input + "\n\nMarket Context:\n" + market_report, files or [])
                
                # Get client representative feedback
                feedback = self.client_feedback_agent.run(jd, files)
                
                return {
                    "type": "review",
                    "job_description": jd,
                    "feedback": feedback,
                    "state": "review_jd"
                }
                
        elif current_state == "review_jd":
            if "looks good" in user_response.lower() or "approved" in user_response.lower():
                return {
                    "type": "complete",
                    "final_job_description": self.conversation_history[-2]["content"],  # Last JD version
                    "state": "complete"
                }
            
            # Get another revision based on feedback
            current_jd = self.conversation_history[-2]["content"]  # Get last JD version
            revised_jd = self.jd_agent.run(f"Previous JD:\n{current_jd}\n\nFeedback:\n{user_response}", files)
            feedback = self.client_feedback_agent.run(revised_jd, files)
            
            return {
                "type": "review",
                "job_description": revised_jd,
                "feedback": feedback,
                "state": "review_jd"
            }
        
        return {"type": "error", "message": "Invalid state"}