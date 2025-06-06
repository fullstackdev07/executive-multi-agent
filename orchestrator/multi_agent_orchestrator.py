from agents.market_intelligence_agent import MarketIntelligenceAgent
from agents.job_description_writer_agent import JobDescriptionWriterAgent
from agents.client_representative_creator_agent import ClientRepresentativeCreatorAgent
from agents.client_representative_agent import ClientRepresentativeAgent
from agents.interview_report_creator_agent import InterviewReportCreatorAgent
import os

class MultiAgentOrchestrator:
    def __init__(self):
        self.market_agent = MarketIntelligenceAgent(verbose=True)
        self.jd_agent = JobDescriptionWriterAgent(verbose=True)
        self.client_rep_creator = ClientRepresentativeCreatorAgent(verbose=True)
        self.client_rep_agent = ClientRepresentativeAgent(verbose=True)
        self.interview_agent = InterviewReportCreatorAgent(verbose=True)

    def _load_files_content(self, file_paths):
        # Helper to load and concatenate file contents
        contents = []
        for path in file_paths:
            if not os.path.exists(path):
                continue
            ext = os.path.splitext(path)[-1].lower()
            try:
                if ext == '.pdf':
                    # Use fitz to extract text from PDF
                    import fitz
                    with fitz.open(path) as doc:
                        text = "".join(page.get_text() for page in doc)
                elif ext == '.txt':
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()
                elif ext == '.json':
                    with open(path, 'r', encoding='utf-8') as f:
                        import json
                        data = json.load(f)
                        text = str(data)
                else:
                    text = ''
                if text.strip():
                    contents.append(f"--- Content from file: {os.path.basename(path)} ---\n{text.strip()}")
            except Exception as e:
                continue
        return "\n\n".join(contents)

    def _summarize_files(self, file_paths):
        """Return a summary of the files: name, type, and a short preview of content."""
        summaries = []
        for path in file_paths:
            if not os.path.exists(path):
                continue
            ext = os.path.splitext(path)[-1].lower()
            try:
                if ext == '.pdf':
                    import fitz
                    with fitz.open(path) as doc:
                        text = "".join(page.get_text() for page in doc)
                        preview = text[:500].replace("\n", " ")
                elif ext == '.txt':
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        preview = text[:500].replace("\n", " ")
                elif ext == '.json':
                    with open(path, 'r', encoding='utf-8') as f:
                        import json
                        data = json.load(f)
                        text = str(data)
                        preview = text[:500].replace("\n", " ")
                else:
                    preview = "(Unsupported file type)"
                summaries.append(f"File: {os.path.basename(path)} | Type: {ext} | Preview: {preview}")
            except Exception as e:
                summaries.append(f"File: {os.path.basename(path)} | Type: {ext} | Error reading file: {e}")
        if not summaries:
            return "No readable files were provided."
        return "\n\n".join(summaries)

    def _detect_intent(self, message: str, file_paths=None) -> str:
        """
        Detects the user's intent based on keywords in the message or file names.
        Returns one of: 'market_intelligence', 'jd_draft', 'client_rep_creation', 'client_feedback', 'interview_report'.
        """
        msg = (message or "").lower()
        if any(k in msg for k in ["market report", "market intelligence", "industry analysis"]):
            return "market_intelligence"
        if any(k in msg for k in ["job description", "jd", "write jd", "create jd"]):
            return "jd_draft"
        if any(k in msg for k in ["persona", "client persona", "create persona"]):
            return "client_rep_creation"
        if any(k in msg for k in ["client representation", "client feedback", "client review"]):
            return "client_feedback"
        if any(k in msg for k in ["interview report", "candidate report", "interview summary"]):
            return "interview_report"
        # Try to infer from file names if message is empty
        if file_paths:
            file_names = " ".join([os.path.basename(f).lower() for f in file_paths])
            if any(k in file_names for k in ["market", "industry"]):
                return "market_intelligence"
            if any(k in file_names for k in ["jd", "job", "description"]):
                return "jd_draft"
            if any(k in file_names for k in ["persona", "client"]):
                return "client_rep_creation"
            if any(k in file_names for k in ["feedback", "review"]):
                return "client_feedback"
            if any(k in file_names for k in ["interview", "candidate", "report"]):
                return "interview_report"
        return "unknown"

    def route_step(self, context, message, file_paths):
        # file_paths: list of file paths (strings)
        step = self._detect_intent(message, file_paths)
        # If both message and files are empty
        if (not message or not message.strip()) and not file_paths:
            return {"error": "No input provided. Please enter a message or upload files."}
        # If only files are provided and intent is unknown, return file summary
        if step == "unknown":
            if file_paths:
                return {"output": "Intent could not be determined from your input. Here is a summary of your uploaded files:\n\n" + self._summarize_files(file_paths)}
            else:
                return {"output": "Sorry, I could not determine your request. Please specify if you want a job description, market report, persona, client feedback, or interview report."}
        if step == "market_intelligence":
            supporting_docs = self._load_files_content(file_paths)
            used_files = [os.path.basename(f) for f in file_paths] if file_paths else []
            market_input = {
                "company_name": message if message else "Unknown",
                "company_location": "Unknown",
                "geography": "Unknown",
                "supporting_documents": supporting_docs,
                "used_files": used_files
            }
            return self.market_agent.run(market_input)
        elif step == "jd_draft":
            file_text = self._load_files_content(file_paths)
            return self.jd_agent.run(manual_input=message or "", file_paths=file_paths)
        elif step == "client_rep_creation":
            return self.client_rep_creator.run(client_description=message or "", transcript_files=file_paths)
        elif step == "client_feedback":
            return self.client_rep_agent.run(user_input=message or "", files=file_paths)
        elif step == "interview_report":
            return self.interview_agent.run(input_text=message or "", attachment_paths=file_paths)
        else:
            return {"output": "Sorry, I could not determine your request. Please specify if you want a job description, market report, persona, client feedback, or interview report."}