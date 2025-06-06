from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, APIRouter
from utils.project_context_manager import ProjectContextManager
from typing import List, Optional
from pathlib import Path
from orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator

router = APIRouter()

def get_project_manager(project_id: str) -> ProjectContextManager:
    # Dependency for loading project context
    if not (Path("project_data") / project_id).exists():
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectContextManager(project_id)

@router.post("/start_project")
def start_project():
    project_id = ProjectContextManager.create_new_project()
    return {"project_id": project_id}

@router.get("/project/{project_id}/state")
def get_project_state(project_id: str, manager: ProjectContextManager = Depends(get_project_manager)):
    context = manager.load_context()
    uploads = manager.list_uploads()
    agent_outputs = manager.list_agent_outputs()
    return {
        "context": context,
        "uploads": uploads,
        "agent_outputs": agent_outputs
    }

@router.post("/project/{project_id}/upload")
async def upload_files(
    project_id: str,
    files: List[UploadFile] = File(...),
    manager: ProjectContextManager = Depends(get_project_manager)
):
    saved_files = []
    for file in files:
        path = manager.save_upload(file)
        saved_files.append(path)
    # Optionally update context with file info
    context = manager.load_context()
    context.setdefault("uploads", []).extend(saved_files)
    manager.save_context(context)
    return {"uploaded_files": saved_files}

@router.post("/project/{project_id}/chat")
async def project_chat(
    project_id: str,
    message: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    manager: ProjectContextManager = Depends(get_project_manager)
):
    # Ensure at least one input is provided
    if (not message or not message.strip()) and not files:
        return {"error": "No input provided. Please enter a message or upload files."}
    # 1. Save user message to context
    context = manager.load_context()
    if message:
        context.setdefault("chat_history", []).append({"role": "user", "message": message})
    # 2. Save uploaded files (if any)
    if files:
        for file in files:
            path = manager.save_upload(file)
            context.setdefault("uploads", []).append(path)
    # 3. Orchestrator: auto-detect intent and route
    orchestrator = MultiAgentOrchestrator()
    agent_output = orchestrator.route_step(context, message, manager.list_uploads())
    # 4. Save agent output
    manager.save_agent_output("auto", agent_output)
    context.setdefault("chat_history", []).append({"role": "agent", "message": agent_output})
    manager.save_context(context)
    return {"output": agent_output}

@router.get("/project/{project_id}/compare_candidates")
def compare_candidates(project_id: str, manager: ProjectContextManager = Depends(get_project_manager)):
    # Load all candidate reports from agent_outputs
    outputs = manager.list_agent_outputs()
    candidate_reports = [v for k, v in outputs.items() if k.startswith("candidate_report")]
    # Implement your comparison logic here
    comparison = {"summary": "Comparison logic not yet implemented", "reports": candidate_reports}
    return comparison