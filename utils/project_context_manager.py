import os
import json
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
from fastapi import UploadFile

PROJECTS_ROOT = Path("project_data")

class ProjectContextManager:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.project_dir = PROJECTS_ROOT / project_id
        self.context_path = self.project_dir / "context.json"
        self.upload_dir = self.project_dir / "uploads"
        self.agent_outputs_dir = self.project_dir / "agent_outputs"
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(exist_ok=True)
        self.agent_outputs_dir.mkdir(exist_ok=True)
        if not self.context_path.exists():
            self.save_context({})

    @staticmethod
    def create_new_project() -> str:
        project_id = str(uuid.uuid4())
        ProjectContextManager(project_id)  # Initializes dirs/files
        return project_id

    def load_context(self) -> Dict[str, Any]:
        if self.context_path.exists():
            with open(self.context_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_context(self, context: Dict[str, Any]):
        with open(self.context_path, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2)

    def save_upload(self, file: UploadFile) -> str:
        file_path = self.upload_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return str(file_path)

    def list_uploads(self) -> List[str]:
        return [str(p) for p in self.upload_dir.iterdir() if p.is_file()]

    def save_agent_output(self, step: str, output: Any):
        output_path = self.agent_outputs_dir / f"{step}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

    def list_agent_outputs(self) -> Dict[str, Any]:
        outputs = {}
        for p in self.agent_outputs_dir.iterdir():
            if p.is_file() and p.suffix == ".json":
                with open(p, "r", encoding="utf-8") as f:
                    outputs[p.stem] = json.load(f)
        return outputs