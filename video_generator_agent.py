import os
import time
import json
import uuid
import logging
from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

import requests

import pandas as pd
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

from openai import AsyncOpenAI
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

import runwayml
from runwayml import RunwayML, APIConnectionError, RateLimitError, APIStatusError

from prompts_filtered import PromptAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class WorkflowState(TypedDict):
    """State management for the video generation workflow."""
    original_prompt: str
    refined_prompt: Optional[str]
    # script: Optional[str]
    video_url: Optional[str]
    user_feedback: Optional[str]
    needs_edit: bool
    error: Optional[str]

class Config:
    """Configuration management for the application."""
    def __init__(self):
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        # self.RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")
        self.GOOGLE_DRIVE_CREDS = json.loads(os.getenv("GOOGLE_DRIVE_CREDS", "{}"))
        self.OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        self.MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
        self.RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")

class PromptEnhancer:
    """Handles prompt enhancement and safety checks."""
    def __init__(self, config: Config):
        self.config = config
        self.client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.config.GROQ_API_KEY
        )

    async def enhance_prompt(self, prompt: str) -> str:
        """Enhance a prompt while ensuring safety and structure."""
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a prompt enhancer for video generation. 
                        Follow these rules:
                        1. Maintain the original intent
                        2. Add structure: [Intro] [Main Content] [Call-to-Action]
                        3. Remove any harmful content
                        4. Make it more engaging and clear
                        Output only the enhanced prompt."""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            raise

class VideoGenerator:
    """Handles video generation using RunwayML Gen-2."""
    
    def __init__(self, config: Config):
        self.api_key = config.RUNWAY_API_KEY
        self.client = RunwayML(api_key=self.api_key)

    def generate_video(self, prompt_image: str, prompt_text: str) -> Any:
        # Create a new image-to-video task using the "gen4_turbo" model
        task = self.client.image_to_video.create(
            model='gen4_turbo',
            prompt_image=prompt_image,
            prompt_text=prompt_text
        )
        task_id = task.id

        # Poll the task until it's complete
        while task.status not in ['SUCCEEDED', 'FAILED']:
            time.sleep(10)  # Wait for 10 seconds before polling
            task = self.client.tasks.retrieve(task_id)

        if task.status == 'SUCCEEDED':
            return task
        else:
            raise Exception(f"Task failed with status: {task.status}")

        

class GoogleDriveUploader:
    """Handles file uploads to Google Drive."""
    def __init__(self, config: Config):
        self.config = config
        self.service = self._initialize_drive_service()

    def _initialize_drive_service(self):
        """Initialize Google Drive service."""
        try:
            credentials = service_account.Credentials.from_service_account_info(
                self.config.GOOGLE_DRIVE_CREDS
            )
            return build('drive', 'v3', credentials=credentials)
        except Exception as e:
            logger.error(f"Error initializing Google Drive service: {e}")
            raise

    def upload_video(self, file_path: str) -> str:
        """Upload video to Google Drive."""
        try:
            file_metadata = {
                'name': f'video_{uuid.uuid4()}.mp4',
                'mimeType': 'video/mp4'
            }
            media = MediaFileUpload(file_path, mimetype='video/mp4')
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            return f"https://drive.google.com/file/d/{file['id']}/view"
        except Exception as e:
            logger.error(f"Error uploading to Google Drive: {e}")
            raise

class WorkflowManager:
    """Manages the entire video generation workflow."""
    def __init__(self, config: Config):
        self.config = config
        self.enhancer = PromptEnhancer(config)
        self.video_generator = VideoGenerator(config)
        self.drive_uploader = GoogleDriveUploader(config)
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the workflow graph."""
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("enhance", self._enhance_prompt)
        workflow.add_node("generate_video", self._generate_video)
        workflow.add_node("upload", self._upload_to_drive)
        workflow.add_node("document", self._document_results)

        # Define edges
        workflow.add_edge(START, "enhance")
        workflow.add_edge("enhance", "generate_video")
        workflow.add_edge("generate_video", "upload")
        workflow.add_edge("upload", "document")
        workflow.add_edge("document", END)

        return workflow.compile()

    async def _enhance_prompt(self, state: WorkflowState) -> WorkflowState:
        """Enhance the prompt."""
        try:
            state["refined_prompt"] = await self.enhancer.enhance_prompt(state["original_prompt"])
            return state
        except Exception as e:
            state["error"] = str(e)
            return state

    async def _generate_video(self, state: WorkflowState) -> WorkflowState:
        """Generate video from refined prompt."""
        try:
            if state["error"]:
                return state
            result = self.video_generator.generate_video(state["refined_prompt"])
            state["video_url"] = result["video_url"]
            state["video_file"] = result["download_url"]
            return state
        except Exception as e:
            state["error"] = str(e)
            return state

    async def _upload_to_drive(self, state: WorkflowState) -> WorkflowState:
        """Upload video to Google Drive."""
        try:
            if state["error"]:
                return state
            state["drive_url"] = self.drive_uploader.upload_video(state["video_file"])
            return state
        except Exception as e:
            state["error"] = str(e)
            return state

    def _document_results(self, state: WorkflowState) -> WorkflowState:
        """Document the results."""
        try:
            if state["error"]:
                return state
            output_file = self.config.OUTPUT_DIR / "video_outputs.csv"
            with open(output_file, "a") as f:
                f.write(f"{state['original_prompt']},{state['refined_prompt']},{state['drive_url']}\n")
            return state
        except Exception as e:
            state["error"] = str(e)
            return state
        

    
    async def process_prompts(self) -> List[Dict[str, Any]]:
        """Process the top 10 harmful prompts through the workflow."""
        # Step 1: Analyze prompts
        analyzer = PromptAnalyzer()
        
        # Load and analyze dataset
        dataset_path = os.getenv("DATASET_PATH", "/kaggle/input/prompts")  # or your actual path
        analyzer.load_data(dataset_path)
        analyzer.analyze_prompts()
        analyzer.analyze_top_harmful_prompts(limit=10)
        
        # Step 2: Pull top prompts from DB
        top_harmful_prompts = analyzer.get_harmful_prompts_from_db()

        if top_harmful_prompts is None or top_harmful_prompts.empty:
            logger.error("No harmful prompts retrieved from the database.")
            return []

        results = []
        for _, row in top_harmful_prompts.iterrows():
            if 'prompt' not in row:
                logger.error("Column 'prompt' not found in row.")
                continue

            initial_state = WorkflowState(
                original_prompt=row["prompt"],  # lowercase if column in DB is lowercase
                refined_prompt=None,
                video_url=None,
                user_feedback=None,
                needs_edit=False,
                error=None
            )

            final_state = await self.workflow.ainvoke(initial_state)
            results.append(final_state)

        print(top_harmful_prompts.head())       # Debug
        print(top_harmful_prompts.columns)      # Debug

        return results


async def main():
    """Main entry point for the video generation workflow."""
    try:
        # Initialize configuration
        config = Config()

        # Initialize workflow manager
        workflow_manager = WorkflowManager(config)

        # Process prompts (handled inside the agent now)
        results = await workflow_manager.process_prompts()

        # Log results
        logger.info(f"Processed {len(results)} prompts")
        for result in results:
            if result.get("error"):
                logger.error(f"Error processing prompt: {result['error']}")
            else:
                logger.info(f"Successfully processed prompt: {result['original_prompt'][:50]}...")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())