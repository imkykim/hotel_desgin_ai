"""
Routes for chat2plan integration for Hotel Design AI.
"""

import os
import sys
import logging
import json
import uuid
import threading
import time
import traceback
import shutil

from fastapi import APIRouter, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

# Only add the parent directory of chat2plan_interaction
chat2plan_parent = "/Users/ky/01_Projects"
if chat2plan_parent not in sys.path:
    sys.path.append(chat2plan_parent)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router - NOTE: The prefix is now "/api/chat2plan" to match your frontend calls
router = APIRouter(prefix="/api/chat2plan", tags=["Chat2Plan Integration"])
sessions = {}
PROJECT_ROOT = Path(__file__).parents[4]


class SessionRequest(BaseModel):
    session_id: str


class ChatMessage(BaseModel):
    session_id: str
    message: str


try:
    logger.info(
        f"Attempting to import ArchitectureAISystem from chat2plan_interaction.main"
    )
    from chat2plan_interaction.main import ArchitectureAISystem

    logger.info("Successfully imported ArchitectureAISystem")
except ImportError as e:
    logger.error(f"Failed to import ArchitectureAISystem: {e}")

    # Fallback: Dummy class for development/testing
    class ArchitectureAISystem:
        def __init__(self):
            self.spatial_understanding_record = ""
            self.user_requirement_guess = ""
            self.workflow_manager = type(
                "obj",
                (object,),
                {"get_current_stage": lambda: "STAGE_REQUIREMENT_GATHERING"},
            )
            self.session_manager = type(
                "obj",
                (object,),
                {
                    "add_user_input": lambda x: None,
                    "add_system_response": lambda x: None,
                    "update_spatial_understanding": lambda x: None,
                },
            )
            self.key_questions = []

        def process_user_input(self, user_input):
            return f"Processed: {user_input}"


# Store chat2plan sessions
chat2plan_sessions = {}


# Models
class Context(BaseModel):
    context: Dict[str, Any] = {}


class ChatMessage(BaseModel):
    session_id: str
    message: str


@router.post("/start")
async def start_chat2plan_session(context: Context = Body(...)):
    """Start a new chat2plan session."""
    try:
        # Create a session ID for this chat session
        session_id = str(uuid.uuid4())

        # Initialize the system
        system = ArchitectureAISystem()

        # Store the system in the sessions dictionary
        sessions[session_id] = system

        # Print debug info
        print(f"New session created with ID: {session_id}")
        print(f"Current stage: {system.workflow_manager.get_current_stage()}")
        print(f"Stage description: {system.workflow_manager.get_stage_description()}")

        # You can use the context to pre-populate some information
        if context.context:
            # Example: add initial input based on form data
            initial_info = (
                f"Hotel type: {context.context.get('hotel_type', 'not specified')}\n"
            )
            initial_info += f"Number of rooms: {context.context.get('num_rooms', 'not specified')}\n"
            initial_info += f"Building dimensions: {context.context.get('building_width', 'not specified')}m × "
            initial_info += (
                f"{context.context.get('building_length', 'not specified')}m × "
            )
            initial_info += (
                f"{context.context.get('building_height', 'not specified')}m\n"
            )

            # Record this as system understanding
            system.spatial_understanding_record = initial_info

            # Log it in the session manager
            system.session_manager.update_spatial_understanding(
                {"content": initial_info}
            )

        return {"session_id": session_id}
    except Exception as e:
        print(f"Error starting chat2plan session: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to start chat session: {str(e)}"
        )


@router.post("/chat")
async def chat2plan_process(msg: ChatMessage = Body(...)):
    """Process a chat message in an existing session."""
    session_id = msg.session_id
    user_input = msg.message

    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session")

    system = sessions[session_id]

    try:
        # Record the user input
        system.session_manager.add_user_input(user_input)

        # Process user input and get the response
        response = system.process_user_input(user_input)

        # Record the system response
        system.session_manager.add_system_response(response)

        # Check if stage changed
        previous_stage = system.workflow_manager.get_current_stage()

        # Check if we should automatically advance to the next stage
        all_questions_answered = True
        for question in system.key_questions:
            if question["status"] == "未知":
                all_questions_answered = False
                break

        stage_changed = False
        if (
            all_questions_answered
            and previous_stage == system.workflow_manager.STAGE_REQUIREMENT_GATHERING
        ):
            # Advance to next stage
            system.workflow_manager.advance_to_next_stage()
            stage_changed = True

        current_stage = system.workflow_manager.get_current_stage()
        stage_description = system.workflow_manager.get_stage_description()

        print(f"Stage: {current_stage}, Changed: {stage_changed}")
        print(f"Stage description: {stage_description}")

        # Return the current requirements guess
        return {
            "response": response,
            "requirements": system.user_requirement_guess,
            "stage_change": stage_changed,
            "current_stage": current_stage,
            "previous_stage": previous_stage if stage_changed else None,
            "stage_description": stage_description,
        }
    except Exception as e:
        print(f"Error processing chat message: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error processing your message: {str(e)}"
        )


@router.get("/state")
async def chat2plan_state(session_id: str):
    """Get the current state of a chat2plan session."""
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session")

    system = sessions[session_id]

    try:
        current_stage = system.workflow_manager.get_current_stage()
        stage_description = system.workflow_manager.get_stage_description()

        # Count resolved key questions
        resolved_questions = 0
        total_questions = len(system.key_questions)

        for question in system.key_questions:
            if question["status"] == "已知":
                resolved_questions += 1

        # Set key questions status in workflow manager
        system.workflow_manager.set_key_questions_status(
            resolved_questions, total_questions
        )

        return {
            "current_stage": current_stage,
            "stage_description": stage_description,
            "user_requirement_guess": system.user_requirement_guess,
            "spatial_understanding_record": system.spatial_understanding_record,
            "key_questions": system.key_questions,
            "resolved_questions": resolved_questions,
            "total_questions": total_questions,
            "all_key_questions_known": resolved_questions == total_questions
            and total_questions > 0,
        }
    except Exception as e:
        print(f"Error getting chat state: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving state: {str(e)}")


@router.post("/skip_stage")
async def skip_stage(session_request: SessionRequest = Body(...)):
    session_id = session_request.session_id

    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session")

    system = sessions[session_id]
    current_stage = system.workflow_manager.get_current_stage()

    print(f"Skipping stage: {current_stage}")

    # Advance to the next stage
    system.workflow_manager.advance_to_next_stage()
    new_stage = system.workflow_manager.get_current_stage()

    print(f"Advanced to stage: {new_stage}")

    # Handle special actions for certain stages
    if new_stage == system.workflow_manager.STAGE_CONSTRAINT_GENERATION:
        # Launch constraint generation in a background thread
        def generate_constraints():
            print("Skip triggered constraint generation...")
            try:
                system.finalize_constraints()
                print("Constraint generation complete!")
                system.workflow_manager.advance_to_next_stage()

                viz_stage = system.workflow_manager.get_current_stage()
                print(f"Now in stage: {viz_stage}")

                # Generate visualization
                print("Generating visualization...")
                output_dir = os.path.join(
                    system.session_manager.get_session_dir(),
                    "constraints_visualization.png",
                )
                system.constraint_visualization.visualize_constraints(
                    system.constraints_all, output_path=output_dir
                )
                print(f"Visualization complete! Files created at: {output_dir}")

                # Wait briefly to allow frontend to update
                time.sleep(2)

                # Advance to refinement stage
                system.workflow_manager.advance_to_next_stage()
                refinement_stage = system.workflow_manager.get_current_stage()
                print(f"Advanced to stage: {refinement_stage}")
            except Exception as e:
                print(f"Error in constraint generation after skip: {str(e)}")
                traceback.print_exc()

        threading.Thread(target=generate_constraints, daemon=True).start()

    elif new_stage == system.workflow_manager.STAGE_SOLUTION_GENERATION:
        # Launch solution generation in a background thread
        def generate_solution():
            try:
                print("Skip triggered solution generation...")
                system.current_solution = system.call_solver(system.constraints_all)
                system.session_manager.add_intermediate_state(
                    f"solution_generation_{system.workflow_manager.current_iteration}",
                    {"solution": system.current_solution},
                )
                print("Solution generation complete!")
                system.workflow_manager.advance_to_next_stage()
                refinement_stage = system.workflow_manager.get_current_stage()
                print(f"Advanced to solution refinement stage: {refinement_stage}")
            except Exception as e:
                print(f"Error in solution generation after skip: {str(e)}")
                traceback.print_exc()

        threading.Thread(target=generate_solution, daemon=True).start()

    return {
        "previous_stage": current_stage,
        "current_stage": new_stage,
        "stage_description": system.workflow_manager.get_stage_description(),
    }


# In chat2plan_routes.py - Add this function
@router.get("/export_requirements")
async def export_requirements(session_id: str):
    """Export the generated hotel requirements JSON to the Hotel Design AI directory."""
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session")

    system = sessions[session_id]

    try:
        # Find the exports directory
        session_dir = system.session_manager.get_session_dir()
        exports_dir = os.path.join(session_dir, "exports")
        source_file = os.path.join(exports_dir, "hotel_requirements.json")

        # Check if file exists
        if not os.path.exists(source_file):
            return {"success": False, "error": "Requirements file not yet generated"}

        # Define destination in Hotel Design AI project
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat2plan_{timestamp}_requirements.json"
        dest_dir = os.path.join(PROJECT_ROOT, "data", "program")
        os.makedirs(dest_dir, exist_ok=True)
        dest_file = os.path.join(dest_dir, filename)

        # Copy the file
        shutil.copy(source_file, dest_file)

        # Read the file to return its contents
        with open(source_file, "r", encoding="utf-8") as f:
            requirements_data = json.load(f)

        return {
            "success": True,
            "requirements": requirements_data,
            "source_file": source_file,
            "destination_file": dest_file,
            "filename": filename,
            "program_id": filename.replace(".json", ""),
        }
    except Exception as e:
        logger.error(f"Error exporting requirements: {str(e)}")
        traceback.print_exc()
        return {"success": False, "error": f"Error exporting requirements: {str(e)}"}
