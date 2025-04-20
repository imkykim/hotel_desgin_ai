"""
Routes for chat2plan integration for Hotel Design AI.
"""

import os
import sys
import logging
import json
import uuid

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

chat2plan_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "/Users/ky/01_Projects/chat2plan_interaction",
    )
)
if chat2plan_path not in sys.path:
    sys.path.append(chat2plan_path)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/chat2plan", tags=["Chat2Plan Integration"])

try:
    # Try to import with a more flexible approach
    logger.info(
        f"Attempting to import ArchitectureAISystem from chat2plan_interaction, sys.path: {sys.path}"
    )
    from chat2plan_interaction.main import ArchitectureAISystem

    logger.info("Successfully imported ArchitectureAISystem")
except ImportError as e:
    logger.error(f"Failed to import ArchitectureAISystem: {e}")
    # Fallback import - you might need to modify this based on your actual structure
    try:
        from main import ArchitectureAISystem

        logger.info("Imported ArchitectureAISystem from direct main")
    except ImportError:
        logger.error("Could not import ArchitectureAISystem even with fallback")

        # Create a dummy class for development/testing
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
async def start_chat2plan_session(context: Context):
    """Start a new chat2plan session."""
    try:
        # Create a session ID for this chat session
        session_id = str(uuid.uuid4())

        # Initialize the system
        system = ArchitectureAISystem()
        chat2plan_sessions[session_id] = system

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
        logger.error(f"Error starting chat2plan session: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start chat session: {str(e)}"
        )


@router.post("/chat")
async def chat2plan_process(msg: ChatMessage):
    """Process a chat message in an existing session."""
    session_id = msg.session_id
    user_input = msg.message

    if not session_id or session_id not in chat2plan_sessions:
        raise HTTPException(status_code=400, detail="Invalid session")

    system = chat2plan_sessions[session_id]

    try:
        # Record the user input
        system.session_manager.add_user_input(user_input)

        # Process user input and get the response
        response = system.process_user_input(user_input)

        # Record the system response
        system.session_manager.add_system_response(response)

        # Check if stage changed
        current_stage = system.workflow_manager.get_current_stage()

        # Return the current requirements guess
        return {
            "response": response,
            "requirements": system.user_requirement_guess,
            "stage_change": False,
            "current_stage": current_stage,
        }
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing your message: {str(e)}"
        )


@router.get("/state")
async def chat2plan_state(session_id: str):
    """Get the current state of a chat2plan session."""
    if not session_id or session_id not in chat2plan_sessions:
        raise HTTPException(status_code=400, detail="Invalid session")

    system = chat2plan_sessions[session_id]

    try:
        current_stage = system.workflow_manager.get_current_stage()

        return {
            "current_stage": current_stage,
            "user_requirement_guess": system.user_requirement_guess,
            "spatial_understanding_record": system.spatial_understanding_record,
            "key_questions": system.key_questions,
        }
    except Exception as e:
        logger.error(f"Error getting chat state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving state: {str(e)}")
