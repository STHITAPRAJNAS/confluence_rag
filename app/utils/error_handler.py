import sys
import traceback
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ErrorHandler:
    def handle_error(self, error: Exception):
        """Handles errors by logging the error message and traceback."""
        logger.error(f"An error occurred: {error}")
        traceback.print_exc()