import logging

from hr_assistant.app import create_app
from hr_assistant.core.config import get_settings

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())

app = create_app(settings)
