import logging
import os
import certifi

log_level_root = os.getenv("LOG_LEVEL_ROOT", "INFO").upper()

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# fix ssl certificates for compiled binaries
# https://github.com/pyinstaller/pyinstaller/issues/7229
# https://stackoverflow.com/questions/55736855/how-to-change-the-cafile-argument-in-the-ssl-module-in-python3
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

# AI parameters
api_key = os.environ["AI_API_KEY"]
model = os.getenv("AI_MODEL", "o1-preview")
ai_api_baseurl = os.getenv("AI_API_BASEURL", None)
timeout = int(os.getenv("AI_TIMEOUT", "120"))
temperature = float(os.getenv("TEMPERATURE", "1"))
system_prompt_unformatted = os.getenv(
    "AI_SYSTEM_PROMPT",
    """
You are a helpful assistant used in a Mattermost chat. The current UTC time is {current_time}. 
Whenever users asks you for help you will provide them with succinct answers formatted using Markdown. Do not be apologetic. 
For tasks requiring reasoning or math, use the Chain-of-Thought methodology to explain your step-by-step calculations or logic before presenting your answer. 
Extra data is sent to you in a structured way, which might include file data, website data, and more, which is sent alongside the user message. 
If a user sends a link, use the extracted URL content provided, do not assume or make up stories based on the URL alone. 
If a user sends a YouTube link, primarily focus on the transcript and do not unnecessarily repeat the title, description or uploader of the video. 
In your answer DO NOT contain the link to the video/website the user just provided to you as the user already knows it, unless the task requires it. 
If your response contains any URLs, make sure to properly escape them using Markdown syntax for display purposes."""
)

# Mattermost server details
mattermost_url = os.environ["MATTERMOST_URL"]
mattermost_scheme = os.getenv("MATTERMOST_SCHEME", "https")
mattermost_port = int(os.getenv("MATTERMOST_PORT", "443"))
mattermost_basepath = os.getenv("MATTERMOST_BASEPATH", "/api/v4")

MATTERMOST_CERT_VERIFY = os.getenv("MATTERMOST_CERT_VERIFY", "TRUE")

# Handle the situation where a string path to a cert file might be handed over
if MATTERMOST_CERT_VERIFY == "TRUE":
    MATTERMOST_CERT_VERIFY = True
if MATTERMOST_CERT_VERIFY == "FALSE":
    MATTERMOST_CERT_VERIFY = False

mattermost_token = os.getenv("MATTERMOST_TOKEN", "")
mattermost_ignore_sender_id = os.getenv("MATTERMOST_IGNORE_SENDER_ID", "").split(",")
mattermost_username = os.getenv("MATTERMOST_USERNAME", "")
mattermost_password = os.getenv("MATTERMOST_PASSWORD", "")
mattermost_mfa_token = os.getenv("MATTERMOST_MFA_TOKEN", "")

typing_indicator_mode_is_full = os.getenv("TYPING_INDICATOR_MODE", "FULL") == "FULL"

flaresolverr_endpoint = os.getenv("FLARESOLVERR_ENDPOINT", "")

browser_executable_path = os.getenv("BROWSER_EXECUTABLE_PATH", "/usr/bin/chromium")

# Maximum website/file size
max_response_size = 1024 * 1024 * int(os.getenv("MAX_RESPONSE_SIZE_MB", "100"))

keep_all_url_content = os.getenv("KEEP_ALL_URL_CONTENT", "TRUE").upper() == "TRUE"
