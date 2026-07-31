import contextvars
from typing import Optional

# Thread-safe context variables cho từng request
request_api_key = contextvars.ContextVar("request_api_key", default=None)
request_base_url = contextvars.ContextVar("request_base_url", default=None)
request_model_name = contextvars.ContextVar("request_model_name", default=None)
