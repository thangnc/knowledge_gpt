from typing import Optional, Dict, Any

from langchain.utils import get_from_dict_or_env
from openai.error import APIConnectionError, APIError, RateLimitError, Timeout
from pydantic import root_validator, BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class OpenAIChats(BaseModel):
    client: Any  #: :meta private:
    complete_model_name: str = "gpt-3.5-turbo"
    openai_api_key: Optional[str] = None

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        try:
            import openai

            openai.api_key = openai_api_key
            values["client"] = openai.ChatCompletion
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        return values

    @retry(
        reraise=True,
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=10, max=60),
        retry=(
                retry_if_exception_type(Timeout)
                | retry_if_exception_type(APIError)
                | retry_if_exception_type(APIConnectionError)
                | retry_if_exception_type(RateLimitError)
        ),
    )
    def _chat_func(self, docs: str, question: str) -> object:
        """Gets an answer to a question from a list of Documents."""

        system_message = """Create a final answer to the given questions using the provided document excerpts(in no particular order) as references. 
        ALWAYS include a "SOURCES" section in your answer including only the minimal set of sources needed to answer the question. 
        If you are unable to answer the question, simply state that you do not know. Do not attempt to fabricate an answer and leave the SOURCES section empty."""

        user_message = """
        ## document
        """ + docs + """
        ## question
        """ + question

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        # Get the answer
        return self.client.create(
            model=self.complete_model_name,
            messages=messages
        ).choices[0].message.content

    def send_chat_message(self, docs: str, question: str):
        embedding = self._chat_func(docs, question)
        return embedding
