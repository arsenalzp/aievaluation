from phoenix.evals import LLM
import openai
import os

MODEL_NAME=os.getenv("OPENAI_MODEL")
base_llm = LLM(provider="openai", model=MODEL_NAME)