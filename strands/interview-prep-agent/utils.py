import os
from pathlib import Path
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from langsmith.integrations.otel import OtelSpanProcessor
from mcp import stdio_client, StdioServerParameters
from pydantic import Field
from pydantic import BaseModel
from typing import Literal
from strands.models.openai import OpenAIModel
from strands.tools.mcp import MCPClient

class Question(BaseModel):
    question: str = Field(description="The interviewFieldtion text")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="The difficulty level")
    tags: list[str] = Field(description="Relevant tags/categories for the question")


class QuestionsOutput(BaseModel):
    topic: str = Field(description="The main topic of the interview questions")
    questions: list[Question] = Field(description="List of interview questions to research")


class AppState:
    stored_questions: QuestionsOutput | None = None
    search_tools: list = []


def setup_langsmith_tracing():
    if not os.getenv("LANGSMITH_API_KEY"):
        return
    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    provider.add_span_processor(
        OtelSpanProcessor(project=os.getenv("LANGSMITH_PROJECT", "interview-prep-agent"))
    )


def create_model(*, temperature: float | None = None) -> OpenAIModel:
    params = {"max_completion_tokens": 32768}
    if temperature is not None:
        params["temperature"] = temperature
    return OpenAIModel(
        client_args={"api_key": os.getenv("OPENAI_API_KEY")},
        model_id="gpt-4.1",
        params=params
    )


def create_mcp_clients() -> dict[str, MCPClient]:
    def make_client(args: list[str], *, env_key: str = None) -> MCPClient:
        env = {**os.environ}
        if env_key:
            env[env_key] = os.getenv(env_key, "")
        return MCPClient(lambda args=args, env=env: stdio_client(
            StdioServerParameters(command="npx", args=["-y"] + args, env=env)
        ))
    
    return {
        "firecrawl": make_client(["firecrawl-mcp"], env_key="FIRECRAWL_API_KEY"),
        "tavily": make_client(["tavily-mcp@0.1.4"], env_key="TAVILY_API_KEY"),
        "brave_search": make_client(["@brave/brave-search-mcp-server"], env_key="BRAVE_API_KEY"),
        "filesystem": make_client(["@modelcontextprotocol/server-filesystem", os.getcwd()]),
    }


def format_study_guide(*, topic: str, questions: list[Question], responses: list[str]) -> str:
    formatted = []
    for i, (q, response) in enumerate(zip(questions, responses)):
        header = f"## Question {i + 1}: {q.question}\n\n**Difficulty:** {q.difficulty} | **Tags:** {', '.join(q.tags)}"
        formatted.append(f"{header}\n\n{response}")
    return f"# {topic} Study Guide\n\n" + "\n\n---\n\n".join(formatted)


def save_study_guide(*, topic: str, content: str) -> Path:
    answers_dir = Path.cwd() / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    file_path = answers_dir / f"{topic.lower().replace(' ', '-')}-study-guide.md"
    file_path.write_text(content)
    return file_path
