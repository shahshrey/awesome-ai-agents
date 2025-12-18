# Interview Prep Agent (Python)

Python port of the interview prep agent using [Strands Agents SDK](https://github.com/strands-agents/sdk-python).

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` file in the parent directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
BRAVE_API_KEY=your_brave_api_key
TAVILY_API_KEY=your_tavily_api_key

# Optional: LangSmith tracing
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=interview-prep-agent
```

4. Run the agent:

```bash
python agents.py
```

## Architecture

- **Orchestrator Agent**: Finds interview questions using multiple search tools (firecrawl, tavily, brave)
- **Research Agents**: Spawned in parallel to answer each question
- **MCP Clients**: Connect to external tools via Model Context Protocol

## Files

- `agents.py` - Main agent logic with orchestrator and research agents
- `logger.py` - Colored console logging
- `requirements.txt` - Python dependencies
