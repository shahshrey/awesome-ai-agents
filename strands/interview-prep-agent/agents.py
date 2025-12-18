import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from strands import Agent, tool

from utils import (
    AppState,
    Question,
    QuestionsOutput,
    format_study_guide,
    save_study_guide,
    create_model,
    create_mcp_clients,
    setup_langsmith_tracing,
)

load_dotenv(dotenv_path=".env")


state = AppState()


@tool
def create_questions_tool(topic: str, questions: list[Question]) -> str:
    """
    Structure interview questions after searching.

    This tool is called by the orchestrator after it has gathered raw questions
    from the search tools. It stores the structured questions in shared state
    for the next step (answer generation).

    Args:
        topic: The interview topic (e.g., "React", "System Design")
        questions: List of Question objects with question text, difficulty, and tags

    Returns:
        Confirmation message prompting the agent to call generate_answers_tool
    """
    state.stored_questions = QuestionsOutput(topic=topic, questions=questions)
    return f'Created {len(questions)} structured questions for "{topic}". Call submit_questions_tool to spawn research agents.'


@tool
def generate_answers_tool() -> str:
    """
    Submit the structured questions to spawn research agents.

    This is the final step in the orchestrator workflow. It:
    1. Retrieves stored questions from shared state
    2. Spawns parallel answer agents (one per question) using ThreadPoolExecutor
    3. Collects all responses and formats them into a study guide
    4. Saves the study guide to disk

    Returns:
        Path to the saved study guide file
    """
    if not state.stored_questions:
        return "Error: No questions found. Call create_questions_tool first."

    topic = state.stored_questions.topic
    questions = state.stored_questions.questions

    # Spawn answer agents in parallel - each agent researches one question
    # max_workers=100 allows high parallelism for faster processing
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(run_answer_agent, q, topic=topic) for q in questions]
        responses = [f.result() for f in futures]

    # Compile all Q&A pairs into a formatted study guide
    study_guide = format_study_guide(
        topic=topic, questions=questions, responses=responses
    )
    file_path = save_study_guide(topic=topic, content=study_guide)

    # Clear state for potential reuse
    state.stored_questions = None
    return f"""Study guide saved to {file_path}
         Don't Repeat the answer in your response, it's already in the study guide file, just provide the file path.
    
    """


def run_answer_agent(question: Question, *, topic: str) -> str:
    answer_agent_prompt = f"""
        <role>
        You are a research agent answering an interview question about {topic}.
        </role>

        <tools>
        Use 2-3 for comprehensive answers:

        <tool name="firecrawl_search">
        Best for docs/tutorials
        Example: {{"query": "{topic} how it works", "limit": 5}}
        </tool>

        <tool name="tavily_search">
        Best for recent/authoritative content
        Example: {{"query": "{topic} guide", "max_results": 5, "search_depth": "advanced"}}
        </tool>

        <tool name="brave_web_search">
        Best for diverse web results
        Example: {{"query": "{topic} explanation", "count": 5}}
        </tool>
        </tools>

        <process>
        1. Search using at least 2 different tools
        2. Synthesize into a comprehensive markdown answer
        </process>

        <format>
        - Clear explanation with key concepts (bold)
        - Code examples if applicable
        - Best practices and common pitfalls
        - Real-world examples
        </format>

        <rules>
        Don't include question in the response.
        </rules>
        """

    agent = Agent(
        model=create_model(temperature=0),
        system_prompt=answer_agent_prompt,
        tools=state.search_tools,
    )

    prompt = f"""
        Answer this {topic} interview question ({question.difficulty} difficulty, 
        tags: {", ".join(question.tags)}): \"{question.question}\"\n
        Use tavily_search, brave_web_search, firecrawl_search to find the answer."""

    response = agent(prompt)
    return response.message["content"][0]["text"]


def run_orchestrator(query: str):
    if not os.getenv("OPENAI_API_KEY"):
        exit(1)

    setup_langsmith_tracing()

    mcp_clients = create_mcp_clients()
    for client in mcp_clients.values():
        client.__enter__()

    try:
        state.search_tools = []
        for client in mcp_clients.values():
            state.search_tools.extend(client.list_tools_sync())

        all_tools = state.search_tools + [create_questions_tool, generate_answers_tool]

        ORCHESTRATOR_PROMPT = """
        <role>
        You are an interview preparation orchestrator.
        </role>

        <workflow>
        <step name="search">
        Call ALL 3 search tools IN PARALLEL:
        - firecrawl_search: {"query": "[TOPIC] interview questions", "limit": 10}
        - tavily_search: {"query": "[TOPIC] interview questions developer", "max_results": 10, "search_depth": "advanced"}
        - brave_web_search: {"query": "[TOPIC] interview questions with answers", "count": 10}
        </step>

        <step name="structure">
        Call create_questions_tool with:
        - topic: the main topic
        - questions: array of {question, difficulty ("easy"/"medium"/"hard"), tags (array)}

        Example: {"topic": "React", "questions": [{"question": "What is useState?", "difficulty": "easy", "tags": ["hooks"]}]}
        </step>

        <step name="submit">
        Call generate_answers_tool to spawn research agents
        </step>
        </workflow>

        <rules>
        - Call each search tool EXACTLY ONCE
        - Extract 15-25+ unique questions from results
        - You MUST call create_questions_tool then generate_answers_tool to complete the task
        - in the end, just return the file path of the study guide. No need to repeat the answer in your response as it's already in the study guide file.
        </rules>
        """
        agent = Agent(
            model=create_model(),
            system_prompt=ORCHESTRATOR_PROMPT,
            tools=all_tools,
        )
        agent(query)
    finally:
        for client in mcp_clients.values():
            client.__exit__(None, None, None)


if __name__ == "__main__":
    run_orchestrator("Find me 10 - 15 aws strands agents interview questions?")
