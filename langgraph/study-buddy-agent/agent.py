#!/usr/bin/env python3
"""Study Buddy - Personal Knowledge Assistant with Redis Memory"""

import os
import logging
from datetime import datetime
from enum import Enum
from typing import Annotated, List, Optional, Dict, Any

import ulid
from pydantic import BaseModel, Field
from redis import Redis
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from tavily import TavilyClient

from redisvl.index import SearchIndex
from redisvl.schema.schema import IndexSchema
from redisvl.query import VectorRangeQuery
from redisvl.query.filter import Tag
from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryType(str, Enum):
    TOPIC = "topic"
    NOTE = "note"
    PROGRESS = "progress"
    PREFERENCE = "preference"


class Memory(BaseModel):
    content: str
    memory_type: MemoryType
    metadata: str = "{}"


class Memories(BaseModel):
    memories: List[Memory]


class StoredMemory(Memory):
    id: str
    memory_id: ulid.ULID = Field(default_factory=lambda: ulid.ULID())
    created_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None
    thread_id: Optional[str] = None


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SYSTEM_USER_ID = "default_user"

redis_client: Optional[Redis] = None
knowledge_index: Optional[SearchIndex] = None
openai_embed: Optional[OpenAITextVectorizer] = None
redis_saver: Optional[RedisSaver] = None
tavily_client: Optional[TavilyClient] = None


def setup_redis() -> bool:
    global redis_client, knowledge_index, openai_embed, redis_saver
    
    try:
        redis_client = Redis.from_url(REDIS_URL)
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return False
    
    openai_embed = OpenAITextVectorizer(model="text-embedding-ada-002")
    
    knowledge_schema = IndexSchema.from_dict({
        "index": {
            "name": "study_buddy_knowledge",
            "prefix": "knowledge",
            "key_separator": ":",
            "storage_type": "json",
        },
        "fields": [
            {"name": "content", "type": "text"},
            {"name": "memory_type", "type": "tag"},
            {"name": "metadata", "type": "text"},
            {"name": "created_at", "type": "text"},
            {"name": "user_id", "type": "tag"},
            {"name": "memory_id", "type": "tag"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": 1536,  # OpenAI embedding dimension
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
        ],
    })
    
    try:
        knowledge_index = SearchIndex(
            schema=knowledge_schema,
            redis_client=redis_client,
            validate_on_load=True
        )
        knowledge_index.create(overwrite=True)
        logger.info("Knowledge index ready")
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        return False
    
    redis_saver = RedisSaver(redis_client=redis_client)
    redis_saver.setup()
    logger.info("Redis checkpointer ready")
    
    # Initialize Tavily client for web search
    global tavily_client
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        tavily_client = TavilyClient(api_key=tavily_api_key)
        logger.info("Tavily web search ready")
    else:
        logger.warning("TAVILY_API_KEY not set - web search will be unavailable")
    
    return True


def similar_memory_exists(
    content: str,
    memory_type: MemoryType,
    user_id: str = SYSTEM_USER_ID,
    distance_threshold: float = 0.1,
) -> bool:
    if not openai_embed or not knowledge_index:
        return False
        
    content_embedding = openai_embed.embed(content)
    filters = (Tag("user_id") == user_id) & (Tag("memory_type") == memory_type.value)
    
    vector_query = VectorRangeQuery(
        vector=content_embedding,
        num_results=1,
        vector_field_name="embedding",
        filter_expression=filters,
        distance_threshold=distance_threshold,
        return_fields=["id"],
    )
    results = knowledge_index.query(vector_query)
    return len(results) > 0


def store_memory(
    content: str,
    memory_type: MemoryType,
    user_id: str = SYSTEM_USER_ID,
    metadata: Optional[str] = None,
) -> bool:
    if not openai_embed or not knowledge_index:
        logger.error("Redis not initialized")
        return False
        
    if metadata is None:
        metadata = "{}"
    
    logger.info(f"Storing {memory_type.value}: {content[:50]}...")
    
    if similar_memory_exists(content, memory_type, user_id):
        logger.info("Similar memory exists, skipping")
        return False
    
    embedding = openai_embed.embed(content)
    
    memory_data = {
        "user_id": user_id,
        "content": content,
        "memory_type": memory_type.value,
        "metadata": metadata,
        "created_at": datetime.now().isoformat(),
        "embedding": embedding,
        "memory_id": str(ulid.ULID()),
    }
    
    try:
        knowledge_index.load([memory_data])
        logger.info(f"Stored {memory_type.value} memory")
        return True
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        return False


def retrieve_memories(
    query: str,
    memory_types: Optional[List[MemoryType]] = None,
    user_id: str = SYSTEM_USER_ID,
    distance_threshold: float = 0.35,
    limit: int = 5,
) -> List[StoredMemory]:
    if not openai_embed or not knowledge_index:
        return []
    
    # Validate query is not empty
    if not query or not query.strip():
        logger.warning("Empty query provided to retrieve_memories")
        return []
    
    logger.debug(f"Searching memories for: {query}")
    
    vector_query = VectorRangeQuery(
        vector=openai_embed.embed(query),
        return_fields=[
            "content", "memory_type", "metadata",
            "created_at", "memory_id", "user_id",
        ],
        num_results=limit,
        vector_field_name="embedding",
        dialect=2,
        distance_threshold=distance_threshold,
    )
    
    base_filters = [f"@user_id:{{{user_id}}}"]
    if memory_types:
        type_values = [mt.value for mt in memory_types]
        base_filters.append(f"@memory_type:{{{'|'.join(type_values)}}}")
    
    vector_query.set_filter(" ".join(base_filters))
    
    results = knowledge_index.query(vector_query)
    
    memories = []
    for doc in results:
        try:
            memory = StoredMemory(
                id=doc["id"],
                memory_id=doc["memory_id"],
                user_id=doc["user_id"],
                memory_type=MemoryType(doc["memory_type"]),
                content=doc["content"],
                created_at=doc["created_at"],
                metadata=doc["metadata"],
            )
            memories.append(memory)
        except Exception as e:
            logger.error(f"Error parsing memory: {e}")
    
    return memories


def get_all_memories(
    user_id: str = SYSTEM_USER_ID,
    memory_type: Optional[MemoryType] = None,
) -> List[Dict[str, Any]]:
    if not knowledge_index or not redis_client:
        return []
    
    memories = []
    cursor = 0
    while True:
        cursor, keys = redis_client.scan(cursor, match="knowledge:*", count=100)
        for key in keys:
            data = redis_client.json().get(key)
            if data and data.get("user_id") == user_id:
                if memory_type is None or data.get("memory_type") == memory_type.value:
                    memories.append(data)
        if cursor == 0:
            break
    return memories


@tool
def save_topic(
    topic: str,
    summary: str,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """
    Save a topic that the user has learned about.
    
    Use this when the user discusses or learns about a specific subject, concept, or topic.
    
    Args:
        topic: The name/title of the topic
        summary: A brief summary of what was learned about this topic
    """
    user_id = config.get("configurable", {}).get("user_id", SYSTEM_USER_ID)
    content = f"Topic: {topic}\nSummary: {summary}"
    metadata = f'{{"topic": "{topic}"}}'
    
    if store_memory(content, MemoryType.TOPIC, user_id, metadata):
        return f"Saved topic '{topic}' to your knowledge base!"
    return f"Topic '{topic}' already exists or couldn't be saved."


@tool
def save_note(
    note: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
    category: Optional[str] = None
) -> str:
    """
    Save a personal note or insight for the user.
    
    Use this when the user shares an important insight, tip, or something they want to remember.
    
    Args:
        note: The note or insight to save
        category: Optional category for the note (e.g., "tip", "insight", "reminder")
    """
    user_id = config.get("configurable", {}).get("user_id", SYSTEM_USER_ID)
    content = f"Note: {note}" + (f" [Category: {category}]" if category else "")
    metadata = f'{{"category": "{category or "general"}"}}'
    
    if store_memory(content, MemoryType.NOTE, user_id, metadata):
        return "Saved your note!"
    return "This note already exists or couldn't be saved."


@tool
def update_learning_progress(
    topic: str,
    status: str,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """
    Update learning progress for a topic.
    
    Use this to track how well the user understands a topic.
    
    Args:
        topic: The topic being tracked
        status: Progress status - one of: "learning", "reviewing", "mastered", "struggling"
    """
    user_id = config.get("configurable", {}).get("user_id", SYSTEM_USER_ID)
    
    valid_statuses = ["learning", "reviewing", "mastered", "struggling"]
    if status.lower() not in valid_statuses:
        return f"Invalid status. Use one of: {', '.join(valid_statuses)}"
    
    content = f"Progress Update: {topic} - Status: {status.lower()}"
    metadata = f'{{"topic": "{topic}", "status": "{status.lower()}"}}'
    
    if store_memory(content, MemoryType.PROGRESS, user_id, metadata):
        return f"Updated progress: '{topic}' is now marked as '{status.lower()}'!"
    return f"Progress for '{topic}' already recorded or couldn't be saved."


@tool  
def save_learning_preference(
    preference: str,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """
    Save a learning style preference.
    
    Use this when the user indicates how they prefer to learn 
    (e.g., "I learn better with examples", "I prefer detailed explanations").
    
    Args:
        preference: Description of the learning preference
    """
    user_id = config.get("configurable", {}).get("user_id", SYSTEM_USER_ID)
    content = f"Learning Preference: {preference}"
    
    if store_memory(content, MemoryType.PREFERENCE, user_id):
        return "Noted your learning preference! I'll keep this in mind."
    return "This preference was already saved."


@tool
def recall_knowledge(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
    memory_types: Optional[List[str]] = None
) -> str:
    """
    Search and recall previously stored knowledge.
    
    Use this to find relevant information from past conversations - topics learned, 
    notes saved, progress updates, or learning preferences.
    
    Args:
        query: What to search for in the knowledge base. Use a descriptive query like 
               "recent topics" or "user preferences" - do NOT pass an empty string.
        memory_types: Optional list of types to filter ("topic", "note", "progress", "preference")
    """
    user_id = config.get("configurable", {}).get("user_id", SYSTEM_USER_ID)
    
    # Handle empty query by fetching recent memories instead
    if not query or not query.strip():
        all_memories = get_all_memories(user_id)
        if not all_memories:
            return "No memories stored yet for this user."
        
        # Sort by created_at descending and take recent ones
        sorted_memories = sorted(
            all_memories, 
            key=lambda x: x.get("created_at", ""), 
            reverse=True
        )[:5]
        
        response = ["Here's what I remember from recent conversations:\n"]
        for mem in sorted_memories:
            mem_type = mem.get("memory_type", "unknown").upper()
            content = mem.get("content", "")
            response.append(f"[{mem_type}] {content}")
        
        return "\n".join(response)
    
    types = None
    if memory_types:
        types = []
        for t in memory_types:
            try:
                types.append(MemoryType(t.lower()))
            except ValueError:
                pass
    
    memories = retrieve_memories(query, types, user_id, limit=5)
    
    if not memories:
        return "No relevant memories found for that query."
    
    response = ["Here's what I remember:\n"]
    for mem in memories:
        response.append(f"[{mem.memory_type.value.upper()}] {mem.content}")
    
    return "\n".join(response)


@tool
def generate_quiz(
    config: Annotated[RunnableConfig, InjectedToolArg],
    topic: Optional[str] = None,
    num_questions: int = 3
) -> str:
    """
    Generate a quiz based on topics the user has learned.
    
    Use this when the user wants to test their knowledge on previously discussed topics.
    
    Args:
        topic: Optional specific topic to quiz on. If not provided, uses recent topics.
        num_questions: Number of questions to generate (1-5)
    """
    user_id = config.get("configurable", {}).get("user_id", SYSTEM_USER_ID)
    
    query = topic if topic else "topics learned concepts studied"
    memories = retrieve_memories(query, [MemoryType.TOPIC], user_id, limit=3)
    
    if not memories:
        return "I don't have any topics stored yet. Let's discuss some subjects first, and then I can quiz you!"
    
    topics_content = "\n".join([m.content for m in memories])
    
    return f"""Quiz Time! Based on what you've learned:

Topics to be quizzed on:
{topics_content}

I'll generate {min(num_questions, 5)} questions about these topics. Ready? Here we go!

(Note: This is a marker for the AI to generate actual quiz questions based on the topics above)"""


@tool
def web_search(query: str, max_results: int = 5, search_depth: str = "basic", topic: str = "general") -> str:
    """
    Search the web for information about any topic.
    
    Use this tool when:
    - The user asks about current events, news, or recent developments
    - You need up-to-date information that might not be in your training data
    - The user wants to learn about something that requires current facts
    - Verifying or supplementing information with latest sources
    
    Args:
        query: The search query - be specific for better results
        max_results: Number of results to return (1-10, default 5)
        search_depth: "basic" for quick results or "advanced" for more comprehensive search
        topic: Category - "general", "news", or "finance"
    """
    if not tavily_client:
        return "Web search is unavailable. Please set the TAVILY_API_KEY environment variable."
    
    try:
        # Clamp max_results to reasonable bounds
        max_results = max(1, min(10, max_results))
        
        # Validate search_depth
        if search_depth not in ["basic", "advanced"]:
            search_depth = "basic"
        
        # Validate topic
        if topic not in ["general", "news", "finance"]:
            topic = "general"
        
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            topic=topic,
            include_answer=True,
        )
        
        # Format the results
        results = []
        
        # Include Tavily's AI-generated answer if available
        if response.get("answer"):
            results.append(f"**Summary:** {response['answer']}\n")
        
        results.append("**Sources:**")
        for i, result in enumerate(response.get("results", []), 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")[:300]  # Truncate content
            results.append(f"\n{i}. **{title}**\n   {content}...\n   Source: {url}")
        
        if not response.get("results"):
            return f"No results found for '{query}'. Try a different search query."
        
        return "\n".join(results)
        
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Search failed: {str(e)}. Please try again."


TOOLS = [
    save_topic,
    save_note,
    update_learning_progress,
    save_learning_preference,
    recall_knowledge,
    generate_quiz,
    web_search,
]

SYSTEM_PROMPT = """You are Study Buddy, an intelligent and friendly personal knowledge assistant. 
Your role is to help users learn, remember, and review information effectively.

Your Capabilities:
1. Topic Tracking: Remember subjects and concepts users discuss with you
2. Note Taking: Save important insights and notes users want to remember  
3. Progress Tracking: Track mastery levels (learning, reviewing, mastered, struggling)
4. Learning Style: Remember user preferences for how they like to learn
5. Knowledge Recall: Search and retrieve past knowledge when relevant
6. Quizzes: Generate quizzes to test knowledge retention
7. Web Search: Search the internet for current information on any topic

Guidelines:
- When a user discusses a new topic, use save_topic to remember it
- When they share an insight or tip, use save_note
- When they express learning preferences, use save_learning_preference
- At the start of conversations or when relevant, use recall_knowledge to personalize responses
- When explaining concepts, adapt to their stored learning preferences
- Be encouraging and supportive

Web Search Guidelines (IMPORTANT):
- Use web_search PROACTIVELY when the user asks about ANY topic you need more information on
- Use web_search when asked to create study guides, explain concepts, or teach about a subject
- Use web_search for current events, recent news, technology topics, or anything that benefits from up-to-date information
- Do NOT ask the user to explain a topic to you - search the web to learn about it first
- When you don't have enough knowledge about something, ALWAYS search before responding

Important: Always check for relevant past knowledge at the start of new conversations 
or when the user asks about something they might have discussed before."""


def create_study_buddy():
    if not redis_saver:
        raise RuntimeError("Redis not initialized. Call setup_redis() first.")
    
    llm = ChatOpenAI(model="gpt-5.2", temperature=0.7).bind_tools(TOOLS)
    
    return create_react_agent(
        model=llm,
        tools=TOOLS,
        checkpointer=redis_saver,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )


class RuntimeState(MessagesState):
    pass


def respond_to_user(state: RuntimeState, config: RunnableConfig) -> RuntimeState:
    agent = create_study_buddy()
    
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        return state
    
    result = agent.invoke({"messages": state["messages"]}, config=config)
    agent_message = result["messages"][-1]
    state["messages"].append(agent_message)
    
    return state


def execute_tools(state: RuntimeState, config: RunnableConfig) -> RuntimeState:
    messages = state["messages"]
    latest_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls),
        None
    )
    
    if not latest_ai:
        return state
    
    tool_messages = []
    tool_map = {t.name: t for t in TOOLS}
    
    for tc in latest_ai.tool_calls:
        tool = tool_map.get(tc["name"])
        if not tool:
            continue
        
        result = tool.invoke(tc["args"], config=config)
        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tc["id"],
            name=tc["name"]
        ))
    
    messages.extend(tool_messages)
    state["messages"] = messages
    return state


def decide_next_step(state: RuntimeState) -> str:
    latest_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None
    )
    if latest_ai and latest_ai.tool_calls:
        return "execute_tools"
    return END


def create_workflow():
    workflow = StateGraph(RuntimeState)
    
    workflow.add_node("agent", respond_to_user)
    workflow.add_node("execute_tools", execute_tools)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        decide_next_step,
        {"execute_tools": "execute_tools", END: END}
    )
    workflow.add_edge("execute_tools", "agent")
    
    return workflow.compile(checkpointer=redis_saver)


def chat(
    user_message: str,
    thread_id: str = "default",
    user_id: str = SYSTEM_USER_ID,
) -> str:
    graph = create_workflow()
    config = RunnableConfig(configurable={"thread_id": thread_id, "user_id": user_id})
    state = RuntimeState(messages=[HumanMessage(content=user_message)])
    
    for result in graph.stream(state, config=config, stream_mode="values"):
        state = RuntimeState(**result)
    
    ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
    if ai_messages:
        return ai_messages[-1].content
    return "I couldn't generate a response. Please try again."


def main():
    print("\nWelcome to Study Buddy! Your Personal Knowledge Assistant")
    print("=" * 60)
    print("I help you learn, remember, and review information.")
    print("Type 'quit' to exit.\n")
    
    if not setup_redis():
        print("Failed to connect to Redis. Make sure Redis is running.")
        return
    
    user_id = input("Enter your name (or press Enter for 'default'): ").strip() or "default"
    thread_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d')}"
    
    print(f"\nHello, {user_id}! Let's start learning together.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\nGreat study session! See you next time!")
                break
            
            response = chat(user_input, thread_id, user_id)
            print(f"\nStudy Buddy: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nStudy session ended. Keep learning!")
            break


if __name__ == "__main__":
    main()
