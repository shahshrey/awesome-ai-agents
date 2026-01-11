# 🧠 Study Buddy - Personal Knowledge Assistant

A next-generation AI-powered study companion that adapts to your learning style, remembers your progress, and helps you master complex subjects faster. Built with LangGraph, Redis vector search, and OpenAI's GPT-5.2.

## 🎥 Demo

Watch the demo video to see Study Buddy in action:

**Option 1: View directly in repository**
- Click on `Study Buddy demo.mp4` in the repository to view the video

**Option 2: Download and play locally**
```bash
# After cloning, open the video file
open "Study Buddy demo.mp4"  # macOS
# or use your default video player
```

**Option 3: Embed in GitHub (if hosted)**
If you upload the video to GitHub Releases or use GitHub's video hosting, you can embed it like this:

```markdown
https://raw.githubusercontent.com/shahshrey/Awesome-ai-agents/main/langgraph/study-buddy-agent/Study%20Buddy%20demo.mp4
```

<details>
<summary>📹 Video Preview (HTML fallback - works when viewing locally)</summary>

<video width="100%" controls>
  <source src="Study Buddy demo.mp4" type="video/mp4">
  Your browser does not support the video tag. Please download the video file to view it.
</video>

</details>

## ✨ Features

### 🎯 Core Capabilities

- **🧠 Adaptive Learning**: Adjusts explanations to your pace and learning style
- **💾 Persistent Memory**: Remembers topics, notes, and progress across sessions using Redis vector search
- **🔍 Knowledge Recall**: Instantly retrieves relevant information from past conversations
- **📝 Smart Note-Taking**: Automatically saves important insights and topics you discuss
- **📊 Progress Tracking**: Tracks your mastery levels (learning, reviewing, mastered, struggling)
- **🎓 Quiz Generation**: Creates personalized quizzes based on topics you've learned
- **🌐 Web Search Integration**: Searches the web for up-to-date information using Tavily API
- **🎨 Beautiful UI**: Modern, glass-morphism design with dark theme

### 🛠️ Memory Types

The agent organizes information into four types:

1. **Topics** - Subjects and concepts you've learned about
2. **Notes** - Personal insights, tips, and important information
3. **Progress** - Learning status for different topics
4. **Preferences** - Your learning style preferences

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Redis server (local or cloud)
- OpenAI API key
- Tavily API key (optional, for web search)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd study-buddy-agent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   REDIS_URL=redis://localhost:6379
   TAVILY_API_KEY=your_tavily_api_key_here  # Optional
   ```

5. **Start Redis**

   **Local Redis:**
   ```bash
   # macOS
   brew install redis
   brew services start redis
   
   # Linux
   sudo apt-get install redis-server
   sudo systemctl start redis
   
   # Docker
   docker run -d -p 6379:6379 redis:latest
   ```

   **Or use Redis Cloud:**
   - Sign up at [Redis Cloud](https://redis.com/try-free/)
   - Get your connection URL and add it to `.env`

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`

## 📖 Usage

### Web Interface (Streamlit)

1. **Start a session**: Enter your identity/name in the sidebar
2. **Chat**: Ask questions, discuss topics, or request explanations
3. **View memories**: Use the "Neural Recall" sidebar to search your knowledge base
4. **Track progress**: The agent automatically saves topics and progress

### Example Interactions

```
You: "Explain quantum computing to me"
Study Buddy: [Provides explanation and saves topic]

You: "I prefer learning with examples"
Study Buddy: [Saves preference and adapts future responses]

You: "What topics have we discussed?"
Study Buddy: [Recalls and lists all topics from memory]

You: "Create a quiz on quantum computing"
Study Buddy: [Generates personalized quiz questions]
```

### Command Line Interface

You can also use the agent directly from the command line:

```bash
python agent.py
```

## 🏗️ Architecture

### Components

```
┌─────────────────┐
│  Streamlit UI   │  (app.py)
│   (Frontend)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Agent Engine   │  (agent.py)
│   (LangGraph)   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ Redis  │ │  OpenAI  │
│ Vector │ │   GPT-5.2│
│ Search │ │          │
└────────┘ └──────────┘
```

### Key Technologies

- **LangGraph**: Agent workflow orchestration
- **LangChain**: LLM integration and tool management
- **Redis + RedisVL**: Vector search and memory storage
- **OpenAI GPT-5.2**: Language model
- **Tavily**: Web search API
- **Streamlit**: Web UI framework

### Memory System

The agent uses Redis with vector embeddings to:
- Store memories with semantic search capabilities
- Retrieve relevant information based on query similarity
- Filter by memory type and user ID
- Prevent duplicate memories using similarity detection

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `REDIS_URL` | Redis connection URL | Yes |
| `TAVILY_API_KEY` | Tavily API key for web search | No |

### Redis Index

The agent automatically creates a Redis index with the following schema:
- **Index Name**: `study_buddy_knowledge`
- **Vector Dimension**: 1536 (OpenAI embeddings)
- **Distance Metric**: Cosine similarity
- **Fields**: content, memory_type, metadata, user_id, embedding

## 🛠️ Development

### Project Structure

```
study-buddy-agent/
├── app.py              # Streamlit UI application
├── agent.py            # Core agent logic and tools
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
└── README.md          # This file
```

### Key Functions

**Memory Management:**
- `store_memory()` - Save a memory to Redis
- `retrieve_memories()` - Search memories by query
- `get_all_memories()` - Get all memories for a user

**Agent Tools:**
- `save_topic()` - Save a learned topic
- `save_note()` - Save a personal note
- `update_learning_progress()` - Track learning progress
- `save_learning_preference()` - Store learning preferences
- `recall_knowledge()` - Search past knowledge
- `generate_quiz()` - Create quiz questions
- `web_search()` - Search the web

### Extending the Agent

To add new capabilities:

1. **Create a new tool** in `agent.py`:
   ```python
   @tool
   def my_new_tool(param: str) -> str:
       """Tool description"""
       # Implementation
       return result
   ```

2. **Add to TOOLS list**:
   ```python
   TOOLS = [..., my_new_tool]
   ```

3. **Update SYSTEM_PROMPT** to guide the agent on when to use it

## 📝 License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [OpenAI](https://openai.com/)
- Memory system uses [RedisVL](https://github.com/RedisVentures/redisvl)

---

**Made with ❤️ for learners everywhere**
