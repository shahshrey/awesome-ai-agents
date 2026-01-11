#!/usr/bin/env python3
"""Study Buddy - Streamlit UI"""

import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Study Buddy",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from agent import (
    setup_redis, chat, get_all_memories, retrieve_memories, MemoryType
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    :root {
        --bg-deep: #050505;
        --bg-glass: rgba(20, 20, 25, 0.7);
        --bg-glass-hover: rgba(30, 30, 35, 0.8);
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glass-hover: rgba(255, 255, 255, 0.15);
        
        --primary: #8b5cf6;       /* Violet 500 */
        --primary-glow: rgba(139, 92, 246, 0.5);
        --secondary: #06b6d4;     /* Cyan 500 */
        --secondary-glow: rgba(6, 182, 212, 0.5);
        
        --text-main: #fafafa;
        --text-muted: #a1a1aa;
        
        --font-display: 'Outfit', sans-serif;
        --font-body: 'Inter', sans-serif;
        
        --radius-sm: 8px;
        --radius-md: 16px;
        --radius-lg: 24px;
        --radius-full: 9999px;
    }
    
    .stApp {
        background-color: var(--bg-deep);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(139, 92, 246, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.08) 0%, transparent 40%);
        color: var(--text-main);
        font-family: var(--font-body);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-display) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-main) !important;
    }
    
    p, span, div {
        font-family: var(--font-body);
        color: var(--text-main);
    }
    
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1000px;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(10, 10, 12, 0.95);
        border-right: 1px solid var(--border-glass);
        backdrop-filter: blur(20px);
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-family: var(--font-display);
    }
    
    .stTextInput input, .stChatInput input {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-main) !important;
        padding: 0.75rem 1rem !important;
        font-family: var(--font-body) !important;
        transition: all 0.2s ease;
    }
    
    .stTextInput input:focus, .stChatInput input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 1px var(--primary-glow) !important;
        background: var(--bg-glass-hover) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-main) !important;
        font-family: var(--font-display) !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.25rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stButton > button:hover {
        border-color: var(--primary) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--primary-glow);
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%) !important;
    }

    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: var(--border-glass-hover);
    }
    
    .memory-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-md);
        padding: 1rem;
        margin-bottom: 0.75rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .memory-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: var(--primary);
        opacity: 0.5;
    }
    
    .memory-card:hover {
        transform: translateX(4px);
        border-color: var(--primary-glow);
        background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    }

    .memory-tag {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.2rem 0.6rem;
        border-radius: var(--radius-full);
        background: rgba(255,255,255,0.05);
        color: var(--text-muted);
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .stChatMessage {
        background: transparent !important;
    }
    
    [data-testid="stChatMessageContent"] {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass);
        backdrop-filter: blur(10px);
        border-radius: var(--radius-md) !important;
        padding: 1rem 1.5rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .stChatMessage[data-testid="user-message"] [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%) !important;
        border-color: rgba(139, 92, 246, 0.2);
    }
    
    .stChatMessage[data-testid="assistant-message"] [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.05) 0%, rgba(6, 182, 212, 0.02) 100%) !important;
        border-color: rgba(6, 182, 212, 0.1);
    }

    .hero-container {
        text-align: center;
        padding: 4rem 1rem;
        position: relative;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(to right, #fff, #a1a1aa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: var(--text-muted);
        max-width: 600px;
        margin: 0 auto 3rem auto;
        font-weight: 300;
        line-height: 1.6;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        margin-bottom: 3rem;
    }
    
    .feature-item {
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--border-glass);
        padding: 1.5rem;
        border-radius: var(--radius-md);
        text-align: left;
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        background: rgba(255,255,255,0.04);
        transform: translateY(-5px);
        border-color: var(--primary-glow);
    }
    
    .feature-icon {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stSpinner > div {
        border-top-color: var(--primary) !important;
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = "default"
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if "redis_ready" not in st.session_state:
        st.session_state.redis_ready = False
    if "show_welcome" not in st.session_state:
        st.session_state.show_welcome = True


def render_sidebar():
    with st.sidebar:
        st.markdown("## Study Buddy")
        st.markdown("<p style='font-size: 0.8rem; color: var(--text-muted); margin-top: -15px; margin-bottom: 20px;'>PERSONAL KNOWLEDGE OS</p>", unsafe_allow_html=True)
        
        with st.expander("User Profile", expanded=True):
            new_user = st.text_input(
                "Identity",
                value=st.session_state.user_id,
                placeholder="Enter identifier...",
                key="user_input"
            )
            
            if new_user != st.session_state.user_id:
                st.session_state.user_id = new_user
                st.session_state.thread_id = f"session_{new_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.rerun()
            
            if st.button("New Session", use_container_width=True):
                st.session_state.messages = []
                st.session_state.thread_id = f"session_{st.session_state.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.show_welcome = True
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("#### Database Stats")
        
        if st.session_state.redis_ready:
            try:
                all_memories = get_all_memories(st.session_state.user_id)
                
                topics = len([m for m in all_memories if m.get("memory_type") == "topic"])
                notes = len([m for m in all_memories if m.get("memory_type") == "note"])
                progress = len([m for m in all_memories if m.get("memory_type") == "progress"])
                
                st.markdown(
                    f'<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 1rem;">'
                    f'<div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; text-align: center;">'
                    f'<div style="font-size: 1.2rem; font-weight: 700; color: var(--primary);">{topics}</div>'
                    f'<div style="font-size: 0.6rem; text-transform: uppercase; color: var(--text-muted);">Topics</div>'
                    f'</div>'
                    f'<div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; text-align: center;">'
                    f'<div style="font-size: 1.2rem; font-weight: 700; color: var(--secondary);">{notes}</div>'
                    f'<div style="font-size: 0.6rem; text-transform: uppercase; color: var(--text-muted);">Notes</div>'
                    f'</div>'
                    f'<div style="background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; text-align: center;">'
                    f'<div style="font-size: 1.2rem; font-weight: 700; color: #fff;">{progress}</div>'
                    f'<div style="font-size: 0.6rem; text-transform: uppercase; color: var(--text-muted);">Items</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                    
            except Exception:
                st.info("Initializing database...")
        else:
            st.markdown("<div style='color: #ef4444; font-size: 0.8rem;'>● Redis Disconnected</div>", unsafe_allow_html=True)
        
        st.markdown("#### Neural Recall")
        
        if st.session_state.redis_ready:
            memory_type_filter = st.selectbox(
                "Filter",
                ["All", "Topics", "Notes", "Progress", "Preferences"],
                key="memory_filter",
                label_visibility="collapsed"
            )
            
            search_query = st.text_input(
                "Search",
                placeholder="Query neural bank...",
                key="memory_search",
                label_visibility="collapsed"
            )
            
            if st.button("Search Database", use_container_width=True):
                if search_query:
                    type_map = {
                        "All": None,
                        "Topics": [MemoryType.TOPIC],
                        "Notes": [MemoryType.NOTE],
                        "Progress": [MemoryType.PROGRESS],
                        "Preferences": [MemoryType.PREFERENCE],
                    }
                    memories = retrieve_memories(
                        search_query,
                        type_map[memory_type_filter],
                        st.session_state.user_id,
                        limit=10
                    )
                    
                    if memories:
                        for mem in memories:
                            truncated = mem.content[:200] + ('...' if len(mem.content) > 200 else '')
                            st.markdown(
                                f'<div class="memory-card">'
                                f'<span class="memory-tag">{mem.memory_type.value}</span>'
                                f'<p style="font-size: 0.85rem; line-height: 1.4; opacity: 0.9;">{truncated}</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.caption("No matching memories found.")


def render_welcome():
    # Hero section
    st.markdown(
        '<div class="hero-container">'
        '<div class="hero-title">Unlock Your<br>Learning Potential</div>'
        '<div class="hero-subtitle">'
        'A next-generation knowledge companion that adapts to your thinking style, '
        'remembers your progress, and helps you master complex subjects faster.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Feature grid - render each item separately for reliability
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            '<div class="feature-item">'
            '<div class="feature-icon">⚡</div>'
            '<div style="font-weight: 600; margin-bottom: 0.25rem;">Adaptive Learning</div>'
            '<div style="font-size: 0.8rem; color: var(--text-muted);">Adjusts to your pace and style</div>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            '<div class="feature-item">'
            '<div class="feature-icon">🧠</div>'
            '<div style="font-weight: 600; margin-bottom: 0.25rem;">Active Recall</div>'
            '<div style="font-size: 0.8rem; color: var(--text-muted);">Never forget what you read</div>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            '<div class="feature-item">'
            '<div class="feature-icon">🔗</div>'
            '<div style="font-weight: 600; margin-bottom: 0.25rem;">Knowledge Graph</div>'
            '<div style="font-size: 0.8rem; color: var(--text-muted);">Connects concepts automatically</div>'
            '</div>',
            unsafe_allow_html=True
        )
    
    st.markdown("")  # Spacer
    st.markdown('<div style="text-align: center; margin-bottom: 1rem; color: var(--text-muted); font-size: 0.9rem;">START A CONVERSATION</div>', unsafe_allow_html=True)
    
    cols = st.columns(3)
    suggestions = [
        "Explain Quantum Computing",
        "Create a Study Plan for Python",
        "Test me on History"
    ]
    
    for col, suggestion in zip(cols, suggestions):
        with col:
            if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.session_state.show_welcome = False
                st.rerun()


def render_chat():
    needs_response = (
        st.session_state.messages and 
        st.session_state.messages[-1]["role"] == "user" and
        not any(m["role"] == "assistant" for i, m in enumerate(st.session_state.messages) if i > 0 and st.session_state.messages[i-1] == st.session_state.messages[-1])
    )
    
    if st.session_state.messages:
        last_user_idx = None
        for i in range(len(st.session_state.messages) - 1, -1, -1):
            if st.session_state.messages[i]["role"] == "user":
                last_user_idx = i
                break
        if last_user_idx is not None:
            needs_response = not any(
                m["role"] == "assistant" 
                for m in st.session_state.messages[last_user_idx + 1:]
            )
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if needs_response and st.session_state.messages:
        pending_msg = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                if st.session_state.redis_ready:
                    response = chat(
                        pending_msg,
                        st.session_state.thread_id,
                        st.session_state.user_id
                    )
                else:
                    response = "⚠️ Database connection unavailable."
            
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    if user_input := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.show_welcome = False
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                if st.session_state.redis_ready:
                    response = chat(
                        user_input,
                        st.session_state.thread_id,
                        st.session_state.user_id
                    )
                else:
                    response = "⚠️ Database connection unavailable."
            
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


def main():
    init_session_state()
    
    if not st.session_state.redis_ready:
        with st.spinner("Connecting to neural core..."):
            st.session_state.redis_ready = setup_redis()
    
    render_sidebar()
    
    if st.session_state.show_welcome and not st.session_state.messages:
        render_welcome()
    
    render_chat()


if __name__ == "__main__":
    main()
