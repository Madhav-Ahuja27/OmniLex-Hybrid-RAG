# """
# app.py
# ======
# Streamlit chatbot frontend for the Multimodal RAG System.
# Supports conversational RAG with streaming responses, document/image
# ingestion, and toggleable HyDE + metadata filtering.
# """

# from __future__ import annotations

# import os
# import tempfile
# from pathlib import Path

# import streamlit as st

# from database_manager import DocumentIngestor
# from retrieval_engine import (
#     RetrievalEngine,
#     GeminiProvider,
#     OllamaProvider,
#     TEXT_ONLY_MODEL_NAME,
#     DEFAULT_MODEL_NAME,
# )

# # ======================================================================
# # Page config
# # ======================================================================

# st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# # ======================================================================
# # Session state defaults
# # ======================================================================

# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "sources" not in st.session_state:
#     st.session_state.sources = {}  # msg_index -> list of sources

# # ======================================================================
# # Sidebar — settings & ingestion
# # ======================================================================

# EMBED_MODELS = {
#     "MiniLM (text-only — fast)": TEXT_ONLY_MODEL_NAME,
#     "CLIP (multimodal — text + images)": DEFAULT_MODEL_NAME,
# }

# LLM_PROVIDERS = ["Google Gemini", "Ollama (local)"]


# @st.cache_resource
# def init_engine(embed_model: str):
#     ingestor = DocumentIngestor()
#     engine = RetrievalEngine(ingestor, model_name=embed_model)
#     return ingestor, engine


# with st.sidebar:
#     st.header("⚙️ Settings")

#     # -- Embedding model --
#     embed_label = st.selectbox("Embedding Model", list(EMBED_MODELS.keys()), index=0)
#     embed_model = EMBED_MODELS[embed_label]
#     ingestor, engine = init_engine(embed_model)

#     # -- LLM provider --
#     llm_choice = st.selectbox("LLM Provider", LLM_PROVIDERS, index=0)

#     if llm_choice == "Google Gemini":
#         api_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
#         gemini_model = st.selectbox("Gemini Model", ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro"], index=0)
#     else:
#         ollama_model = st.text_input("Ollama Model", value="qwen2.5:7b")
#         ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")

#     # -- Retrieval options --
#     st.divider()
#     top_k = st.slider("Results to retrieve", 1, 10, 3)
#     use_hyde = st.checkbox("Enable HyDE pre-retrieval")
#     filter_tasks = st.checkbox("Filter by 'Task' category only")

#     # -- Ingestion --
#     st.divider()
#     st.header("📥 Ingest Data")

#     uploaded_files = st.file_uploader("Upload Documents", type=["txt", "json", "md"], accept_multiple_files=True)
#     uploaded_images = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
#     category = st.selectbox("Category", ["Literature", "Task", "Personal Note", "Reference"])

#     if st.button("🔄 Process & Index"):
#         for uf in (uploaded_files or []):
#             raw = uf.read().decode("utf-8", errors="replace")
#             ext = Path(uf.name).suffix.lower()
#             if ext == ".json":
#                 tmp = os.path.join(tempfile.gettempdir(), uf.name)
#                 with open(tmp, "w", encoding="utf-8") as fh:
#                     fh.write(raw)
#                 ingestor.ingest_json_notes(tmp)
#             elif ext == ".md":
#                 ingestor.ingest_markdown(raw, {"Category": category, "Source": uf.name})
#             else:
#                 ingestor.ingest_text(raw, {"Category": category, "Source": uf.name})

#         for img_file in (uploaded_images or []):
#             tmp = os.path.join(tempfile.gettempdir(), img_file.name)
#             with open(tmp, "wb") as fh:
#                 fh.write(img_file.read())
#             ingestor.ingest_image(tmp, {"Category": category, "Source": img_file.name})

#         engine.build_indexes()
#         stats = ingestor.get_stats()
#         st.success(f"✅ Indexed! {stats['parents']} parents · {stats['children']} children · {stats['images']} images")

#     # -- Stats --
#     st.divider()
#     stats = ingestor.get_stats()
#     st.caption(f"Corpus: {stats['parents']} parents · {stats['children']} children · {stats['images']} images")

#     # -- Clear chat --
#     if st.button("🗑️ Clear Chat"):
#         st.session_state.messages = []
#         st.session_state.sources = {}
#         st.rerun()


# # ======================================================================
# # Build LLM provider
# # ======================================================================

# def get_llm():
#     if llm_choice == "Google Gemini":
#         if not api_key:
#             st.error("Please enter your Gemini API key in the sidebar.")
#             st.stop()
#         return GeminiProvider(api_key=api_key, model_name=gemini_model)
#     else:
#         return OllamaProvider(model_name=ollama_model, base_url=ollama_url)


# # ======================================================================
# # Chat UI
# # ======================================================================

# st.title("🤖 RAG Chatbot")
# st.caption("Upload documents in the sidebar, then chat with your knowledge base.")

# # Render existing messages
# for i, msg in enumerate(st.session_state.messages):
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         # Show sources for assistant messages
#         if msg["role"] == "assistant" and i in st.session_state.sources:
#             srcs = st.session_state.sources[i]
#             if srcs:
#                 with st.expander(f"📎 Sources ({len(srcs)})"):
#                     for j, src in enumerate(srcs):
#                         if src["type"] == "text":
#                             st.text(src["content"][:300] + "...")
#                         else:
#                             st.image(src["path"], caption=str(src.get("metadata", {})), width=200)

# # Chat input
# if prompt := st.chat_input("Ask something about your documents..."):
#     # Show user message
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Build LLM
#     llm = get_llm()

#     # Prepare filters
#     filters = {"Category": "Task"} if filter_tasks else {}

#     # Retrieve + stream response
#     with st.chat_message("assistant"):
#         if not engine.corpus_data:
#             response_text = "⚠️ The index is empty. Please upload and index some documents first using the sidebar."
#             st.markdown(response_text)
#             sources = []
#         else:
#             # Convert chat history (skip system-level messages)
#             history = [
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages[:-1]  # exclude current user msg
#             ]

#             stream, sources = engine.chat_stream(
#                 user_query=prompt,
#                 llm=llm,
#                 chat_history=history,
#                 top_k=top_k,
#                 metadata_filters=filters,
#                 use_hyde=use_hyde,
#             )

#             response_text = st.write_stream(stream)

#             # Show sources
#             if sources:
#                 with st.expander(f"📎 Sources ({len(sources)})"):
#                     for src in sources:
#                         if src["type"] == "text":
#                             st.text(src["content"][:300] + "...")
#                         else:
#                             st.image(src["path"], caption=str(src.get("metadata", {})), width=200)

#     # Save to history
#     assistant_idx = len(st.session_state.messages)
#     st.session_state.messages.append({"role": "assistant", "content": response_text})
#     st.session_state.sources[assistant_idx] = sources


"""
app.py
======
Streamlit chatbot frontend for the Multimodal RAG System.
Supports conversational RAG with streaming responses, document/image
ingestion, and toggleable HyDE + metadata filtering.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from database_manager import DocumentIngestor
from retrieval_engine import (
    RetrievalEngine,
    GeminiProvider,
    OllamaProvider,
    GroqProvider,
    TEXT_ONLY_MODEL_NAME,
    DEFAULT_MODEL_NAME,
)

# ======================================================================
# Page config
# ======================================================================

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# ======================================================================
# Session state defaults
# ======================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = {}  # msg_index -> list of sources

# ======================================================================
# Sidebar — settings & ingestion
# ======================================================================

EMBED_MODELS = {
    "MiniLM (text-only — fast)": TEXT_ONLY_MODEL_NAME,
    "CLIP (multimodal — text + images)": DEFAULT_MODEL_NAME,
}

LLM_PROVIDERS = ["Groq (Llama 3.3)", "Google Gemini", "Ollama (local)"]


@st.cache_resource
def init_engine(embed_model: str):
    ingestor = DocumentIngestor()
    engine = RetrievalEngine(ingestor, model_name=embed_model)
    return ingestor, engine


with st.sidebar:
    st.header("⚙️ Settings")

    # -- Embedding model --
    embed_label = st.selectbox("Embedding Model", list(EMBED_MODELS.keys()), index=0)
    embed_model = EMBED_MODELS[embed_label]
    ingestor, engine = init_engine(embed_model)

    # -- LLM provider --
    llm_choice = st.selectbox("LLM Provider", LLM_PROVIDERS, index=0)

    if llm_choice == "Groq (Llama 3.3)":
        groq_api_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
        groq_model = st.selectbox("Groq Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"], index=0)
    elif llm_choice == "Google Gemini":
        api_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
        gemini_model = st.selectbox("Gemini Model", ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro"], index=0)
    else:
        ollama_model = st.text_input("Ollama Model", value="qwen2.5:7b")
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")

    # -- Retrieval options --
    st.divider()
    top_k = st.slider("Results to retrieve", 1, 10, 3)
    use_hyde = st.checkbox("Enable HyDE pre-retrieval")
    filter_tasks = st.checkbox("Filter by 'Task' category only")

    # -- Ingestion --
    st.divider()
    st.header("📥 Ingest Data")

    uploaded_files = st.file_uploader("Upload Documents", type=["txt", "json", "md"], accept_multiple_files=True)
    uploaded_images = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    category = st.selectbox("Category", ["Literature", "Task", "Personal Note", "Reference"])

    if st.button("🔄 Process & Index"):
        for uf in (uploaded_files or []):
            raw = uf.read().decode("utf-8", errors="replace")
            ext = Path(uf.name).suffix.lower()
            if ext == ".json":
                tmp = os.path.join(tempfile.gettempdir(), uf.name)
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(raw)
                ingestor.ingest_json_notes(tmp)
            elif ext == ".md":
                ingestor.ingest_markdown(raw, {"Category": category, "Source": uf.name})
            else:
                ingestor.ingest_text(raw, {"Category": category, "Source": uf.name})

        for img_file in (uploaded_images or []):
            tmp = os.path.join(tempfile.gettempdir(), img_file.name)
            with open(tmp, "wb") as fh:
                fh.write(img_file.read())
            ingestor.ingest_image(tmp, {"Category": category, "Source": img_file.name})

        engine.build_indexes()
        stats = ingestor.get_stats()
        st.success(f"✅ Indexed! {stats['parents']} parents · {stats['children']} children · {stats['images']} images")

    # -- Stats --
    st.divider()
    stats = ingestor.get_stats()
    st.caption(f"Corpus: {stats['parents']} parents · {stats['children']} children · {stats['images']} images")

    # -- Clear chat --
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.sources = {}
        st.rerun()


# ======================================================================
# Build LLM provider
# ======================================================================

def get_llm():
    if llm_choice == "Groq (Llama 3.3)":
        if not groq_api_key:
            st.error("Please enter your Groq API key in the sidebar.")
            st.stop()
        return GroqProvider(api_key=groq_api_key, model_name=groq_model)
    elif llm_choice == "Google Gemini":
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
            st.stop()
        return GeminiProvider(api_key=api_key, model_name=gemini_model)
    else:
        return OllamaProvider(model_name=ollama_model, base_url=ollama_url)


# ======================================================================
# Chat UI
# ======================================================================

st.title("🤖 RAG Chatbot")
st.caption("Upload documents in the sidebar, then chat with your knowledge base.")

# Render existing messages
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show sources for assistant messages
        if msg["role"] == "assistant" and i in st.session_state.sources:
            srcs = st.session_state.sources[i]
            if srcs:
                with st.expander(f"📎 Sources ({len(srcs)})"):
                    for j, src in enumerate(srcs):
                        if src["type"] == "text":
                            st.text(src["content"][:300] + "...")
                        else:
                            st.image(src["path"], caption=str(src.get("metadata", {})), width=200)

# Chat input
if prompt := st.chat_input("Ask something about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build LLM
    llm = get_llm()

    # Prepare filters
    filters = {"Category": "Task"} if filter_tasks else {}

    # Retrieve + stream response
    with st.chat_message("assistant"):
        if not engine.corpus_data:
            response_text = "⚠️ The index is empty. Please upload and index some documents first using the sidebar."
            st.markdown(response_text)
            sources = []
        else:
            # Convert chat history (skip system-level messages)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]  # exclude current user msg
            ]

            stream, sources = engine.chat_stream(
                user_query=prompt,
                llm=llm,
                chat_history=history,
                top_k=top_k,
                metadata_filters=filters,
                use_hyde=use_hyde,
            )

            response_text = st.write_stream(stream)

            # Show sources
            if sources:
                with st.expander(f"📎 Sources ({len(sources)})"):
                    for src in sources:
                        if src["type"] == "text":
                            st.text(src["content"][:300] + "...")
                        else:
                            st.image(src["path"], caption=str(src.get("metadata", {})), width=200)

    # Save to history
    assistant_idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.sources[assistant_idx] = sources