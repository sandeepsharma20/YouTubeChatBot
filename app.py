import streamlit as st

from src.transcript import get_transcript
from src.text_processor import clean_transcript, split_transcript
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.rag import RAGChatbot


# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="YouTube Chatbot",
    page_icon="🎥",
    layout="wide"
)


# -----------------------------------
# Title
# -----------------------------------

st.title("🎥 YouTube Video Chatbot")

st.write(
    "Enter a YouTube video URL and ask questions about its content."
)


# -----------------------------------
# Initialize session state
# -----------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------------
# YouTube URL
# -----------------------------------

youtube_url = st.text_input(
    "Enter YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)


# -----------------------------------
# Process Video
# -----------------------------------

if st.button("🚀 Process Video"):

    if not youtube_url:

        st.warning("Please enter a YouTube URL.")

    else:

        try:

            # -------------------------
            # Get transcript
            # -------------------------

            with st.spinner("Getting YouTube transcript..."):

                transcript = get_transcript(youtube_url)

            st.success(
                "Transcript retrieved successfully!"
            )


            # -------------------------
            # Clean transcript
            # -------------------------

            with st.spinner("Cleaning transcript..."):

                cleaned_transcript = clean_transcript(
                    transcript
                )


            # -------------------------
            # Create chunks
            # -------------------------

            with st.spinner("Creating transcript chunks..."):

                chunks = split_transcript(
                    cleaned_transcript
                )

            st.info(
                f"Created {len(chunks)} transcript chunks."
            )


            # -------------------------
            # Create embeddings
            # -------------------------

            with st.spinner("Creating embeddings..."):

                embedding_model = EmbeddingModel()

                embeddings = (
                    embedding_model
                    .embed_documents(chunks)
                )


            # -------------------------
            # Create FAISS index
            # -------------------------

            with st.spinner("Building vector database..."):

                vector_store = VectorStore()

                vector_store.add_documents(
                    chunks,
                    embeddings
                )


            # -------------------------
            # Create chatbot
            # -------------------------

            chatbot = RAGChatbot()


            # -------------------------
            # Save in session state
            # -------------------------

            st.session_state.embedding_model = (
                embedding_model
            )

            st.session_state.vector_store = (
                vector_store
            )

            st.session_state.chatbot = chatbot

            st.session_state.video_processed = True


            # -------------------------
            # Clear previous chat
            # -------------------------

            st.session_state.chat_history = []


            st.success(
                "🎉 Video processed successfully!"
            )


        except Exception as e:

            st.error(
                f"Something went wrong: {str(e)}"
            )


# -----------------------------------
# Chat section
# -----------------------------------

if st.session_state.video_processed:

    st.divider()

    st.subheader(
        "💬 Ask Questions About the Video"
    )


    # -----------------------------------
    # Display previous messages
    # -----------------------------------

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # -----------------------------------
    # Chat input
    # -----------------------------------

    question = st.chat_input(
        "Ask something about the video..."
    )


    if question:

        # -----------------------------
        # Display user question
        # -----------------------------

        with st.chat_message("user"):

            st.write(question)


        # -----------------------------
        # Create question embedding
        # -----------------------------

        query_embedding = (
            st.session_state
            .embedding_model
            .embed_query(question)
        )


        # -----------------------------
        # Retrieve relevant chunks
        # -----------------------------

        results = (
            st.session_state
            .vector_store
            .search(
                query_embedding,
                top_k=3
            )
        )


        # -----------------------------
        # Generate answer
        # -----------------------------

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = (
                    st.session_state
                    .chatbot
                    .generate_answer(
                        question,
                        results,
                        st.session_state.chat_history
                    )
                )

            st.write(answer)


        # -----------------------------
        # Save user question
        # -----------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )


        # -----------------------------
        # Save assistant answer
        # -----------------------------

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )