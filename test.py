from src.transcript import get_transcript
from src.text_processor import clean_transcript, split_transcript
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.rag import RAGChatbot


# -----------------------------------
# 1. YouTube URL
# -----------------------------------

youtube_url = input("Enter YouTube URL: ")


# -----------------------------------
# 2. Get transcript
# -----------------------------------

print("\nGetting transcript...")

transcript = get_transcript(youtube_url)

print("Transcript retrieved successfully.")
print("Transcript length:", len(transcript))


# -----------------------------------
# 3. Clean transcript
# -----------------------------------

print("\nCleaning transcript...")

cleaned_transcript = clean_transcript(transcript)


# -----------------------------------
# 4. Split transcript into chunks
# -----------------------------------

print("Creating chunks...")

chunks = split_transcript(cleaned_transcript)

print("Number of chunks:", len(chunks))


# -----------------------------------
# 5. Create embeddings
# -----------------------------------

print("\nCreating embeddings...")

embedding_model = EmbeddingModel()

embeddings = embedding_model.embed_documents(chunks)

print("Embedding shape:", embeddings.shape)


# -----------------------------------
# 6. Create FAISS vector store
# -----------------------------------

print("\nCreating FAISS index...")

vector_store = VectorStore()

vector_store.add_documents(
    chunks,
    embeddings
)

print("FAISS index created successfully.")


# -----------------------------------
# 7. Ask question
# -----------------------------------

question = input("\nAsk a question about the video: ")


# -----------------------------------
# 8. Embed question
# -----------------------------------

query_embedding = embedding_model.embed_query(question)


# -----------------------------------
# 9. Retrieve relevant chunks
# -----------------------------------

results = vector_store.search(
    query_embedding,
    top_k=3
)


print("\nRetrieved chunks:")

for i, result in enumerate(results, start=1):
    print(f"\n--- Chunk {i} ---")
    print("Score:", result["score"])
    print(result["document"][:500])


# -----------------------------------
# 10. Generate answer using Groq
# -----------------------------------

print("\nGenerating answer...")

chatbot = RAGChatbot()

answer = chatbot.generate_answer(
    question,
    results
)


# -----------------------------------
# 11. Final answer
# -----------------------------------

print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(answer)