from groq import Groq
import os


class RAGChatbot:
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    def generate_answer(self, question, retrieved_documents):

        context = "\n\n".join(
            [
                document["document"]
                for document in retrieved_documents
            ]
        )

        prompt = f"""
You are a helpful YouTube video assistant.

Answer the user's question using ONLY the context provided below.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the video."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        return response.choices[0].message.content