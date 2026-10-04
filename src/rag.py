from groq import Groq
import os


class RAGChatbot:

    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )


    def generate_answer(
        self,
        question,
        retrieved_documents,
        chat_history=None
    ):

        # -----------------------------------
        # Create context from retrieved chunks
        # -----------------------------------

        context = "\n\n".join(
            [
                document["document"]
                for document in retrieved_documents
            ]
        )


        # -----------------------------------
        # Create conversation history
        # -----------------------------------

        history_text = ""

        if chat_history:

            for message in chat_history:

                history_text += (
                    f"{message['role'].upper()}: "
                    f"{message['content']}\n"
                )


        # -----------------------------------
        # Create prompt
        # -----------------------------------

        prompt = f"""
You are a helpful YouTube video assistant.

Answer the user's question using ONLY the video context provided below.

You may use the conversation history to understand references
such as "it", "they", "this", or "that".

However, the actual answer must come from the video context.

If the answer cannot be found in the video context, say:

"I couldn't find the answer in the video."

Do not make up information.

--------------------
CONVERSATION HISTORY
--------------------

{history_text}

--------------------
VIDEO CONTEXT
--------------------

{context}

--------------------
CURRENT QUESTION
--------------------

{question}

--------------------
ANSWER
--------------------
"""


        # -----------------------------------
        # Call Groq
        # -----------------------------------

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