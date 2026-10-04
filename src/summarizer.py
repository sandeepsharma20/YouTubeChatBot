import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=GROQ_API_KEY
)


MODEL_NAME = "openai/gpt-oss-20b"


def summarize_chunk(chunk):
    """
    Summarize one transcript chunk.
    """

    prompt = f"""
You are an expert YouTube video summarizer.

Summarize the following transcript chunk.

Focus on:
- Important ideas
- Important concepts
- Facts
- Examples
- Key takeaways

Do not add information that is not present in the transcript.

Transcript chunk:

{chunk}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=700
    )

    return response.choices[0].message.content


def generate_final_summary(chunk_summaries):
    """
    Combine all chunk summaries into one final summary.
    """

    combined_summaries = "\n\n".join(chunk_summaries)

    prompt = f"""
You are an expert YouTube video summarizer.

Below are summaries of different sections of a video.

Create one coherent final summary.

Include:

1. Overall summary
2. Main points
3. Important concepts
4. Key takeaways

Avoid unnecessary repetition.

Section summaries:

{combined_summaries}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1200
    )

    return response.choices[0].message.content


def summarize_transcript(chunks):
    """
    Summarize all transcript chunks.
    """

    chunk_summaries = []

    for chunk in chunks:

        summary = summarize_chunk(chunk)

        chunk_summaries.append(summary)

    final_summary = generate_final_summary(
        chunk_summaries
    )

    return final_summary