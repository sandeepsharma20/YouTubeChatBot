from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_transcript(text):
    """
    Clean unnecessary spaces from transcript.
    """
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text


def split_transcript(text):
    """
    Split transcript into smaller chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    return chunks