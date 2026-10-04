from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(youtube_url):
    """
    Extract the video ID from a YouTube URL.
    """

    if "v=" in youtube_url:
        return youtube_url.split("v=")[1].split("&")[0]

    if "youtu.be/" in youtube_url:
        return youtube_url.split("youtu.be/")[1].split("?")[0]

    raise ValueError("Invalid YouTube URL")


def get_transcript(youtube_url):
    """
    Fetch transcript for a YouTube video.
    """

    video_id = extract_video_id(youtube_url)

    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id,
    languages=["en","hi"])

    text = " ".join(snippet.text for snippet in transcript)

    return text