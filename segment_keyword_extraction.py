import assemblyai as aai
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
import torch 
import re
from transformers import pipeline

# Setup your AssemblyAI API key
aai.settings.api_key = "1e6e8914d2db4accb91327492212920b"

# URL of the file to transcribe
FILE_URL = './videos/lec1.mp4'

#config = aai.TranscriptionConfig(auto_highlights=True)

config = aai.TranscriptionConfig(
auto_highlights=True,
  speech_model=aai.SpeechModel.best,
  auto_chapters=True,
  language_detection=True
)

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(FILE_URL, config=config)

def format_time(ms):
    """ Helper function to convert milliseconds to a readable format """
    seconds = ms // 1000
    minutes = seconds // 60
    hours = minutes // 60
    return f"{hours:02}:{minutes % 60:02}:{seconds % 60:02}"

if transcript.status == aai.TranscriptStatus.completed:
    print("Transcription Text:\n", transcript.text)
    
    def extract_keywords(text, top_n=3):
        stop_words = list(stopwords.words('english'))  # Convert set to list
        vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=100)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        sorted_items = sorted(zip(vectorizer.idf_, feature_names), reverse=True)
        keywords = [word for score, word in sorted_items[:top_n]]
        return keywords

    # Extract keywords
    keywords = extract_keywords(transcript.text)

    # Initialize the summarization pipeline
    summarizer = pipeline("summarization" , framework='pt')

    # Generate summary for a given text
    def generate_summary(text):
        summary_list = summarizer(text, max_length=20, min_length=5, do_sample=False)
        return summary_list[0]['summary_text']

    # Detect chapters and generate names with timestamps and durations
    def detect_chapters_and_names(transcript, window_size=10, keywords=None):
        chapters = []
        sentences = sent_tokenize(transcript.text)
        word_index = 0  # Track the index in the flattened word list
        chapter_count = 1
        current_chapter_start = None
        chapter_text = []

        for i, sentence in enumerate(sentences):
            contains_keyword = any(keyword.lower() in sentence.lower() for keyword in keywords)
            if contains_keyword and current_chapter_start is None:
                # Begin a new chapter
                while word_index < len(transcript.words) and transcript.words[word_index].text not in sentence:
                    word_index += 1
                current_chapter_start = transcript.words[word_index].start
                chapter_text = [sentence]
            elif contains_keyword:
                # Continue current chapter
                chapter_text.append(sentence)
            elif not contains_keyword and current_chapter_start is not None:
                # End current chapter
                end_time = transcript.words[word_index - 1].end  # get end of the last keyword-inclusive sentence
                chapter_name = generate_summary(' '.join(chapter_text))
                chapters.append((chapter_count, chapter_name, format_time(current_chapter_start), format_time(end_time)))
                chapter_count += 1
                current_chapter_start = None
                chapter_text = []

            # Update the word index to track sentence location
            word_index += len(sentence.split())

        # Check if the last chapter ends at the transcript's end
        if current_chapter_start is not None:
            end_time = transcript.words[-1].end
            chapter_name = generate_summary(' '.join(chapter_text))
            chapters.append((chapter_count, chapter_name, format_time(current_chapter_start), format_time(end_time)))

        # Adjust first and last chapter names
        if chapters:
            chapters[0] = (chapters[0][0], "Introduction", chapters[0][2], chapters[0][3])
            chapters[-1] = (chapters[-1][0], "Ending", chapters[-1][2], chapters[-1][3])

        return chapters

    # Assuming transcript and keywords are already obtained as shown in previous code
    chapters = detect_chapters_and_names(transcript, keywords=keywords)

    if chapters:
        print("Detected Chapters with Timestamps and Durations:")
        for chap in chapters:
            print(f"Chapter {chap[0]}: {chap[1]} (Starts at {chap[2]}, Ends at {chap[3]})")
    else:
        print("No chapters detected.")
else:
    print("Transcription failed or is not completed.")

# Function to convert HH:MM:SS to total seconds
def format_time_to_seconds(time_str):
    """ Convert HH:MM:SS to total seconds """
    parts = list(map(int, time_str.split(':')))
    return parts[0] * 3600 + parts[1] * 60 + parts[2]

# Function to generate HTML content
def generate_html(video_url, chapters):
    html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Chapters</title>
</head>
<body>
    <h1>Video Chapters</h1>
    <video id="video" width="600" controls>
        <source src="{video_url}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <ul>
    '''
    for chap in chapters:
        chapter_num, chapter_name, start_time, end_time = chap
        start_seconds = format_time_to_seconds(start_time)
        html_content += f'<li><a href="#" onclick="document.getElementById(\'video\').currentTime={start_seconds};document.getElementById(\'video\').play();">Chapter {chapter_num}: {chapter_name} (Starts at {start_time}, Ends at {end_time})</a></li>\n'
    
    html_content += '''
    </ul>
</body>
</html>
'''
    return html_content

# Generate HTML content
video_url = FILE_URL  # Use the correct path to your video file
html_output = generate_html(video_url, chapters)

# Save the HTML content to a file
with open('video_chapters.html', 'w') as file:
    file.write(html_output)

print("HTML file 'video_chapters.html' has been generated.")