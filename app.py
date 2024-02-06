from flask import Flask, render_template, request, jsonify
import threading, uuid
from datetime import datetime
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import re
import random, time
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from operator import itemgetter
from langchain_community.document_loaders import JSONLoader

import os
os.environ["OPENAI_API_KEY"] = "your_api_key"


app = Flask(__name__)
def get_video_id_from_url(url):
    video_id = re.search(r'(?<=v=)[^&#]+', url)
    if video_id is None:
        video_id = re.search(r'(?<=be/)[^&#]+', url)
    return video_id.group(0) if video_id else None

def save_transcript_to_file(video_url, output_file, session_id):
    video_id = get_video_id_from_url(video_url)
    if video_id is None:
        print("Invalid YouTube URL")
        return

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    local_timestamps = transcripts_with_timestamps[session_id] = {}
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in transcript:
            f.write(entry["text"] + " ")
            local_timestamps[entry['text']] = entry['start']

def get_timestamp(session_id, sentence):
    # Check if the sentence exists in the transcript and print the timestamp
    for transcript_sentence, timestamp in transcripts_with_timestamps[session_id].items():
        if sentence in transcript_sentence:
            #print(f"A sentence containing '{sentence}' starts at {timestamp} seconds in the video.")
            return int(timestamp)
    return None

# Global variable to store the index
index = None
transcripts_with_timestamps = {}
video_urls = {}

@app.route('/')
def home():
    return render_template('index.html')

# Global variable to store the indexes
indexes = {}

@app.route('/save_transcript', methods=['POST'])
def save_transcript():
    start_time = time.time()  # Start the timer
    session_id = str(uuid.uuid4())  # Generate a unique ID for this session
    transcript_file = f"transcripts/{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(0, 100)}.txt"
    video_url = request.form['video_url']
    video_urls[session_id] = video_url.split("&")[0]
    save_transcript_to_file(video_url, transcript_file, session_id)
    print(f"Save transcript took {time.time() - start_time} seconds to execute.")
    start_time = time.time()  # Start the timer

    loader = TextLoader(transcript_file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Split QA texts took {time.time() - start_time} seconds to execute.")
    start_time = time.time()  # Start the timer

    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read()
    texts_sum = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_text(transcript)
    print(f"Split Sum texts took {time.time() - start_time} seconds to execute.")
    start_time = time.time()  # Start the timer
    # Create Document objects from the transcript parts
    docs = [Document(page_content=t) for t in texts_sum[:3]]
    print(len(texts_sum))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), chain_type="stuff",
                                     retriever=docsearch.as_retriever(
                                         # search_type="similarity_score_threshold",
                                         # search_kwargs={'k': 5, 'score_threshold': 0.4}
                                     ),
                                     return_source_documents=True)

    indexes[session_id] = qa
    print(f"QA Embedding took {time.time() - start_time} seconds to execute.")
    start_time = time.time()  # Start the timer

    # Load the summarization chain
    chain = load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce")
    summary = chain.run(docs)
    print(f"Summarization took {time.time() - start_time} seconds to execute.")
    # print(summary)
    return jsonify({'session_id': session_id, 'summary': summary})

@app.route('/query', methods=['POST'])
def query():
    start_time = time.time()  # Start the timer
    
    session_id = request.form['session_id']  # The client must send the session ID with each request
    if session_id in indexes:
        user_query = request.form['query']
        output = indexes[session_id]({"query": user_query})

        result = output["result"]
        print(f"Answering took {time.time() - start_time} seconds to execute.")
        clip_links = []
        for doc in output["source_documents"]:
            timestamp = get_timestamp(session_id, doc.page_content[:20])
            if timestamp is not None:  # If the timestamp is not None, add it to the timestamp_str
                link = video_urls[session_id]+"&t="+str(timestamp)+"s"
                clip_links.append(link)

        return jsonify({'result': result, 'clip_links': clip_links, 'clip_content': [doc.page_content for doc in output['source_documents']]})
    else:
        return "No transcript loaded", 400


if __name__ == '__main__':
    app.run(debug=True)
    # print("start")
