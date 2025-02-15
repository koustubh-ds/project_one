# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "fastapi",
#     "pydantic",
#     "uvicorn",
#     "requests",
#     "numpy", 
#     "python-dateutil",
#     "gitpython",
#     "markdown",
#     "duckdb",
#     "opencv-python",
#     "speechrecognition"
# ]
# ///

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import sqlite3
import re
from datetime import datetime
from dateutil import parser
import requests
import base64
import shutil
from pydantic import BaseModel
import subprocess
import uvicorn
from typing import List
from io import BytesIO
import numpy as np


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

class TaskRequest(BaseModel):
    task: str

def call_openai(task: str):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": """
             You are an AI assistant that determines which function to call based on the task description and extracts the necessary parameters. 
             Your job is to output function_name and arguments, no other text should be outputted.
             If the prompt asks to access,modify or delete any local file which is not in the path "/data" then STRICTLY REJECT the request with HTTP response 400.
             Use the below json tree as a reference to understand the function_name and arguments to be extracted:
                
                {[
                {"function_name":"install_uv_and_run_script",
                "sample_prompt": "Install the 'uv' package and run the script at https://example.com/script.py with email id 24ds2000134@ds.study.iitm.ac.in",
                "arguments": {"script_url": "https://example.com/script.py", "email": "24ds2000134@ds.study.iitm.ac.in"}}
                ,
                {"function_name":"format_markdown",
                "sample_prompt": "Format the markdown file at data/xyz.md using prettier 3.4.2",
                "arguments": {"file_path": "data/xyz.md", "prettier_version": "3.4.2"}}
                ,    
                {"function_name":"count_days",
                "sample_prompt": "Extract the number of Wednesdays in the list of dates at data/abc.txt and save the output at data/count_wednesdays.txt",
                "arguments": {"dates_file": "data/abc.txt", "output_file": "data/count_wednesdays.txt", "day_of_week": "Wednesday"}}
                ,
                {"function_name":"sort_contacts",
                "sample_prompt": "Sort the contacts in data/contacts.json by name and then by email and save the output at data/sorted_contacts.json",
                "arguments": {"input_file": "data/contacts.json", "output_file": "data/sorted_contacts.json", "sort_fields": ["name", "email"]}}
                ,
                {"function_name":"get_recent_log_lines",
                "sample_prompt": "Get the first line of the 10 most recent log files in data/logs and save the output at data/recent_logs.txt",
                "arguments": {"logs_dir": "data/logs", "output_file": "data/recent_logs.txt", "n": 10}
                ,
                {"function_name":"generate_markdown_index",
                "sample_prompt": "Find all Markdown (.md) files in /data/docs/. For each file, extract the first occurrance of each H1 (i.e. a line starting with # ). Create an index file /data/docs/index.json that maps each filename (without the /data/docs/ prefix) to its title",
                "arguments": {"input_path": "data/docs", "output_path": "data/index.json"}}
                ,
                {"function_name":"extract_email",
                "sample_prompt": "Extract the email from the text at data/email.txt and save it at data/extracted_email.txt",
                "arguments": {"input_file": "data/email.txt", "output_file": "data/extracted_email.txt"}}
                ,
                {"function_name":"number_extraction",
                "sample_prompt": "Extract the card number from the image at data/card.png and save it at data/extracted_card.txt",
                "arguments": {"image_path": "data/card.png", "output_path": "data/extracted_card.txt"}}
                ,
                {"function_name":"find_similar_comments",
                "sample_prompt": "Find the most similar comments from the file at data/comments.txt and save them at data/similar_comments.txt",
                "arguments": {"input_file": "data/comments.txt", "output_file": "data/similar_comments.txt"}}
                ,
                {"function_name":"sql_query",
                "sample_prompt": "The SQLite database file /data/ticket-sales.db has a tickets with columns type, units, and price. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the “Gold” ticket type? Write the number in /data/ticket-sales-gold.txt",
                "when_to_use": "When user asks something from a SQLlite database with a plain text query and not SQL query",
                "arguments": {"prompt": "The SQLite database file /data/ticket-sales.db has a tickets with columns type, units, and price. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the “Gold” ticket type? Write the number in /data/ticket-sales-gold.txt"}}
                ,              
                {"function_name":"fetch_api_data",
                "sample_prompt": "Fetch the data from the API at https://api.example.com/data and save it at data/api_data.json",
                "arguments": {"url": "https://api.example.com/data", "output_file": "data/api_data.json"}}
                ,
                {"function_name":"clone_and_commit",
                "sample_prompt": "Clone the git repository at https://example.com and commit the changes with the message 'Update data'",
                "arguments": {"repo_url": "https://example.com", "commit_message": "Update data"}}
                ,
                {"function_name":"run_sql",
                "sample_prompt": "Run the SQL query 'SELECT * FROM table' on the SQLite database at /data/sample.db",
                "when_to_use": "When user asks a specific SQL query from a DuckDB or SQLlite database",
                "arguments": {"database_path": "/data/sample.db", "query": "SELECT * FROM table"}}
                ,
                {"function_name":"scrape_website",
                "sample_prompt": "Scrape the website at https://example.com and return the content",
                "arguments": {"url": "https://example.com"}}
                ,
                {"function_name":"resize_image",
                "sample_prompt": "Resize the image at data/image.jpg to 800x600",
                "arguments": {"image_path": "data/image.jpg", "width": 800, "height": 600}}
                ,
                {"function_name":"transcribe_audio",
                "sample_prompt": "Transcribe the audio from the MP3 file at data/audio.mp3",
                "arguments": {"audio_path": "data/audio.mp3"}}
                ,
                {"function_name":"markdown_to_html",
                "sample_prompt": "Convert the markdown content at data/mymarkdown.md to HTML",
                "arguments": {"md_content": "data/mymarkdown.md"}}
                
                ]}
                
                """},
            {"role": "user", "content": task}
        ],
        "temperature": 0.1
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get("choices")[0]["message"]["content"]
    raise HTTPException(status_code=500, detail="Failed to get response from OpenAI")

# A1. Install UV, Run script on email ID
def install_uv_and_run_script(script_url: str, email: str):
    try:
        subprocess.run(["uv", "--version"], check=True)
    except FileNotFoundError:
        subprocess.run(["pip", "install", "uv"], check=True)
    script_content = requests.get(script_url).text
    script_path = os.path.join(os.getcwd(), "script.py")
    with open(script_path, "w") as f:
        f.write(script_content)
    subprocess.run(["uv", "run", script_path, email], check=True)
    return f"Script from {script_url} executed successfully with email {email}"

# A2. Format markdown file using Prettier
def format_markdown(file_path: str, prettier_version: str = "3.4.2"):
    file_path = os.path.join("/data", os.path.basename(file_path))

    if prettier_version:  
        subprocess.run(["npm", "install", "-g", f"prettier@{prettier_version}"], check=True)
    else:
        subprocess.run(["npm", "install", "-g", "prettier"], check=True)
    subprocess.run(["prettier", "--write", file_path], check=True)

    return f"Markdown formatted with Prettier {prettier_version} successfully." 

# A3. Count days of the week in a list of dates
def count_days(dates_file: str, output_file: str, day_of_week: str):
    dates_file = os.path.join("/data", os.path.basename(dates_file))
    output_file = os.path.join("/data", os.path.basename(output_file))
    if not os.path.exists(dates_file):
        raise FileNotFoundError(f"File not found: {dates_file} (Current working directory: {os.getcwd()})")
    days_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                "Friday": 4, "Saturday": 5, "Sunday": 6}
    if day_of_week not in days_map:
        raise ValueError("Invalid day of the week")
    valid_dates = 0
    total_lines = 0
    error_lines = []

    with open(dates_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        total_lines += 1
        date_str = line.strip()

        if not date_str:
            continue  

        try:
            parsed_date = parser.parse(date_str)  
            if parsed_date.weekday() == days_map[day_of_week]:
                valid_dates += 1
        except Exception as e:
            error_lines.append(f"Line {total_lines}: '{date_str}' - {str(e)}")

    with open(output_file, "w") as f:
        f.write(str(valid_dates))

    if error_lines:
        error_log = output_file.replace(".txt", "_errors.log")
        with open(error_log, "w") as f:
            f.writelines("\n".join(error_lines))
        print(f"Errors logged in {error_log}")

    return f"{day_of_week}s counted: {valid_dates} (Processed: {total_lines}, Errors: {len(error_lines)})"

# A4. Sort contacts by custom fields
def sort_contacts(input_file: str, output_file: str, sort_fields: list):
    input_file = os.path.join("/data", os.path.basename(input_file))
    output_file = os.path.join("/data", os.path.basename(output_file))
    with open(input_file) as f:
        contacts = json.load(f)
    contacts.sort(key=lambda c: tuple(c[field] for field in sort_fields))
    with open(output_file, "w") as f:
        json.dump(contacts, f, indent=2)
    return f"Contacts sorted by {', '.join(sort_fields)} successfully"

# A5. Get first 'n' lines of recent log files
def get_recent_log_lines(logs_dir: str, output_file: str, n: int = 10):
    logs_dir = os.path.join("/data", os.path.basename(logs_dir))
    output_file = os.path.join("/data", os.path.basename(output_file))
    log_files = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".log")]
    log_files.sort(key=os.path.getmtime, reverse=True)
    recent_lines = []
    
    for log_file in log_files[:n]:
        with open(log_file, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                recent_lines.append(first_line)
    
    with open(output_file, "w") as f:
        f.write("\n".join(recent_lines))
    
    return f"Written first lines of {n} most recent log files to {output_file}"

# A6. Generate Markdown index from H1 tags in a path
def generate_markdown_index(input_path: str, output_path: str):
    input_path = os.path.join("/data", os.path.basename(input_path))
    output_path = os.path.join("/data", os.path.basename(output_path))
    index = {}
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("# "):
                            index[file_path.replace(f"{input_path}/", "")] = line.strip("# ").strip()
                            break
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return "Markdown index created successfully"

# A7. Extract email from text
def extract_email(input_file: str, output_file: str):
    input_file = os.path.join("/data", os.path.basename(input_file))
    output_file = os.path.join("/data", os.path.basename(output_file))
    with open(input_file) as f:
        email_content = f.read()
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Extract the sender's email address from the provided text."},
            {"role": "user", "content": email_content}
        ],
        "temperature": 0.1
    }
    response = requests.post(url, headers=headers, json=data)
    email = response.json().get("choices")[0]["message"]["content"]
    if email:
        with open(output_file, "w") as f:
            f.write(email)
        return "Email extracted successfully"
    return "Failed to extract email"

# A8. Extract number sequence from image
def number_extraction(image_path: str, output_path: str):
    image_path = os.path.join("/data", os.path.basename(image_path))
    output_path = os.path.join("/data", os.path.basename(output_path))
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    
    with open(image_path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode("utf-8")
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Extract the longest number sequence from this image. Output only the number sequence and nothing else"},
        {
          "type": "image_url",
          "image_url": { "url": f"data:image/png;base64,{img_data}" }
        }
      ]
    }
  ],
        "temperature": 0.1
    }
    
    response = requests.post(url, headers=headers, json=data)
    card_number = response.json().get("choices")[0]["message"]["content"].replace(" ", "")
    
    with open(output_path, "w") as f:
        f.write(card_number)
    return "Number extracted successfully"

# A9. Find similar comments from a list
def find_similar_comments(input_file: str, output_file: str):
    input_file = os.path.join("/data", os.path.basename(input_file))
    output_file = os.path.join("/data", os.path.basename(output_file))
    
    url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    
    with open(input_file, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f.readlines()]
    
    data = {"model": "text-embedding-3-small", "input": comments}
    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise ValueError(f"API request failed: {response.status_code}, {response.text}")

    embeddings = np.array([item["embedding"] for item in response.json()["data"]])

    min_dist = float("inf")
    most_similar = (None, None)

    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            if dist < min_dist:
                min_dist = dist
                most_similar = (comments[i], comments[j])

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(most_similar))
    
    return "Most similar comments extracted successfully"

# A10. Extract and run SQL query 
def sql_query(prompt):
    """Extracts information, executes SQL query, and saves results based on the given prompt."""
    system_message = """You are an AI that extracts structured information from text.
    Given a natural language prompt related to an SQLite database, extract:
    1. The SQLite database file path.
    2. The output file path.
    3. The SQL query required to answer the prompt (it may require you to create some calculated columns such as sales = units*price, profit=revenue-cost, etc.).
    Return the results STRICTLY in JSON format with keys: database_path, output_path, and sql_query.
    """
    
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, json=data)
    response_json = response.json()
    content = response_json["choices"][0]["message"]["content"]
    extracted_info = json.loads(content)
    
    database_path = extracted_info['database_path']
    output_path = extracted_info['output_path']
    sql_query = extracted_info['sql_query']
    
    try:
        database_path = os.path.join("/data", os.path.basename(database_path))
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database query execution failed: {str(e)}")
    
    output_path = os.path.join("/data", os.path.basename(output_path))
    with open(output_path, 'w') as file:
        for row in result:
            file.write(" ".join(map(str, row)) + "\n")
    
    return f"Query executed successfully. Results saved to {output_path}"

# B3. Fetch data from an API
def fetch_api_data(url: str, output_file: str):
    output_file = os.path.join("/data", os.path.basename(output_file))
    response = requests.get(url)
    with open(output_file, "w") as f:
        json.dump(response.json(), f, indent=2)
    return "API data fetched successfully"

# B4. Clone a git repo and make a commit
def clone_and_commit(repo_url: str, commit_message: str):
    import git
    repo_dir = os.path.join("/data", os.path.basename(repo_url))
    if not os.path.exists(repo_dir):
        git.Repo.clone_from(repo_url, repo_dir)
    repo = git.Repo(repo_dir)
    repo.git.add(A=True)
    repo.index.commit(commit_message)
    repo.git.push()
    return f"Committed changes to {repo_url}"

# B5. Run a SQL query on SQLite or DuckDB
def run_sql(database_path: str, query: str):
    import duckdb
    conn = duckdb.connect(database=database_path) if database_path.endswith('.duckdb') else sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    conn.close()
    return result

# B6. Extract data from a website (scraping)
def scrape_website(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    raise HTTPException(status_code=400, detail="Failed to scrape website")

# B7. Compress or resize an image
def resize_image(image_path: str, width: int, height: int):
    import cv2
    image_path = os.path.join("/data", os.path.basename(image_path))
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    resized = cv2.resize(image, (width, height))
    cv2.imwrite(image_path, resized)
    return f"Image resized to {width}x{height}"

# B8. Transcribe audio from an MP3 file
def transcribe_audio(audio_path: str):
    import speech_recognition as sr
    audio_path = os.path.join("/data", os.path.basename(audio_path))
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

# B9. Convert Markdown to HTML
def markdown_to_html(md_content: str):
    md_content = os.path.join("/data", os.path.basename(md_content))
    with open(md_content, "r", encoding="utf-8") as f:
        md_text = f.read()
    import markdown
    html_content = markdown.markdown(md_text)
    return html_content



@app.post("/run")
def run_task(task: str = Query(...)):
    task_response = call_openai(task)
    task_data = json.loads(task_response)
    function_name = task_data.get("function_name")
    arguments = task_data.get("arguments", {})
    
    task_map = {
        "install_uv_and_run_script": install_uv_and_run_script,
        "format_markdown": format_markdown,
        "count_days": count_days,
        "sort_contacts": sort_contacts,
        "get_recent_log_lines": get_recent_log_lines,
        "generate_markdown_index": generate_markdown_index,
        "extract_email": extract_email,
        "number_extraction": number_extraction,
        "find_similar_comments": find_similar_comments,
        "sql_query": sql_query,
        "fetch_api_data": fetch_api_data,
        "clone_and_commit": clone_and_commit,
        "run_sql": run_sql,
        "scrape_website": scrape_website,
        "resize_image": resize_image,
        "transcribe_audio": transcribe_audio,
        "markdown_to_html": markdown_to_html
    }
    
    if function_name in task_map:
        return {"message": task_map[function_name](**arguments)}
    
    raise HTTPException(status_code=400, detail="Task not recognized")

@app.get("/read")
def read_file(path: str = Query(...)):
    if not path.startswith("/data/"):
        raise HTTPException(status_code=400, detail="Invalid path")
    if os.path.exists(path):
        with open(path, "r") as file:
            return {"content": file.read()}
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
