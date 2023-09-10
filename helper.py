from PyPDF2 import PdfReader

import os
from dotenv import load_dotenv
import json
import requests
from bs4 import BeautifulSoup
import re

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")

# 1. Extract Text from pdf
def extractTextFromPdf(uploaded_file):
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        text = ''
        for page_num in range(len(reader.pages) ):
            page =reader.pages[page_num]
            text += page.extract_text()                          
        return text

# 3. get job URL or description
def scrape_website(url: str):
    print("Scraping website...")
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    data = {
        "url": url
    }
    data_json = json.dumps(data)
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        return cleaned_text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


