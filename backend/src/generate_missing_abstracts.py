# pip install google-generativeai selenium beautifulsoup4 pandas python-dotenv

import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv
import google.generativeai as genai

# === LOAD API KEY ===
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
FAILED_PATH = os.path.join(DATA_DIR, "failed_articles.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "generated_abstracts.csv")

# === SELENIUM CONFIG ===
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=options)

# === FUNCTIONS ===
def extract_full_text(url):
    """Extracts the main text from the article page using Selenium and BeautifulSoup."""
    try:
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs if len(p.get_text()) > 50)
        return text.strip()[:7000]  # Limit length for API
    except Exception as e:
        print(f"Error extracting text from {url}: {e}")
        return None

def generate_summary(title, text):
    """Generates an academic-style abstract using Gemini."""
    try:
        prompt = f"""
You are a scientific assistant. Given the following article titled:
'{title}'

Generate an English academic-style abstract of maximum 200 words.
Do not include section titles or labels. Summarize in a single paragraph the main objective, methods, results, and conclusions.

Article text:
{text[:6000]}
"""
        response = model.generate_content(prompt)
        return response.text.strip() if response else None
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

# === PROCESSING ===
def generate_missing_abstracts():
    df = pd.read_csv(FAILED_PATH)
    print(f"Processing {len(df)} articles without abstracts...")

    generated_data = []

    for i, row in df.iterrows():
        title, link = row["title"], row["link"]
        print(f"({i+1}/{len(df)}) Generating abstract for: {title}")

        full_text = extract_full_text(link)
        if not full_text:
            print("Could not retrieve full text.")
            continue

        abstract = generate_summary(title, full_text)
        if abstract:
            generated_data.append({
                "title": title,
                "link": link,
                "generated_abstract": abstract
            })
            print("Abstract generated.")
        else:
            print("No abstract generated.")

        time.sleep(2)  # Avoid overloading API

    if generated_data:
        os.makedirs(DATA_DIR, exist_ok=True)
        pd.DataFrame(generated_data).to_csv(OUTPUT_PATH, index=False)
        print(f"\nGenerated abstracts saved to {OUTPUT_PATH}")
    else:
        print("\nNo new abstracts were generated.")

    driver.quit()

if __name__ == "__main__":
    generate_missing_abstracts()
