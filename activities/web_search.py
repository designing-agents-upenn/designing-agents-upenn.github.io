import json
import os
from urllib.request import Request, urlopen
import re

def search_and_read(query, num_results=3, max_chars=500):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("Missing TAVILY_API_KEY environment variable.")
        return []

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": num_results,
        "search_depth": "basic",
        "include_raw_content": True,
    }
    request = Request(
        "https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        response = urlopen(request, timeout=20).read().decode("utf-8")
        data = json.loads(response)
    except Exception as exc:
        print(f"Tavily request failed: {exc!r}")
        return []

    output = []
    for item in data.get("results", []):
        url = item.get("url")
        content = item.get("raw_content") or item.get("content") or ""
        if url:
            output.append({"url": url, "text": content[:max_chars]})
    return output

def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    text = text.replace("  ", " ")
    #remove markdown and html tags
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"<.*?>", "", text)
    return text


if __name__ == "__main__":
    pages = search_and_read("Charli XCX winning a Grammy 2026", num_results=10, max_chars=2000)
    if not pages:
        print("No results found.")
    for page in pages:
        print(clean_text(page["text"]) or "(No text extracted)")
        print("\n---\n")