import pdfplumber
from openai import OpenAI
from config import API_KEY, BASE_URL, DEFAULT_MODEL

def pdf_to_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def query_model(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prompt to the SAIA LLM and return the response text."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # === CONFIGURE ===
    pdf_file = "C:/Master thesis/files/pdf/rag.pdf"  # Replace with your PDF path
    model_name = DEFAULT_MODEL
    user_prompt = "Summarize the following document in a few sentences:"

    # === PROCESS PDF ===
    pdf_text = pdf_to_text(pdf_file)
    full_prompt = f"{user_prompt}\n\n{pdf_text}"

    # === QUERY MODEL ===
    result = query_model(full_prompt, model=model_name)

    # === OUTPUT ===
    print("=== Model Response ===")
    print(result)

    # Optional: save response to file
    with open("pdf_summary.txt", "w", encoding="utf-8") as f:
        f.write(result)
