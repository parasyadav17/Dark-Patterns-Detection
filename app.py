from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from typing import Optional
import uvicorn

# Define the FastAPI app
app = FastAPI()

# Load the trained model and vectorizer
try:
    with open('xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model or vectorizer: {str(e)}")

# Define the input schema for prediction
class URLInput(BaseModel):
    url: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Deceptive URL Detection API!"}

@app.post("/check-url")
def check_url(input_data: URLInput):
    try:
        # Extract the URL from the input data
        url = input_data.url

        if not url:
            raise HTTPException(status_code=400, detail="No URL provided.")

        # Simulating text scraping (Replace with actual scraping logic if needed)
        scraped_text = scrape_text_from_url(url)

        if not scraped_text:
            raise HTTPException(status_code=400, detail="Unable to scrape content from the URL.")

        # Vectorize the scraped text
        text_vector = vectorizer.transform([scraped_text])

        # Predict the class
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0][1] * 100

        return {
            "url": url,
            "prediction": "Deceptive" if prediction == 1 else "Non-Deceptive",
            "probability": f"{probability:.2f}%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Function to scrape text from a given URL (Mock implementation for now)
def scrape_text_from_url(url: str) -> Optional[str]:
    import requests
    from bs4 import BeautifulSoup

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text if text.strip() else None
    except Exception as e:
        print(f"Error while scraping: {str(e)}")
        return None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
