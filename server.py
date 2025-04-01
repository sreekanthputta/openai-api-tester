import openai
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()

# Set OpenAI API Key
OPENAI_API_KEY = "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"
client = openai.OpenAI(api_key=OPENAI_API_KEY)  # ✅ Use OpenAI Client

# File storage setup
UPLOAD_FOLDER = "uploads"
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# Static files & templates setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """Render the HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate/")
async def generate(request: Request, prompt: str = Form(...), file: UploadFile = File(None)):
    """
    Processes user input, generates AI text & image.
    """
    try:
        file_path = None
        if file:
            file_path = f"{UPLOAD_FOLDER}/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())

        # ✅ Generate text from GPT-4 Turbo (New API Format)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        text_response = response.choices[0].message.content

        # ✅ Generate image from DALL·E (New API Format)
        image_response = client.images.generate(
            model="dall-e-3",  # ✅ Ensure latest model
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = image_response.data[0].url

        return templates.TemplateResponse("index.html", {
            "request": request,
            "generated_text": text_response,
            "generated_image_url": image_url,
            "uploaded_file_path": file_path
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e)
        })

# Run server on 0.0.0.0
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

