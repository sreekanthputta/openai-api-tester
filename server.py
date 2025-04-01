import openai
import uvicorn
import os
import json # For formatting embedding/moderation results
from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uuid # For unique audio filenames

# Initialize FastAPI app
app = FastAPI()

# Set OpenAI API Key
OPENAI_API_KEY = "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"
client = openai.OpenAI(api_key=OPENAI_API_KEY)  # ✅ Use OpenAI Client

# File storage setup
UPLOAD_FOLDER = Path("uploads")
AUDIO_OUTPUT_FOLDER = Path("static/audio_out") # Serve generated audio from static
UPLOAD_FOLDER.mkdir(exist_ok=True)
AUDIO_OUTPUT_FOLDER.mkdir(exist_ok=True)


# Static files & templates setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """Render the HTML page."""
    # Pass the list of models to the template (optional, but good practice)
    # models = [...] # Could fetch dynamically if needed
    return templates.TemplateResponse("index.html", {"request": request})

# Serve generated audio files
@app.get("/static/audio_out/{filename}")
async def get_audio_file(filename: str):
    file_path = AUDIO_OUTPUT_FOLDER / filename
    if file_path.is_file():
        return FileResponse(file_path)
    return HTMLResponse(status_code=404, content="File not found")


@app.post("/generate/")
async def generate(
    request: Request,
    model: str = Form(...),
    prompt: str = Form(None), # Optional depending on model
    # file: UploadFile = File(None), # Old generic file upload
    image_file: UploadFile = File(None), # Specific for image models (future)
    audio_file: UploadFile = File(None)  # Specific for audio models
):
    """
    Processes user input based on the selected model.
    """
    context = {"request": request, "selected_model": model} # Keep track of selected model for reload
    uploaded_file_path = None
    temp_audio_path = None # For processing uploaded audio

    try:
        # --- Input Handling ---
        if audio_file:
            # Save uploaded audio temporarily for processing
            temp_audio_path = UPLOAD_FOLDER / f"temp_{uuid.uuid4()}_{audio_file.filename}"
            with open(temp_audio_path, "wb") as f:
                f.write(await audio_file.read())
            uploaded_file_path = str(temp_audio_path) # Show user the temp path

        # --- Model Logic ---
        if model in ["dall-e-2", "dall-e-3"]:
            if not prompt: raise ValueError("Prompt is required for DALL-E models.")
            image_response = client.images.generate(
                model=model,
                prompt=prompt,
                n=1,
                size="1024x1024" # Consider making size dynamic later
            )
            context["generated_image_url"] = image_response.data[0].url

        elif model in ["gpt-4o-mini-search-preview", "gpt-4o-mini"]:
             if not prompt: raise ValueError("Prompt is required for GPT models.")
             # Basic text generation for now
             # TODO: Add image handling for gpt-4o-mini vision later
             response = client.chat.completions.create(
                 model=model,
                 messages=[{"role": "user", "content": prompt}]
             )
             context["generated_text"] = response.choices[0].message.content

        elif model in ["gpt-4o-mini-audio-preview", "whisper-1"]:
            if not temp_audio_path: raise ValueError("Audio file is required for transcription models.")
            with open(temp_audio_path, "rb") as audio_data:
                transcript = client.audio.transcriptions.create(
                    model=model,
                    file=audio_data
                )
            context["generated_text"] = transcript.text # Whisper returns a transcript object

        elif model == "tts-1":
            if not prompt: raise ValueError("Prompt is required for TTS models.")
            response = client.audio.speech.create(
                model=model,
                voice="alloy", # Or other voices: echo, fable, onyx, nova, shimmer
                input=prompt
            )
            # Save the audio stream to a file
            output_filename = f"{uuid.uuid4()}.mp3"
            output_path = AUDIO_OUTPUT_FOLDER / output_filename
            response.stream_to_file(output_path)
            context["generated_audio_url"] = f"/static/audio_out/{output_filename}" # URL path

        elif model in ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"]:
            if not prompt: raise ValueError("Prompt is required for embedding models.")
            response = client.embeddings.create(
                model=model,
                input=prompt
                # encoding_format="float" # or "base64"
            )
            # Format embedding for display (e.g., show first few dimensions)
            embedding_preview = response.data[0].embedding[:5] # Show first 5 dimensions
            context["generated_text"] = f"Embedding (first 5 dimensions):\n{embedding_preview}\n...\nTotal dimensions: {len(response.data[0].embedding)}"

        elif model == "omni-moderation-latest":
            if not prompt: raise ValueError("Prompt is required for moderation models.")
            response = client.moderations.create(
                model=model, # Though 'latest' is often implied if model omitted
                input=prompt
            )
            # Format moderation results for display
            context["generated_text"] = json.dumps(response.results[0].model_dump(), indent=2) # Use model_dump() for pydantic v2

        else:
            context["error"] = f"Model '{model}' is not supported yet."

        # Add uploaded file path info if relevant
        if uploaded_file_path:
             context["uploaded_file_path"] = uploaded_file_path


    except Exception as e:
        context["error"] = f"An error occurred: {str(e)}"

    finally:
        # Clean up temporary uploaded audio file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return templates.TemplateResponse("index.html", context)


# Run server on 0.0.0.0
if __name__ == "__main__":
    # Ensure API key is set (better to use environment variables in production)
    if not OPENAI_API_KEY or "sk-proj-" not in OPENAI_API_KEY: # Basic check
       print("Error: OPENAI_API_KEY not found or invalid.")
       # Consider exiting or raising an error
    else:
        print("OpenAI API Key loaded.")
        # Use import string for reload to work correctly
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
