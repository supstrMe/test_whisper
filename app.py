from potassium import Potassium, Request, Response

from transformers import pipeline
import torch
from io import BytesIO
import os
import base64


app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline(
        model='openai/whisper-base',
        chunk_length_s=30,
        device=device,
    )
    context = {
        "model": model
    }

    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    outputs = model(prompt)

    mp3BytesString = request.get('mp3BytesString', None)
    if mp3BytesString == None:
        return {'message': "No input provided"}

    mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
    with open('input.mp3', 'wb') as file:
        file.write(mp3Bytes.getbuffer())

    # Run the model
    result = model.transcribe("input.mp3")
    os.remove("input.mp3")

    return Response(
        json=result,
        status=200
    )


if __name__ == "__main__":
    app.serve()
