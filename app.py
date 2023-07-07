from potassium import Potassium, Request, Response

from transformers import pipeline
import torch
from io import BytesIO
import os
import base64
import soundfile as sf
import scipy.signal as sps
import numpy as np


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
    SAMPLE_RATE = 16000
    model = context.get("model")

    mp3BytesString = request.get('mp3BytesString', None)
    if mp3BytesString == None:
        return {'message': "No input provided"}

    output, samplerate = sf.read(BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1"))))
    if samplerate != SAMPLE_RATE:
        number_of_samples = round(len(output) * float(SAMPLE_RATE) / samplerate)
        output = sps.resample(output, number_of_samples)
    output = np.array(output).astype(np.float32)

    # Run the model
    result = model.transcribe(output)
    os.remove("input.mp3")

    return Response(
        json=result,
        status=200
    )


if __name__ == "__main__":
    app.serve()
