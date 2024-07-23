from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class TextModel(BaseModel):
    ticket: str


# load model and tokenizer
loaded_model = AutoModelForSequenceClassification.from_pretrained("./intelligence/model")
loaded_tokenizer = AutoTokenizer.from_pretrained("./intelligence/model")

@app.post("/api/v1/classify_text")
async def ClassifyText(payload: TextModel):
    try:
        inputs = loaded_tokenizer(payload.ticket, return_tensors="pt")
        with torch.no_grad():
            logits = loaded_model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        prediction = loaded_model.config.id2label[predicted_class_id]
        return JSONResponse(prediction, 200)
    except Exception as e:
        print("error occured while classifying Text: ", str(e))
        return JSONResponse("Server Error", 500)




if __name__ == '__main__':
       uvicorn.run("server:app", host="0.0.0.0")