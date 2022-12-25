from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

class Input(BaseModel):
    text: str

class Output(BaseModel):
    sentiment: str

# Load the saved model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(inputs: Input):
    # Use the model to predict the sentiment of the input text
    prediction = model.predict(inputs.text)
    return Output(sentiment=prediction)

# Run the FastAPI server
if __name__ == '__revised__':
    app.run(host='0.0.0.0', port=8000)