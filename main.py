import joblib

def gender_features(word):
    return {'last_letter':word[-1]}


loaded_model = joblib.load('finalized_model.sav')

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, you can predict the gender of a person given his first name"}

@app.get("/{prenom}")
async def read_item(prenom: str):
    return {"sexe": loaded_model.classify(gender_features(prenom))}
