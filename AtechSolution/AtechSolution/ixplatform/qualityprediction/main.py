from fastapi import FastAPI
from business_logic.models.prediction import Prediction
import uvicorn
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/get_prediction")
async def get_prediction(working_number):
    working_number = str(working_number)
    res=Prediction(working_number)
    return res
if __name__ == '__main__':
    uvicorn.run("main:app",host="0.0.0.0",port=25000,reload=True)
