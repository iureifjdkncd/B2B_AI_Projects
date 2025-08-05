from fastapi import FastAPI
from business_logic.models import prediction
import uvicorn
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/get_prediction")
async def get_prediction(machine_number, uni_cavity):
    print("before calling")
    machine_number=int(machine_number) # machine number int
    uni_cavity = str(uni_cavity)       # uni_cavity must be string format 
    res=prediction.PredictionNeonent(machine_number,uni_cavity)
    return res
if __name__ == '__main__':
    uvicorn.run("main:app",host="0.0.0.0",port=25000,reload=True)
