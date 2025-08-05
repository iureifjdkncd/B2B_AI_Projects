@echo off
rem Set the title of the CMD window
title ATECH predictionAPI
echo Activating virtual environment...


rem Activate the virtual environment
call "C:\virtualenvs\PredictionAPIEnv\Scripts\activate" 

echo Virtual environment activated.

echo Starting uvicorn server...

rem Start the uvicorn server
call uvicorn main:app --host 0.0.0.0 --port 25000

echo uvicorn server started.
