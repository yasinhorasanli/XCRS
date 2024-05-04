## Setup

Make sure to install the dependencies:

```bash
# pip
pip install google-generativeai 
pip install -U voyageai
pip install fastapi
...

```
## Run
Run the backend with uvicorn

```bash
# go to backend/src directory
cd backend/src

```

Start the backend server on `http://localhost:8000`:


```bash
# run 
uvicorn main:app --reload 

```

Check `http://localhost:8000/docs`
