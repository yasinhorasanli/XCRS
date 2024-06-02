## Setup

Make sure to install the dependencies:

```bash
# pip
pip install -U google-generativeai 
pip install -U voyageai
pip install openai
pip install mistralai
pip install cohere

pip install fastapi
pip install pandas

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
