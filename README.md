# KIVALGO

## How to run the app
1. Create a local Docker image from the Dockerfile with `docker build -t kivalgo .`
2. Start the container from the image, ensuring to map the port 8501 `docker run -p 8501:8501 kivalgo`

## Manual running 
1. Create a virtual environment by doing `python3 -m venv venv`
2. Then go into the virtual environment: `source venv/bin/activate`
3. Install the packages required: `pip3 install -r requirements.txt`
4. You will need to install git-lfs to download model locally on machine
- On Mac with `brew install git-lfs`
5. `cd models` go to the models folder, and clone the model which will be used: `git clone https://huggingface.co/microsoft/Phi-3.5-mini-instruct` and `git clone https://huggingface.co/thenlper/gte-small`

To run the app, simply do
`streamlit run streamlit_app.py`

## Data
This directory contains data to be used to create vector database.

## Loaders
This directory contains modules responsible for loading different types of data.

## Models
This directory contains the model which will be used

## Routes
This directory is intended to contain the route (blueprint) definitions for the application.
Its like a higher level abstraction of the app, where the actual app will make POST-request to start the predictions etc.

## Services
This directory is intended to contain service modules. Services usually encapsulate business logic and interact with data models or external APIs.

## Templates
This directory contains the actual HTMl to render the app on the web

## Utils
This directory contains different kinds of helper functions 



