#!/bin/bash

mkdir -p {api,model,app/pages,data,assets}

touch api/{main.py,inference.py,gradcam.py}
touch model/{train.py,evaluate.py,dataset.py,test.py}
touch app/{streamlit_app.py,pages/model_info.py}
touch {Dockerfile,docker-compose.yml,requirements.txt,README.md,.gitignore}

echo "Project structure created successfully."