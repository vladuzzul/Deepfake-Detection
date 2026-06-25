# Final-Project
The project for my AI Engineer Course at SDA


The link to the used dataset: https://www.kaggle.com/datasets/chuneeb/deepfake-detection-dataset-2026
The link to the notebook this code is based on: https://www.kaggle.com/code/miadul/deepfake-detection-using-efficientnet-and-explaina

Run locally:

```bash
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
```

Run with Docker:

```bash
docker build -t deepfake-detection .
docker run --rm -p 8000:8000 deepfake-detection
```

Open the app in the browser at:

```text
http://localhost:8000
```