---

# 🧠 Machine Learning Model Integration & Deployment

This project demonstrates the deployment of a pre-trained PyTorch model as a production-ready API using **FastAPI**. The API accepts structured input data and returns a model prediction. The app is containerized using Docker for consistent, portable deployment across any environment.

---

## 📌 Task Overview

As part of the second question in the technical assessment, this task required:

✅ Integrating a pre-trained PyTorch model  
✅ Exposing it as a RESTful API using FastAPI  
✅ Accepting structured JSON input and returning predictions  
✅ Packaging the solution into a Docker container  
✅ Providing production-readiness through best practices  

---

## 🚀 Key Features

- **FastAPI Backend** – Lightweight and asynchronous REST API.
- **PyTorch Model Integration** – Uses a saved `.pt` model for inference.
- **Structured Input** – Accepts numerical feature arrays via POST request.
- **Dockerized App** – Fully containerized for seamless deployment.
- **JSON Responses** – Returns clean, user-friendly prediction output.

---

## 📁 Project Structure

```
ML_model_api/
├── app/
│   ├── main.py           # FastAPI server and routes
│   ├── model.py          # Load and use the PyTorch model
│   ├── requirements.txt  # Python dependencies
├── Dockerfile            # Docker container instructions
├── README.md             # Documentation
└── screenshots/          # Demo screenshots
```

---

## 🛠️ Setup Instructions

### 🔹 Option 1: Run Locally (Without Docker)

1. Install Python packages:
```bash
pip install -r app/requirements.txt
```

2. Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8001/docs) to access the interactive Swagger UI.

---

### 🔹 Option 2: Run via Docker

1. Build the Docker image:
```bash
docker build -t fastapi-pytorch-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 fastapi-pytorch-api
```

Visit: [http://localhost:8000/docs](http://localhost:8001/docs)

---

## 📬 Example Request & Response

### ✅ POST `/predict`
**Input (JSON):**
```json
{
  "values": [0.5, 0.8, 0.1, 0.4, 0.6, 0.9, 0.7, 0.2, 0.3, 0.1]
}
```

**Output (JSON):**
```json
{
  "prediction": 1
}
```

---

## 🧪 Testing the API

Test the endpoint via:
- **Swagger UI** (`/docs`)
- **Postman**
- **Curl**:
```bash
curl -X POST http://localhost:8000/predict \
 -H "Content-Type: application/json" \
 -d '{"values":[0.5,0.8,0.1,0.4,0.6,0.9,0.7,0.2,0.3,0.1]}'
```

---

## 📷 Screenshots

| Swagger UI | Postman |
|------------|---------|
| ![Swagger Screenshot](screenshot_get1) | ![Postman Screenshot](Screenshot_postman) |
| ![Swagger Screenshot](screenshot_get2) | 
| ![Swagger Screenshot](screenshots_predict_demo1) | 
| ![Swagger Screenshot](screenshots_predict_demo2) | 
| ![Swagger Screenshot](screenshots_predict_demo3) | 

---

## 🧰 Production Notes

- **Versioning**: Maintain versioned model files (e.g., `model_v1.pt`) for tracking and rollbacks.
- **Logging**: Add structured logging (e.g., `loguru`) to monitor requests and errors.
- **Validation**: Add input validation and type checking with Pydantic (already used by FastAPI).
- **Monitoring**: Integrate Prometheus or similar tools for tracking model performance in production.

---

## 📦 Deployment Checklist

✅ FastAPI with Uvicorn  
✅ Dockerized Application  
✅ Local + Containerized Testing  
✅ Sample Requests & Screenshots  
✅ Project README & GitHub Hosting  

---

## 📄 License

This project is open-source and available under the MIT License.

---
