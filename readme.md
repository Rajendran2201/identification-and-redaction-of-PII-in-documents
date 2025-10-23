# PII Identification System

The PII (Personally Identifiable Information) Identification System is an end-to-end application designed to detect sensitive information using **Donut model & YOLOv8**. It provides a scalable **FastAPI backend** for model inference and API management, along with a modern **React + Tailwind CSS frontend** for user interaction.

---

## Features

* PII detection powered by YOLOv8 object detection.
* REST API backend built with FastAPI for high-performance inference.
* React-based frontend with Tailwind CSS for a clean and responsive interface.
* Modular architecture separating backend and frontend for ease of development and deployment.
* Support for virtual environments and reproducible dependency management.

---

## Tech Stack

**Backend**

* FastAPI (REST API framework)
* YOLOv8 (Ultralytics) for object detection
* Python 3.9+

**Frontend**

* React
* Tailwind CSS

**Other**

* Node.js & npm
* Python virtual environment (`venv`)

---

## Project Structure

```
PII-Identification-System/
├── backend
│   ├── main.py              # FastAPI application entry point
│   ├── requirements.txt     # Python dependencies
│   ├── weights/             # YOLOv8 model weights
│   └── venv/                # Python virtual environment (ignored in Git)
│
├── frontend
│   ├── src/                 # React components
│   ├── public/              # Static files
│   ├── package.json         # Frontend dependencies
│   ├── tailwind.config.js   # Tailwind configuration
│   └── postcss.config.js    # PostCSS configuration
│
├── .gitignore
├── README.md
└── package.json
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn main:app --reload
```

The backend will be available at: `http://127.0.0.1:8000`

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

The frontend will be available at: `http://localhost:3000`

---

## Usage

1. Start both the backend and frontend servers.
2. Open the frontend in your browser at `http://localhost:3000`.
3. Upload an image or document containing sensitive information.
4. The system will run inference and display detected PII regions.

---

## API Endpoints

| Method | Endpoint    | Description                                                  |
| ------ | ----------- | ------------------------------------------------------------ |
| POST   | `/predict/` | Accepts an image for inference and returns detected PII data |
| GET    | `/health/`  | Health check endpoint                                        |

---

## Dependencies

* **Backend**: FastAPI, Uvicorn, Ultralytics (YOLOv8), Pydantic
* **Frontend**: React, Tailwind CSS

For the full list, refer to `requirements.txt` and `package.json`.

---

## Architecture

```mermaid
graph TD
    %% Frontend Section
    subgraph A[Frontend - React + Tailwind CSS]
        style A fill:#F0F9FF,stroke:#38BDF8,stroke-width:1.5px,color:#0369A1
        A1[User Interface]:::frontend -->|Uploads documents / images| A2[Request Handler]:::frontend
    end

    %% Backend Section
    subgraph B[Backend - FastAPI]
        style B fill:#FFFBEB,stroke:#FACC15,stroke-width:1.5px,color:#854D0E
        B1[API Gateway / Router]:::backend
        B2[Inference Engine]:::backend
        B3[Model Manager]:::backend
        B4[Preprocessing & Postprocessing]:::backend
        B1 --> B2
        B2 --> B3
        B2 --> B4
    end

    %% Model Section
    subgraph C[Models]
        style C fill:#FEF2F2,stroke:#F87171,stroke-width:1.5px,color:#7F1D1D
        C1[YOLOv8 - Object Detection]:::model
        C2[Donut Model - Document Parsing]:::model
        B3 --> C1
        B3 --> C2
    end

    %% Storage & System Components
    subgraph D[System Components]
        style D fill:#ECFDF5,stroke:#4ADE80,stroke-width:1.5px,color:#065F46
        D1[File Storage / Uploads]:::storage
        D2[Logs & Metadata]:::storage
        D3[Virtual Environment / Dependency Management]:::storage
    end

    %% Connections
    A2 -->|Sends REST API Requests - JSON / File Upload| B1
    B4 -->|PII Results - Detected Fields, Bounding Boxes| A1
    B2 --> D2
    B1 --> D1
    B --> D3

    %% External Integrations
    subgraph E[Deployment & Infrastructure]
        style E fill:#F5F3FF,stroke:#A78BFA,stroke-width:1.5px,color:#4C1D95
        E1[Docker / Containerization]:::infra
        E2[CI/CD Pipeline]:::infra
    end

    D3 --> E1
    E1 --> E2

    %% Styles
    classDef frontend fill:#E0F2FE,stroke:#38BDF8,color:#0369A1;
    classDef backend fill:#FEF9C3,stroke:#FACC15,color:#854D0E;
    classDef model fill:#FEE2E2,stroke:#F87171,color:#7F1D1D;
    classDef storage fill:#DCFCE7,stroke:#4ADE80,color:#065F46;
    classDef infra fill:#EDE9FE,stroke:#A78BFA,color:#4C1D95;


```

## Contributing

Contributions are encouraged. Please fork the repository, make your changes, and submit a pull request for review.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

