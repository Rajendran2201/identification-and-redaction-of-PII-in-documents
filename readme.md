# PII Identification System 

This project is a **PII (Personally Identifiable Information) Identification System** that uses **YOLOv8** for detecting sensitive information and provides an interactive **React + Tailwind CSS** frontend, along with a **FastAPI backend** for inference and API management.

---

## 🛠 Tech Stack
- **Backend**
  - [FastAPI](https://fastapi.tiangolo.com/) - REST API framework
  - [YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection model for PII identification
  - Python 3.9+
- **Frontend**
  - [React](https://react.dev/) - UI library
  - [Tailwind CSS](https://tailwindcss.com/) - Styling
- **Other**
  - Node.js & npm
  - Virtual environment (`venv`) for backend dependencies

---

## Project Structure
├── backend
│ ├── main.py # FastAPI app entry point
│ ├── requirements.txt # Python dependencies
│ ├── weights/ # YOLOv8 model weights
│ └── venv/ # Python virtual environment (ignored in Git)
│
├── frontend
│ ├── src/ # React components
│ ├── public/ # Static files
│ ├── package.json # Frontend dependencies
│ ├── tailwind.config.js # Tailwind configuration
│ └── postcss.config.js # PostCSS config
│
├── .gitignore
├── README.md
└── package.json


---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

###  2. Backend Setup
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn main:app --reload

## Frontend Setup
cd frontend

# Install dependencies
npm install

# Start development server
npm start
