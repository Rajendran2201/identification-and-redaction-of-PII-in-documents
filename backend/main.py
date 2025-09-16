from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from io import BytesIO
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PII Redaction API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None


class YOLOModel:
    def __init__(self, model_path="weights/best.pt"):
        """Initialize YOLOv8 model"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            from ultralytics import YOLO  # import YOLOv8

            self.model = YOLO(model_path)
            logger.info(f"YOLOv8 model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            raise

    def predict_and_redact(self, image: Image.Image) -> Image.Image:
        """Predict PII locations and redact them"""
        try:
            img_array = np.array(image)

            # Run inference
            results = self.model(img_array)

            # Redact detected regions
            redacted_img = img_array.copy()
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

                for (x1, y1, x2, y2), conf in zip(boxes, confs):
                    if conf > 0.5:  # confidence threshold
                        cv2.rectangle(
                            redacted_img,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 0, 0),
                            -1,
                        )

            return Image.fromarray(redacted_img)

        except Exception as e:
            logger.error(f"Error in predict_and_redact: {e}")
            raise


@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = YOLOModel("weights/best.pt")
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def create_pdf_from_images(images: List[Image.Image]) -> BytesIO:
    """Create PDF from list of PIL images"""
    pdf_buffer = BytesIO()
    rgb_images = [convert_to_rgb(img) for img in images]

    if rgb_images:
        rgb_images[0].save(
            pdf_buffer,
            format="PDF",
            save_all=True,
            append_images=rgb_images[1:] if len(rgb_images) > 1 else [],
            quality=95,
        )

    pdf_buffer.seek(0)
    return pdf_buffer


def extract_images_from_pdf(pdf_bytes: bytes) -> List[Image.Image]:
    """Extract images from PDF"""
    images = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        image = Image.open(BytesIO(img_data))
        images.append(image)

    pdf_document.close()
    return images


@app.get("/")
async def root():
    return {"message": "PII Redaction API is running", "model_loaded": model is not None}


@app.post("/redact-image")
async def redact_single_image(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    redacted_image = model.predict_and_redact(image)

    img_buffer = BytesIO()
    redacted_image.save(img_buffer, format="PNG", quality=95)
    img_buffer.seek(0)

    return StreamingResponse(
        img_buffer,
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename=redacted_{file.filename}"},
    )


@app.post("/redact-pdf")
async def redact_pdf(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    pdf_bytes = await file.read()
    extracted_images = extract_images_from_pdf(pdf_bytes)

    if not extracted_images:
        raise HTTPException(status_code=400, detail="No images found in PDF")

    redacted_images = [model.predict_and_redact(img) for img in extracted_images]
    pdf_buffer = create_pdf_from_images(redacted_images)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=redacted_{file.filename}"},
    )


@app.post("/redact-multiple-images")
async def redact_multiple_images(files: List[UploadFile] = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    redacted_images = []
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        redacted_images.append(model.predict_and_redact(image))

    if not redacted_images:
        raise HTTPException(status_code=400, detail="No valid images found")

    pdf_buffer = create_pdf_from_images(redacted_images)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=redacted_images.pdf"},
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(model.device) if model else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
