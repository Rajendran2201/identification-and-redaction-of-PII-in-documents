from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
from io import BytesIO
import tempfile
import os
import uuid
from typing import List, Optional, Dict, Any
import logging
import json

# Presidio imports
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# OCR imports
import pytesseract
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available. Install with: pip install easyocr")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced PII Redaction API with Presidio", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000", 
        "http://localhost:5173",
        "https://*.vercel.app",  # Add this for Vercel
        "https://your-app-name.vercel.app"  # Replace with your actual domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
yolo_model = None
presidio_analyzer = None
presidio_anonymizer = None
ocr_reader = None

class YOLOModel:
    def __init__(self, model_url="https://anonify-pii-model.s3.ap-south-1.amazonaws.com/best.pt", model_path="/tmp/best.pt"):
        """Initialize YOLO model with S3 download"""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Use /tmp directory for Vercel serverless
            if not os.path.exists(model_path):
                logger.info(f"Downloading model from S3: {model_url}")
                
                # Create directory if needed
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                # Download with timeout and error handling
                try:
                    response = requests.get(model_url, stream=True, timeout=300)
                    response.raise_for_status()
                    
                    total_size = int(response.headers.get('content-length', 0))
                    logger.info(f"Downloading model ({total_size / (1024*1024):.1f} MB)...")
                    
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    logger.info("Model downloaded successfully")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to download model: {e}")
                    raise
            
            # Load the model
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.model.to(self.device)
                self.model_type = "ultralytics"
                logger.info(f"YOLO model loaded successfully on {self.device}")
            except ImportError:
                # Fallback to torch hub if ultralytics not available
                try:
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)
                    self.model.to(self.device)
                    self.model_type = "torch_hub"
                    logger.info(f"YOLO model loaded via torch.hub on {self.device}")
                except Exception as e:
                    logger.error(f"Failed to load YOLO model: {e}")
                    raise
                    
        except Exception as e:
            logger.error(f"Error initializing YOLO model: {e}")
            raise

    def predict_and_get_boxes(self, image):
        """Get bounding boxes without redaction"""
        try:
            img_array = np.array(image)
            
            if self.model_type == "ultralytics":
                results = self.model(img_array)
                boxes = []
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            if confidence > 0.5:
                                boxes.append({
                                    'x1': int(x1), 'y1': int(y1),
                                    'x2': int(x2), 'y2': int(y2),
                                    'confidence': float(confidence),
                                    'type': 'visual_pii'
                                })
            else:  # torch_hub
                results = self.model(img_array)
                predictions_df = results.pandas().xyxy[0]
                boxes = []
                for _, row in predictions_df.iterrows():
                    if row['confidence'] > 0.5:
                        boxes.append({
                            'x1': int(row['xmin']), 'y1': int(row['ymin']),
                            'x2': int(row['xmax']), 'y2': int(row['ymax']),
                            'confidence': float(row['confidence']),
                            'type': 'visual_pii'
                        })
            
            return boxes
        except Exception as e:
            logger.error(f"Error in YOLO prediction: {e}")
            return []

class PresidioModel:
    def __init__(self):
        """Initialize Presidio analyzer and anonymizer"""
        try:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            logger.info("Presidio model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Presidio: {e}")
            raise

    def analyze_text(self, text: str, language: str = 'en') -> List[Dict]:
        """Analyze text for PII"""
        try:
            results = self.analyzer.analyze(text=text, language=language)
            return [
                {
                    'entity_type': result.entity_type,
                    'start': result.start,
                    'end': result.end,
                    'confidence': result.score,
                    'text': text[result.start:result.end]
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return []

    def anonymize_text(self, text: str, method: str = 'replace', language: str = 'en') -> str:
        """Anonymize PII in text"""
        try:
            results = self.analyzer.analyze(text=text, language=language)
            
            # Define anonymization operators
            operators = {}
            if method == 'redact':
                operators = {"DEFAULT": OperatorConfig("redact")}
            elif method == 'hash':
                operators = {"DEFAULT": OperatorConfig("hash")}
            elif method == 'mask':
                operators = {"DEFAULT": OperatorConfig("mask", {"chars_to_mask": 3, "masking_char": "*"})}
            else:  # replace (default)
                operators = {"DEFAULT": OperatorConfig("replace")}
            
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=operators
            )
            
            return anonymized_result.text
        except Exception as e:
            logger.error(f"Error anonymizing text: {e}")
            return text

class OCRService:
    def __init__(self):
        """Initialize OCR service"""
        self.ocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en'])
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")

    def extract_text_with_coordinates(self, image) -> List[Dict]:
        """Extract text with bounding box coordinates"""
        try:
            img_array = np.array(image)
            text_regions = []
            
            # Try EasyOCR first (more accurate coordinates)
            if self.ocr_reader is not None:
                try:
                    results = self.ocr_reader.readtext(img_array)
                    for (bbox, text, confidence) in results:
                        if confidence > 0.5 and text.strip():
                            # Convert bbox to x1, y1, x2, y2 format
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            x1, y1 = int(min(x_coords)), int(min(y_coords))
                            x2, y2 = int(max(x_coords)), int(max(y_coords))
                            
                            text_regions.append({
                                'text': text.strip(),
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'confidence': float(confidence)
                            })
                    return text_regions
                except Exception as e:
                    logger.warning(f"EasyOCR failed: {e}, falling back to Tesseract")
            
            # Fallback to Tesseract
            try:
                # Get detailed text data from Tesseract
                data = pytesseract.image_to_data(img_array, output_type=pytesseract.Output.DICT)
                
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    conf = int(data['conf'][i])
                    
                    if text and conf > 50:  # Filter low confidence
                        x1, y1 = data['left'][i], data['top'][i]
                        w, h = data['width'][i], data['height'][i]
                        x2, y2 = x1 + w, y1 + h
                        
                        text_regions.append({
                            'text': text,
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'confidence': conf / 100.0
                        })
                
                return text_regions
                
            except Exception as e:
                logger.error(f"Tesseract OCR failed: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Error in OCR text extraction: {e}")
            return []

    def extract_plain_text(self, image) -> str:
        """Extract plain text without coordinates"""
        try:
            img_array = np.array(image)
            
            if self.ocr_reader is not None:
                try:
                    results = self.ocr_reader.readtext(img_array)
                    texts = [text for (_, text, confidence) in results if confidence > 0.5]
                    return ' '.join(texts)
                except Exception:
                    pass
            
            # Fallback to Tesseract
            try:
                return pytesseract.image_to_string(img_array).strip()
            except Exception as e:
                logger.error(f"Text extraction failed: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"Error extracting plain text: {e}")
            return ""

class HybridPIIProcessor:
    def __init__(self, yolo_model, presidio_model, ocr_service):
        self.yolo_model = yolo_model
        self.presidio_model = presidio_model
        self.ocr_service = ocr_service

    def process_image(self, image, anonymization_method='replace', combine_detections=True):
        """Process image with both YOLO and Presidio detection"""
        try:
            all_boxes = []
            
            # 1. YOLO visual detection
            if self.yolo_model:
                visual_boxes = self.yolo_model.predict_and_get_boxes(image)
                all_boxes.extend(visual_boxes)
            
            # 2. OCR + Presidio text detection
            if combine_detections and self.ocr_service and self.presidio_model:
                text_regions = self.ocr_service.extract_text_with_coordinates(image)
                
                for region in text_regions:
                    text = region['text']
                    pii_entities = self.presidio_model.analyze_text(text)
                    
                    if pii_entities:  # If PII found in this text region
                        all_boxes.append({
                            'x1': region['x1'], 'y1': region['y1'],
                            'x2': region['x2'], 'y2': region['y2'],
                            'confidence': region['confidence'],
                            'type': 'text_pii',
                            'entities': pii_entities,
                            'original_text': text
                        })
            
            # 3. Apply redaction
            redacted_image = self._apply_redaction(image, all_boxes, anonymization_method)
            
            return redacted_image, all_boxes
            
        except Exception as e:
            logger.error(f"Error in hybrid processing: {e}")
            return image, []

    def _apply_redaction(self, image, boxes, method='redact'):
        """Apply redaction to image based on detected boxes"""
        img_array = np.array(image)
        
        for box in boxes:
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            
            if method == 'redact' or method == 'blackout':
                # Black rectangle
                cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 0, 0), -1)
            elif method == 'blur':
                # Blur the region
                roi = img_array[y1:y2, x1:x2]
                blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                img_array[y1:y2, x1:x2] = blurred_roi
            elif method == 'replace' and box.get('type') == 'text_pii':
                # Replace text with anonymized version
                try:
                    original_text = box.get('original_text', '')
                    anonymized_text = self.presidio_model.anonymize_text(original_text, method='replace')
                    
                    # Draw white rectangle and add anonymized text
                    cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 255, 255), -1)
                    
                    # Add text (simplified - you might want to improve font handling)
                    pil_img = Image.fromarray(img_array)
                    draw = ImageDraw.Draw(pil_img)
                    try:
                        font = ImageFont.truetype("arial.ttf", 12)
                    except:
                        font = ImageFont.load_default()
                    
                    draw.text((x1, y1), anonymized_text, fill=(0, 0, 0), font=font)
                    img_array = np.array(pil_img)
                    
                except Exception as e:
                    logger.warning(f"Text replacement failed, using blackout: {e}")
                    cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        return Image.fromarray(img_array)

# Initialize services
@app.on_event("startup")
async def startup_event():
    global yolo_model, presidio_analyzer, presidio_anonymizer, ocr_reader
    try:
        # Initialize YOLO model
        try:
            yolo_model = YOLOModel("weights/best.pt")
            logger.info("YOLO model initialized successfully")
        except Exception as e:
            logger.warning(f"YOLO model initialization failed: {e}")
            yolo_model = None
        
        # Initialize Presidio
        try:
            presidio_model = PresidioModel()
            presidio_analyzer = presidio_model.analyzer
            presidio_anonymizer = presidio_model.anonymizer
            logger.info("Presidio initialized successfully")
        except Exception as e:
            logger.warning(f"Presidio initialization failed: {e}")
            presidio_analyzer = None
            presidio_anonymizer = None
        
        # Initialize OCR
        try:
            ocr_reader = OCRService()
            logger.info("OCR service initialized successfully")
        except Exception as e:
            logger.warning(f"OCR initialization failed: {e}")
            ocr_reader = None
            
    except Exception as e:
        logger.error(f"Startup failed: {e}")

# Create global processor
def get_hybrid_processor():
    presidio_model = PresidioModel() if presidio_analyzer else None
    return HybridPIIProcessor(yolo_model, presidio_model, ocr_reader)

# Utility functions (keeping existing ones)
def convert_to_rgb(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def create_pdf_from_images(images: List[Image.Image]) -> BytesIO:
    try:
        pdf_buffer = BytesIO()
        rgb_images = [convert_to_rgb(img) for img in images]
        
        if rgb_images:
            rgb_images[0].save(
                pdf_buffer, 
                format='PDF', 
                save_all=True, 
                append_images=rgb_images[1:] if len(rgb_images) > 1 else [],
                quality=95
            )
        
        pdf_buffer.seek(0)
        return pdf_buffer
    except Exception as e:
        logger.error(f"Error creating PDF: {e}")
        raise

def extract_images_from_pdf(pdf_bytes: bytes) -> List[Image.Image]:
    try:
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
    except Exception as e:
        logger.error(f"Error extracting images from PDF: {e}")
        raise

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Enhanced PII Redaction API with Presidio",
        "yolo_model_loaded": yolo_model is not None,
        "presidio_loaded": presidio_analyzer is not None,
        "ocr_available": ocr_reader is not None
    }

@app.post("/redact-image-hybrid")
async def redact_image_hybrid(
    file: UploadFile = File(...),
    method: str = Form("redact"),  # redact, blur, replace
    combine_detections: bool = Form(True)
):
    """Enhanced endpoint: YOLO + Presidio hybrid detection"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        processor = get_hybrid_processor()
        redacted_image, detection_info = processor.process_image(
            image, 
            anonymization_method=method,
            combine_detections=combine_detections
        )
        
        # Return image
        img_buffer = BytesIO()
        redacted_image.save(img_buffer, format='PNG', quality=95)
        img_buffer.seek(0)
        
        return StreamingResponse(
            BytesIO(img_buffer.read()),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=hybrid_redacted_{file.filename}"}
        )
    
    except Exception as e:
        logger.error(f"Error in hybrid image redaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/anonymize-text")
async def anonymize_text(
    text: str = Form(...),
    method: str = Form("replace"),  # replace, redact, hash, mask
    language: str = Form("en")
):
    """Endpoint: Pure text anonymization"""
    try:
        if not presidio_analyzer:
            raise HTTPException(status_code=503, detail="Presidio not available")
        
        presidio_model = PresidioModel()
        
        # Get analysis results
        pii_entities = presidio_model.analyze_text(text, language)
        
        # Get anonymized text
        anonymized_text = presidio_model.anonymize_text(text, method, language)
        
        return {
            "original_text": text,
            "anonymized_text": anonymized_text,
            "detected_entities": pii_entities,
            "method": method
        }
    
    except Exception as e:
        logger.error(f"Error in text anonymization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-and-anonymize-text")
async def extract_and_anonymize_text(
    file: UploadFile = File(...),
    method: str = Form("replace")
):
    """Endpoint: OCR + Text anonymization"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if not ocr_reader or not presidio_analyzer:
            raise HTTPException(status_code=503, detail="OCR or Presidio not available")
        
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Extract text
        extracted_text = ocr_reader.extract_plain_text(image)
        
        if not extracted_text.strip():
            return {
                "extracted_text": "",
                "anonymized_text": "",
                "detected_entities": [],
                "message": "No text found in image"
            }
        
        # Anonymize text
        presidio_model = PresidioModel()
        pii_entities = presidio_model.analyze_text(extracted_text)
        anonymized_text = presidio_model.anonymize_text(extracted_text, method)
        
        return {
            "extracted_text": extracted_text,
            "anonymized_text": anonymized_text,
            "detected_entities": pii_entities,
            "method": method
        }
    
    except Exception as e:
        logger.error(f"Error in OCR + text anonymization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Keep existing endpoints for backward compatibility
@app.post("/redact-image")
async def redact_single_image(file: UploadFile = File(...)):
    """Original endpoint: YOLO-only detection"""
    try:
        if not yolo_model:
            raise HTTPException(status_code=503, detail="YOLO model not loaded")
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Use YOLO only
        boxes = yolo_model.predict_and_get_boxes(image)
        processor = get_hybrid_processor()
        redacted_image = processor._apply_redaction(image, boxes, 'redact')
        
        img_buffer = BytesIO()
        redacted_image.save(img_buffer, format='PNG', quality=95)
        img_buffer.seek(0)
        
        return StreamingResponse(
            BytesIO(img_buffer.read()),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=redacted_{file.filename}"}
        )
    
    except Exception as e:
        logger.error(f"Error in single image redaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ... (keep other existing endpoints: redact-pdf, redact-multiple-images)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "yolo_model": yolo_model is not None,
            "presidio_analyzer": presidio_analyzer is not None,
            "ocr_service": ocr_reader is not None
        }
    }

@app.post("/redact-pdf")
async def redact_pdf(
    file: UploadFile = File(...),
    method: str = Form("redact"),
    hybrid_mode: bool = Form(True)
):
    """Enhanced PDF endpoint with hybrid detection"""
    try:
        if file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        pdf_bytes = await file.read()
        extracted_images = extract_images_from_pdf(pdf_bytes)
        
        if not extracted_images:
            raise HTTPException(status_code=400, detail="No images found in PDF")
        
        redacted_images = []
        processor = get_hybrid_processor()
        
        for image in extracted_images:
            if hybrid_mode and yolo_model and presidio_analyzer:
                redacted_image, _ = processor.process_image(
                    image, 
                    anonymization_method=method,
                    combine_detections=True
                )
            elif yolo_model:
                # YOLO only
                boxes = yolo_model.predict_and_get_boxes(image)
                redacted_image = processor._apply_redaction(image, boxes, method)
            else:
                # No processing available
                redacted_image = image
            
            redacted_images.append(redacted_image)
        
        pdf_buffer = create_pdf_from_images(redacted_images)
        
        return StreamingResponse(
            BytesIO(pdf_buffer.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=hybrid_redacted_{file.filename}"}
        )
    
    except Exception as e:
        logger.error(f"Error in PDF redaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/redact-multiple-images")
async def redact_multiple_images(
    files: List[UploadFile] = File(...),
    method: str = Form("redact"),
    hybrid_mode: bool = Form(True)
):
    """Enhanced multiple images endpoint with hybrid detection"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        redacted_images = []
        processor = get_hybrid_processor()
        
        for file in files:
            if not file.content_type.startswith('image/'):
                continue
            
            image_bytes = await file.read()
            image = Image.open(BytesIO(image_bytes))
            
            if hybrid_mode and yolo_model and presidio_analyzer:
                redacted_image, _ = processor.process_image(
                    image,
                    anonymization_method=method,
                    combine_detections=True
                )
            elif yolo_model:
                # YOLO only
                boxes = yolo_model.predict_and_get_boxes(image)
                redacted_image = processor._apply_redaction(image, boxes, method)
            else:
                # No processing available
                redacted_image = image
            
            redacted_images.append(redacted_image)
        
        if not redacted_images:
            raise HTTPException(status_code=400, detail="No valid images found")
        
        pdf_buffer = create_pdf_from_images(redacted_images)
        
        return StreamingResponse(
            BytesIO(pdf_buffer.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=hybrid_redacted_images.pdf"}
        )
    
    except Exception as e:
        logger.error(f"Error in multiple images redaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """Endpoint: Analyze image and return detection info without redaction"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        results = {
            "visual_detections": [],
            "text_detections": [],
            "extracted_text": "",
            "pii_entities": []
        }
        
        # YOLO detection
        if yolo_model:
            visual_boxes = yolo_model.predict_and_get_boxes(image)
            results["visual_detections"] = visual_boxes
        
        # OCR + Presidio detection
        if ocr_reader and presidio_analyzer:
            # Extract text with coordinates
            text_regions = ocr_reader.extract_text_with_coordinates(image)
            results["text_detections"] = text_regions
            
            # Extract plain text
            plain_text = ocr_reader.extract_plain_text(image)
            results["extracted_text"] = plain_text
            
            # Analyze for PII
            if plain_text.strip():
                presidio_model = PresidioModel()
                pii_entities = presidio_model.analyze_text(plain_text)
                results["pii_entities"] = pii_entities
        
        return results
    
    except Exception as e:
        logger.error(f"Error in image analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported-entities")
async def get_supported_entities():
    """Get list of PII entities supported by Presidio"""
    try:
        if not presidio_analyzer:
            raise HTTPException(status_code=503, detail="Presidio not available")
        
        # Get supported entities
        supported_entities = presidio_analyzer.get_supported_entities()
        
        return {
            "supported_entities": supported_entities,
            "entity_descriptions": {
                "PERSON": "Names of people",
                "EMAIL_ADDRESS": "Email addresses",
                "PHONE_NUMBER": "Phone numbers",
                "CREDIT_CARD": "Credit card numbers",
                "CRYPTO": "Cryptocurrency addresses",
                "DATE_TIME": "Dates and times",
                "IBAN_CODE": "International Bank Account Numbers",
                "IP_ADDRESS": "IP addresses",
                "LOCATION": "Geographic locations",
                "MEDICAL_LICENSE": "Medical license numbers",
                "NRP": "Norwegian personal numbers",
                "SSN": "Social Security Numbers",
                "UK_NHS": "UK National Health Service numbers",
                "US_BANK_NUMBER": "US bank account numbers",
                "US_DRIVER_LICENSE": "US driver license numbers",
                "US_ITIN": "US Individual Taxpayer Identification Numbers",
                "US_PASSPORT": "US passport numbers",
                "URL": "Web URLs"
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting supported entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/custom-anonymization")
async def custom_anonymization(
    text: str = Form(...),
    entities_to_anonymize: str = Form(""),  # Comma-separated list
    anonymization_config: str = Form("{}"),  # JSON string
    language: str = Form("en")
):
    """Custom anonymization with specific entity types and methods"""
    try:
        if not presidio_analyzer:
            raise HTTPException(status_code=503, detail="Presidio not available")
        
        # Parse entities to anonymize
        target_entities = []
        if entities_to_anonymize.strip():
            target_entities = [e.strip().upper() for e in entities_to_anonymize.split(",")]
        
        # Parse anonymization config
        try:
            config = json.loads(anonymization_config) if anonymization_config.strip() else {}
        except json.JSONDecodeError:
            config = {}
        
        # Analyze text
        analyzer_results = presidio_analyzer.analyze(text=text, language=language)
        
        # Filter by target entities if specified
        if target_entities:
            analyzer_results = [
                result for result in analyzer_results 
                if result.entity_type in target_entities
            ]
        
        # Build operators config
        operators = {}
        for entity_type in set(result.entity_type for result in analyzer_results):
            if entity_type in config:
                entity_config = config[entity_type]
                operator_type = entity_config.get("type", "replace")
                operator_params = entity_config.get("params", {})
                operators[entity_type] = OperatorConfig(operator_type, operator_params)
            else:
                operators[entity_type] = OperatorConfig("replace")
        
        # Anonymize
        anonymized_result = presidio_anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=operators
        )
        
        return {
            "original_text": text,
            "anonymized_text": anonymized_result.text,
            "detected_entities": [
                {
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "confidence": result.score,
                    "text": text[result.start:result.end]
                }
                for result in analyzer_results
            ],
            "applied_config": config
        }
    
    except Exception as e:
        logger.error(f"Error in custom anonymization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)