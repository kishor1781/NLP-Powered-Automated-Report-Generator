import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import PyPDF2
import fitz
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx import Presentation
from openpyxl import load_workbook
import cv2
import pytesseract
import easyocr
import numpy as np
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import yaml
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PIL import Image
import torch
from text_analyzer_advanced import TextAnalyzer

class DocumentProcessor:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with advanced LLM integration and GPU support."""
        self.config = self.load_config(config_path)
        self.base_dir = Path(self.config["input_dir"])
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index = defaultdict(list)
        self.summaries = {}
        self.embeddings = {}

        self.device = torch.device("cuda" if self.config["use_gpu"] and torch.cuda.is_available() else "cpu")
        logging.info(f"Device set to use {self.device}")

        self.text_analyzer = TextAnalyzer(self.config)
        self.easyocr_reader = easyocr.Reader(self.config["ocr_languages"], gpu=self.config["use_gpu"])
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.image_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=0 if self.config["use_gpu"] else -1)
        self.setup_logging()

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_logging(self) -> None:
        """Configure logging to file and console."""
        log_file = self.output_dir / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )

    def format_summary_text(self) -> str:
        """Format summary text for export."""
        summary_text = "DOCUMENT SUMMARY\n" + "=" * 70 + "\n\n"
        for file_path, summary in self.summaries.items():
            summary_text += (
                f"File: {summary['filename']}\nPath: {summary['path']}\nSize: {summary['size_mb']} MB\n"
                f"Modified: {summary['modified']}\nCreated: {summary['created']}\nType: {summary['type']}\n"
                f"Pages: {summary['pages']}\nHas Text: {summary['has_text']}\nHas Images: {summary['has_images']}\n"
                f"Content Summary: {summary['content_summary']}\nKey Points: {', '.join(summary['key_points'])}\n"
                f"Main Topics: {', '.join(summary['main_topics'])}\nImage Captions: {', '.join(summary['image_captions'])}\n"
                f"Category: {summary['category']}\n\n" + "=" * 70 + "\n\n"
            )
        return summary_text

    def scan_directory(self) -> Dict[str, List[str]]:
        """Scan directory for supported file types."""
        file_types = defaultdict(list)
        logging.info(f"Scanning directory: {self.base_dir}")
        supported_extensions = {'.pdf', '.docx', '.pptx', '.xlsx', '.jpg', '.jpeg', '.png', '.txt'}
        for file_path in self.base_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                file_types[file_path.suffix.lower()].append(str(file_path))
        logging.info(f"Found {sum(len(files) for files in file_types.values())} files")
        return dict(file_types)

    def create_file_summary(self, file_path: str) -> Dict:
        """Create a summary dictionary for a file."""
        path = Path(file_path)
        stats = path.stat()
        return {
            'filename': path.name, 'path': str(path), 'size_mb': round(stats.st_size / (1024 * 1024), 2),
            'modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
            'created': datetime.fromtimestamp(stats.st_ctime).isoformat(),
            'type': path.suffix.lower(), 'pages': self.get_page_count(path),
            'has_text': False, 'has_images': False, 'content_summary': '', 'key_points': [],
            'main_topics': [], 'image_captions': [], 'category': 'unknown'
        }

    def get_page_count(self, path: Path) -> int:
        """Get page/slide/sheet count based on file type."""
        try:
            ext = path.suffix.lower()
            if ext == '.pdf':
                with open(path, 'rb') as f:
                    return len(PyPDF2.PdfReader(f).pages)
            elif ext == '.docx':
                return len(Document(path).paragraphs)
            elif ext == '.pptx':
                return len(Presentation(path).slides)
            elif ext == '.xlsx':
                return len(load_workbook(path, read_only=True).sheetnames)
            return 1
        except Exception as e:
            logging.error(f"Error getting page count for {path}: {str(e)}")
            return 1

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for OCR and captioning."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance > 500:
            return image
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        return cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    @lru_cache(maxsize=5000)
    def extract_text_from_image(self, image_path: str) -> Tuple[str, str, float]:
        """Extract text and caption from an image file with caching."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return "", "Failed to read image", 0.0
            preprocessed = self.preprocess_image(image)
            tesseract_text = pytesseract.image_to_string(preprocessed, lang='+'.join(self.config["ocr_languages"]))
            easyocr_results = self.easyocr_reader.readtext(preprocessed)
            easyocr_text = " ".join([result[1] for result in easyocr_results])
            confidence = np.mean([result[2] for result in easyocr_results]) if easyocr_results else 0.7
            text = tesseract_text if len(tesseract_text) > len(easyocr_text) else easyocr_text
            caption = self.image_captioner(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))[0]['generated_text']
            return text, caption, confidence
        except Exception as e:
            logging.error(f"Error performing OCR on {image_path}: {str(e)}")
            return "", f"OCR failed: {str(e)}", 0.0

    def extract_text_from_image_array(self, image: np.ndarray) -> Tuple[str, str, float]:
        """Extract text and caption from an image array."""
        try:
            preprocessed = self.preprocess_image(image)
            tesseract_text = pytesseract.image_to_string(preprocessed, lang='+'.join(self.config["ocr_languages"]))
            easyocr_results = self.easyocr_reader.readtext(preprocessed)
            easyocr_text = " ".join([result[1] for result in easyocr_results])
            confidence = np.mean([result[2] for result in easyocr_results]) if easyocr_results else 0.7
            text = tesseract_text if len(tesseract_text) > len(easyocr_text) else easyocr_text
            caption = self.image_captioner(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))[0]['generated_text']
            return text, caption, confidence
        except Exception as e:
            logging.error(f"Error processing image array: {str(e)}")
            return "", f"OCR failed: {str(e)}", 0.0

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, bool, List[str]]:
        """Extract text and images from PDFs."""
        try:
            full_text, image_captions = [], []
            has_images = False
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                full_text.append(f"\n--- Page {page_num + 1} ---")
                full_text.append(page.get_text())
                for img in page.get_images(full=True):
                    has_images = True
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    nparr = np.frombuffer(base_image["image"], np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is not None:
                        img_text, img_caption, _ = self.extract_text_from_image_array(image)
                        image_captions.append(f"Image on page {page_num + 1}: {img_text or img_caption}")
            doc.close()
            return "\n".join(full_text), has_images, image_captions
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return "", False, []

    def extract_text_from_docx(self, docx_path: str) -> Tuple[str, bool, List[str]]:
        """Extract text and images from Word documents."""
        try:
            doc = Document(docx_path)
            text = [p.text for p in doc.paragraphs]
            image_captions = []
            has_images = False
            for rel in doc.part.rels.values():
                if rel.reltype == RT.IMAGE:
                    has_images = True
                    image_part = rel.target_part
                    image_bytes = image_part.blob
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img_text, img_caption, _ = self.extract_text_from_image_array(img)
                        if img_text or img_caption:
                            image_captions.append(f"Image: {img_text or img_caption}")
            return "\n".join(text), has_images, image_captions
        except Exception as e:
            logging.error(f"Error processing DOCX {docx_path}: {str(e)}")
            return "", False, []

    def extract_text_from_pptx(self, pptx_path: str) -> Tuple[str, bool, List[str]]:
        """Extract text and images from PowerPoint presentations."""
        try:
            prs = Presentation(pptx_path)
            text = []
            image_captions = []
            has_images = False
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text.append(shape.text)
                    elif shape.shape_type == 13:  # Picture
                        has_images = True
                        image = shape.image
                        image_bytes = image.blob
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            img_text, img_caption, _ = self.extract_text_from_image_array(img)
                            if img_text or img_caption:
                                image_captions.append(f"Image: {img_text or img_caption}")
            return "\n".join(text), has_images, image_captions
        except Exception as e:
            logging.error(f"Error processing PPTX {pptx_path}: {str(e)}")
            return "", False, []

    def extract_text_from_xlsx(self, xlsx_path: str) -> Tuple[str, bool, List[str]]:
        """Extract text from Excel files."""
        try:
            wb = load_workbook(xlsx_path, read_only=True)
            text = []
            for sheet in wb:
                for row in sheet.iter_rows():
                    for cell in row:
                        if cell.value:
                            text.append(str(cell.value))
            wb.close()
            return "\n".join(text), False, []
        except Exception as e:
            logging.error(f"Error processing XLSX {xlsx_path}: {str(e)}")
            return "", False, []

    def extract_text_content(self, file_path: str) -> Tuple[str, bool, List[str]]:
        """Extract content based on file type."""
        path = Path(file_path)
        ext = path.suffix.lower()
        try:
            if ext == '.txt':
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read(), False, []
            elif ext == '.pdf':
                return self.extract_text_from_pdf(str(path))
            elif ext == '.docx':
                return self.extract_text_from_docx(str(path))
            elif ext == '.pptx':
                return self.extract_text_from_pptx(str(path))
            elif ext == '.xlsx':
                return self.extract_text_from_xlsx(str(path))
            elif ext in ['.jpg', '.jpeg', '.png']:
                text, caption, _ = self.extract_text_from_image(str(path))
                return text, True, [caption]
            return "", False, []
        except Exception as e:
            logging.error(f"Error extracting text from {path}: {str(e)}")
            return "", False, []

    def process_file(self, file_path: str) -> Tuple[str, Dict]:
        """Process a single file with LLM enhancements."""
        summary = self.create_file_summary(file_path)
        content, has_images, image_captions = self.extract_text_content(file_path)
        if content or image_captions:
            full_content = '\n'.join([content] + image_captions)
            embedding = self.embedder.encode(full_content, batch_size=self.config["batch_size"])
            self.embeddings[file_path] = embedding
            self.index_keywords(full_content, file_path)
            summary.update({
                'has_text': bool(content), 'has_images': has_images, 'image_captions': image_captions,
                'content_summary': self.text_analyzer.summarize(full_content),
                'key_points': self.text_analyzer.extract_key_points(full_content),
                'main_topics': self.text_analyzer.extract_main_topics(full_content),
                'category': self.text_analyzer.classify_document(full_content)
            })
        return file_path, summary

    def process_documents(self) -> None:
        """Process all documents in parallel."""
        start_time = datetime.now()
        file_types = self.scan_directory()
        total_files = sum(len(files) for files in file_types.values())
        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            futures = [executor.submit(self.process_file, file_path) 
                       for file_list in file_types.values() for file_path in file_list]
            for future in tqdm(futures, total=total_files, desc="Processing files"):
                file_path, summary = future.result()
                self.summaries[file_path] = summary
        self.export_summary()
        logging.info(f"Processing completed in {(datetime.now() - start_time).total_seconds():.2f} seconds")

    def index_keywords(self, text: str, file_path: str) -> None:
        """Index key phrases for search."""
        key_phrases = self.text_analyzer.extract_key_phrases(text)
        for phrase in key_phrases:
            self.index[phrase].append(file_path)

    def export_summary(self) -> None:
        """Export summaries and keyword index to files."""
        if not self.summaries:
            logging.warning("No summaries to export.")
            return
        with open(self.output_dir / "document_summary.txt", 'w', encoding='utf-8') as f:
            f.write(self.format_summary_text())
        with open(self.output_dir / "keyword_index.txt", 'w', encoding='utf-8') as f:
            f.write("KEYWORD INDEX\n" + "=" * 70 + "\n\n")
            for keyword, files in sorted(self.index.items()):
                f.write(f"Keyword: {keyword}\nFiles:\n" + "\n".join(f"  - {file}" for file in files) + "\n\n")
        logging.info(f"Exported results to {self.output_dir}")

    def search(self, query: str) -> List[str]:
        """Perform semantic search across processed documents."""
        query_embedding = self.embedder.encode(query)
        results = []
        for file_path, embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            if similarity > 0.7:
                results.append((file_path, similarity))
        return [r[0] for r in sorted(results, key=lambda x: x[1], reverse=True)]

    def ask_question(self, question: str, context: str = None) -> str:
        """Answer a question based on document content."""
        if context is None:
            context = "\n".join([s['content_summary'] for s in self.summaries.values()])
        return self.text_analyzer.answer_question(question, context)

def main():
    """Example usage of the DocumentProcessor."""
    processor = DocumentProcessor()
    processor.process_documents()
    search_results = processor.search("artificial intelligence applications")
    print("Search Results:", search_results)
    answer = processor.ask_question("What are the main topics discussed?")
    print("Question Answer:", answer)
    test_prompt = "Explain the significance of artificial intelligence."
    generated_text = processor.text_analyzer.generate_text(test_prompt)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()