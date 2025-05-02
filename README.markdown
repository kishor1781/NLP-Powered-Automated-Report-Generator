# Document Processing and Text Analysis

This project provides a Python-based system for processing various document types (PDF, DOCX, PPTX, XLSX, images, and text) and performing advanced text analysis using large language models (LLMs). It includes OCR capabilities, semantic search, text summarization, question answering, and text generation, leveraging Hugging Face models and other NLP tools.

## Features
- **Document Processing**: Extracts text and images from multiple file formats (PDF, DOCX, PPTX, XLSX, JPG, PNG, TXT).
- **OCR Support**: Uses Tesseract and EasyOCR for text extraction from images, with multilingual support.
- **Text Analysis**:
  - Summarization using BART.
  - Question answering with RoBERTa.
  - Zero-shot classification with BART-MNLI.
  - Text generation via Mixtral (API or local model).
  - Key phrase and entity extraction using spaCy.
- **Semantic Search**: Uses Sentence Transformers for embedding-based search across processed documents.
- **GPU Acceleration**: Supports CUDA for faster processing when available.
- **Parallel Processing**: Utilizes ThreadPoolExecutor for efficient document handling.

## Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for accelerated processing)
- Tesseract OCR installed (see [Tesseract Installation](https://github.com/tesseract-ocr/tesseract))
- A Hugging Face API key for Mixtral text generation (optional)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Tesseract OCR and ensure it's accessible in your system PATH.
4. (Optional) Set up a Hugging Face API key in `config.yaml` for text generation.

## Configuration
The project uses a `config.yaml` file to manage settings. Key configurations include:
- `input_dir`: Directory containing input documents.
- `output_dir`: Directory for output summaries and logs.
- `use_gpu`: Enable/disable GPU usage.
- `max_workers`: Number of parallel workers for processing.
- `batch_size`: Batch size for embedding generation.
- `ocr_languages`: List of languages for OCR (e.g., `['en']` for English).
- `llm_models`: Specifies models for summarization, question answering, text generation, and classification.
- `use_hf_api`: Enable Hugging Face Inference API for text generation.
- `hf_api_key`: Your Hugging Face API key (if using API).

Example `config.yaml`:
```yaml
input_dir: 'C:\virtualbox\input_dir'
output_dir: 'C:\virtualbox\output_dir'
use_gpu: true
max_workers: 4
batch_size: 32
ocr_languages:
  - en
llm_models:
  summarization: "facebook/bart-large-cnn"
  question_answering: "deepset/roberta-base-squad2"
  text_generation: "mistralai/Mixtral-8x7B-Instruct-v0.1"
  classification: "facebook/bart-large-mnli"
use_hf_api: true
hf_api_key: "your-hf-api-key"
```

## Usage
Run the main script to process documents:
```bash
python document_processor.py
```
This will:
1. Scan the input directory for supported file types.
2. Process each file to extract text and images.
3. Generate summaries, key points, topics, and embeddings.
4. Export results to `output_dir` as `document_summary.txt` and `keyword_index.txt`.

### Example Commands
- Perform a semantic search:
  ```python
  processor = DocumentProcessor()
  results = processor.search("artificial intelligence applications")
  print(results)
  ```
- Answer a question based on processed documents:
  ```python
  answer = processor.ask_question("What are the main topics discussed?")
  print(answer)
  ```
- Generate text:
  ```python
  text = processor.text_analyzer.generate_text("Explain the significance of artificial intelligence.")
  print(text)
  ```

## Project Structure
```
your-repo/
├── config.yaml                # Configuration file
├── document_processor.py      # Main document processing script
├── text_analyzer_advanced.py  # Text analysis with LLMs
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── LICENSE                   # License file
└── .gitignore                # Git ignore file
```

## Outputs
- **document_summary.txt**: Summaries of processed files, including metadata, content summaries, key points, topics, and image captions.
- **keyword_index.txt**: Indexed key phrases mapped to files for quick lookup.
- **Logs**: Processing logs saved in `output_dir` with timestamps.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, please open an issue on GitHub or contact [raghukishor1781@gmail.com].
