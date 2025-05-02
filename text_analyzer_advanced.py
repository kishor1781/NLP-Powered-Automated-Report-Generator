import logging
from typing import List
import re
import spacy
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
from accelerate import infer_auto_device_map, init_empty_weights
import os
import requests

class TextAnalyzer:
    def __init__(self, config: dict):
        """Initialize with multiple LLMs for various tasks."""
        self.config = config
        self.nlp = spacy.load('en_core_web_sm')  # Multilingual spaCy model
        self.device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")
        logging.info(f"Device set to use {self.device}")

        # Load smaller models directly
        self.summarizer = pipeline(
            "summarization", 
            model=config["llm_models"]["summarization"], 
            device=0 if config["use_gpu"] else -1
        )
        self.qa_model = pipeline(
            "question-answering", 
            model=config["llm_models"]["question_answering"], 
            device=0 if config["use_gpu"] else -1
        )
        self.classifier = pipeline(
            "zero-shot-classification", 
            model=config["llm_models"]["classification"], 
            device=0 if config["use_gpu"] else -1
        )

        # Check for API usage
        self.use_api = self.config.get("use_hf_api", False)
        self.hf_api_key = self.config.get("hf_api_key", None)
        if self.use_api and self.hf_api_key:
            self.hf_api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
            self.headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            logging.info("Configured to use Hugging Face Inference API for text generation.")
        else:
            # Load large text generation model with offloading
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            offload_dir = os.path.join(os.getcwd(), "offload")
            os.makedirs(offload_dir, exist_ok=True)

            try:
                model_config = AutoConfig.from_pretrained(config["llm_models"]["text_generation"])
                with init_empty_weights():
                    model = AutoModelForCausalLM.from_config(model_config)

                max_memory = {0: "10GiB", "cpu": "20GiB"}
                device_map = infer_auto_device_map(model, max_memory=max_memory)

                self.gen_tokenizer = AutoTokenizer.from_pretrained(config["llm_models"]["text_generation"])
                self.gen_model = AutoModelForCausalLM.from_pretrained(
                    config["llm_models"]["text_generation"],
                    quantization_config=quantization_config,
                    device_map=device_map,
                    offload_folder=offload_dir
                )
                logging.info(f"Loaded text generation model with device_map: {device_map}")
            except Exception as e:
                logging.error(f"Failed to load Mixtral model: {str(e)}. Using fallback model.")
                self.gen_tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.gen_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        text = re.sub(r'\s+', ' ', text.strip())
        doc = self.nlp(text)
        return ' '.join(token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 2)

    def extract_key_phrases(self, text: str, max_phrases: int = 30) -> List[str]:
        """Extract key phrases using spaCy."""
        doc = self.nlp(text)
        key_phrases = set()
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:
                key_phrases.add(chunk.text.lower())
        for ent in doc.ents:
            key_phrases.add(ent.text.lower())
        return sorted(list(key_phrases), key=lambda x: (-len(x.split()), x))[:max_phrases]

    def summarize(self, text: str, max_length: int = 150) -> str:
        """Summarize text using BART."""
        try:
            if len(text.split()) < 50:
                return text
            return self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
        except Exception as e:
            logging.error(f"Error in summarization: {str(e)}")
            return text[:500] + '...'

    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points based on entities and numerical data."""
        doc = self.nlp(text)
        key_points = []
        for sent in doc.sents:
            score = len([ent for ent in sent.ents]) * 2 + len([token for token in sent if token.like_num]) * 1.5
            if score >= 5:
                key_points.append(sent.text.strip())
        return key_points[:10]

    def extract_main_topics(self, text: str, num_topics: int = 10) -> List[str]:
        """Extract main topics using zero-shot classification."""
        doc = self.nlp(text)
        candidates = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4]
        candidates = list(set(candidates))[:20]
        if not candidates:
            return []
        result = self.classifier(text, candidate_labels=candidates, multi_label=True)
        return [label for label, score in zip(result['labels'], result['scores']) if score > 0.6][:num_topics]

    def classify_document(self, text: str) -> str:
        """Classify document into predefined categories."""
        categories = ["technical", "legal", "personal", "business", "academic", "other"]
        result = self.classifier(text, candidate_labels=categories)
        return result['labels'][0]

    def answer_question(self, question: str, context: str) -> str:
        """Answer questions using RoBERTa QA model."""
        try:
            return self.qa_model(question=question, context=context)["answer"]
        except Exception as e:
            logging.error(f"Error in question answering: {str(e)}")
            return "Unable to answer."

    def generate_text(self, prompt: str, max_length: int = 200) -> str:
        """Generate text using Mixtral via API or fallback to local model."""
        if self.use_api and self.hf_api_key:
            try:
                payload = {
                    "inputs": f"[INST] {prompt} [/INST]",
                    "parameters": {"max_new_tokens": max_length, "do_sample": True}
                }
                response = requests.post(self.hf_api_url, headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json()[0]["generated_text"]
            except Exception as e:
                logging.error(f"API call failed: {str(e)}")
                return "API generation failed."
        else:
            try:
                inputs = self.gen_tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.gen_model.generate(**inputs, max_length=max_length, num_return_sequences=1, do_sample=True)
                return self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                logging.error(f"Error in text generation: {str(e)}")
                return "Generation failed."