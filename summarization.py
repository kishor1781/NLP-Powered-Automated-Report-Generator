from transformers import BertTokenizer, BertForSequenceClassification, pipeline

def summarize_text(text, max_length=150):
    # Load pre-trained BERT model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    
    # Create a summarization pipeline
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    
    # Generate summary
    summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
    
    return summary[0]['summary_text']