import argparse
from data_preprocessing import preprocess_data
from summarization import summarize_text
from report_generation import generate_report
from evaluation import evaluate_accuracy

def main(input_file, output_file):
    # Preprocess the data
    preprocessed_data = preprocess_data(input_file)
    
    # Summarize the text using BERT
    summary = summarize_text(preprocessed_data)
    
    # Generate the report using GPT
    report = generate_report(summary)
    
    # Write the report to the output file
    with open(output_file, 'w') as f:
        f.write(report)
    
    # Evaluate the accuracy
    accuracy = evaluate_accuracy(preprocessed_data, report)
    print(f"Report generation accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP-Powered Automated Report Generator")
    parser.add_argument("input_file", help="Path to the input file containing unstructured data")
    parser.add_argument("output_file", help="Path to save the generated report")
    args = parser.parse_args()
    
    main(args.input_file, args.output_file)