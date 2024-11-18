import openai

# Set your OpenAI API key
openai.api_key = "your-api-key-here"

def generate_report(summary):
    prompt = f"Based on the following summary, generate a detailed technical report:\n\n{summary}\n\nTechnical Report:"
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    report = response.choices[0].text.strip()
    return report