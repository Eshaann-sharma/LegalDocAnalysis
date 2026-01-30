from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="google/flan-t5-base"
)

def summarize_text(text):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = []

    for chunk in chunks[:3]:   # limit for demo
        out = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(out[0]['summary_text'])

    return " ".join(summaries)
