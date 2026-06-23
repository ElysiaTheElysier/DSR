import anthropic

client = anthropic.Anthropic(
    api_key="AGOP-BAE4-B148-1E86"  
)

response = client.messages.create(
    model="claude-sonnet-3-5",
    max_tokens=10000,
    messages=[{
        "role": "user",
        "content": "Describe the key semantic features of this text for ML classification..."
    }]
)
