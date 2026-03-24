import os
from huggingface_hub import InferenceClient

"""
Follow the instructions here to get a Hugging Face token:
https://huggingface.co/docs/hub/en/security-tokens
"""
HF_TOKEN = os.getenv("HF_TOKEN") 

if not HF_TOKEN:
    raise ValueError(
        "Missing Hugging Face token. Set HF_TOKEN in your environment variables."
    )

client = InferenceClient(
    api_key=HF_TOKEN,
)

"""
The following is an example of how to use the Kimi-K2.5 model to generate text.
This is an easy task, so we disabled thinking to get a faster response.
"""
def text_example():
    completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2.5:novita",
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?",
            }
        ],
        extra_body={'thinking': {'type': 'disabled'}}
    )
    print(completion.choices[0].message.content)

"""
The following is an example of how to use the Kimi-K2.5 model to generate text from an image.
This is a more complex task, so we keep thinking enabled (default) to get a more accurate response.
"""
def image_example():
    completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2.5:novita",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in one sentence."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                        }
                    }
                ]
            }
        ],
    )
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    print("Text example:")
    text_example()
    print("\nImage example:")
    image_example()