import base64
from mimetypes import guess_type
from openai import AzureOpenAI
from typing import List, Dict, Any

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


# Function to construct input message for calling OpenAI models
def construct_input_message(user_text, image_paths, image_detail="auto", system_text=None):
    # TODO: set the input class
    # TODO: check input class and file existence
    
    content = [
        {"type": "text", "text": user_text}
    ]
    
    for image_path in image_paths:
        encoded_img = local_image_to_data_url(image_path)
        if encoded_img:
            content.append({
                "type": "image_url",
                "image_url": {"url": encoded_img},
                "detail": image_detail
            })

    input_message = []
    if system_text is not None:
        input_message.append({"role": "system", "content": system_text})
    
    input_message.append({"role": "user", "content": content})
    
    return input_message


def azure_openai_model(model_name: str):
    import os
    OPENAIKEY = os.environ.get("AZURE_OPENAI_API_KEY", "YOUR_AZURE_API_KEY")
    URL = {
        "gpt-4o": os.environ.get("AZURE_OPENAI_ENDPOINT", "https://YOUR_ENDPOINT.openai.azure.com/") + f"openai/deployments/{model_name}/chat/completions?api-version=2025-01-01-preview",
    }
    API_VERSION = "2025-01-01-preview"
    
    return  AzureOpenAI(
        base_url=URL[model_name],
        api_version=API_VERSION,
        api_key=OPENAIKEY
    )

def azure_openai_query_structured(model: AzureOpenAI, model_name: str, input_message: List[Dict[str, Any]], response_format: Any):
    response = model.chat.completions.parse(
        model=model_name,
        messages=input_message,
        response_format=response_format
    )
    response = response.choices[0].message.parsed
    return response