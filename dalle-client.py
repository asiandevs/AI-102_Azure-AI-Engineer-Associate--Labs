import os
import json
import requests
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        # Get configuration settings
        load_dotenv()
        endpoint = os.getenv("ENDPOINT")
        model_deployment = os.getenv("MODEL_DEPLOYMENT")
        api_version = os.getenv("API_VERSION")

        # Initialize the client
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(
                exclude_environment_credential=True,
                exclude_managed_identity_credential=True
            ),
            "https://cognitiveservices.azure.com/.default"
        )

        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider
        )

        img_no = 0
        # Loop until the user types 'quit'
        while True:
            input_text = input("Enter the prompt (or type 'quit' to exit): ")
            if input_text.lower() == "quit":
                break
            if len(input_text) == 0:
                print("Please enter a prompt.")
                continue

            # Generate an image
            result = client.images.generate(
                model=model_deployment,
                prompt=input_text,
                n=1
            )

            json_response = json.loads(result.model_dump_json())
            image_url = json_response["data"][0]["url"]

            # Save the image
            img_no += 1
            file_name = f"image_{img_no}.png"
            save_image(image_url, file_name)

    except Exception as ex:
        print(ex)

def save_image(image_url, file_name):
    # Set the directory for the stored image
    image_dir = os.path.join(os.getcwd(), 'images')

    # If the directory doesn't exist, create it
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    # Initialize the image path
    image_path = os.path.join(image_dir, file_name)

    # Retrieve and save the generated image
    generated_image = requests.get(image_url).content
    with open(image_path, "wb") as image_file:
        image_file.write(generated_image)
    print(f"Image saved as {image_path}")

if __name__ == '__main__':
    main()
