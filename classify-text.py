##git clone https://github.com/microsoftlearning/mslearn-ai-language
#Update the configuration values to include the endpoint and a key from the Azure Language resource you created (available on the Keys and Endpoint page for your Azure AI Language resource in the Azure portal).
#The file should already contain the project and deployment names for your text classification model.

from dotenv import load_dotenv
import os

# Import Azure namespaces
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient


def main():
    try:
        # Load environment variables
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')
        project_name = os.getenv('PROJECT')
        deployment_name = os.getenv('DEPLOYMENT')

        # Create client using endpoint and key
        credential = AzureKeyCredential(ai_key)
        ai_client = TextAnalyticsClient(endpoint=ai_endpoint, credential=credential)

        # Collect documents from "articles" folder
        batched_documents = []
        articles_folder = 'articles'
        files = os.listdir(articles_folder)

        for file_name in files:
            text = open(os.path.join(articles_folder, file_name), encoding='utf8').read()
            batched_documents.append(text)

        # Run classification
        operation = ai_client.begin_single_label_classify(
            documents=batched_documents,
            project_name=project_name,
            deployment_name=deployment_name
        )

        document_results = operation.result()

        # Display results
        for doc, classification_result in zip(files, document_results):
            if not classification_result.is_error:
                classification = classification_result.classifications[0]
                print(
                    f"{doc} was classified as '{classification.category}' "
                    f"with confidence score {classification.confidence_score:.2f}."
                )
            else:
                print(
                    f"{doc} has an error with code '{classification_result.error.code}' "
                    f"and message '{classification_result.error.message}'"
                )

    except Exception as ex:
        print("Error:", ex)


if __name__ == "__main__":
    main()
