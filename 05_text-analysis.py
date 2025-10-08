##https://github.com/MicrosoftLearning/mslearn-ai-language/tree/main/Labfiles

from dotenv import load_dotenv
import os

# Import namespaces
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

def main():
    try:
        # -----------------------------
        # Load configuration settings
        # -----------------------------
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # -----------------------------
        # Create client using endpoint and key
        # -----------------------------
        credential = AzureKeyCredential(ai_key)
        ai_client = TextAnalyticsClient(endpoint=ai_endpoint, credential=credential)

        # -----------------------------
        # Analyze each text file in the reviews folder
        # -----------------------------
        reviews_folder = 'reviews'
        for file_name in os.listdir(reviews_folder):
            file_path = os.path.join(reviews_folder, file_name)
            with open(file_path, encoding='utf8') as f:
                text = f.read()

            print('\n-------------\n' + file_name)
            print('\nText:\n' + text)

            # -----------------------------
            # Get language
            # -----------------------------
            detected_language = ai_client.detect_language(documents=[text])[0]
            print('\nLanguage: {}'.format(detected_language.primary_language.name))

            # -----------------------------
            # Get sentiment
            # -----------------------------
            sentiment_analysis = ai_client.analyze_sentiment(documents=[text])[0]
            print("Sentiment: {}".format(sentiment_analysis.sentiment))

            # -----------------------------
            # Get key phrases
            # -----------------------------
            key_phrases = ai_client.extract_key_phrases(documents=[text])[0].key_phrases
            if key_phrases:
                print("\nKey Phrases:")
                for phrase in key_phrases:
                    print('\t{}'.format(phrase))

            # -----------------------------
            # Get entities
            # -----------------------------
            entities = ai_client.recognize_entities(documents=[text])[0].entities
            if entities:
                print("\nEntities:")
                for entity in entities:
                    print('\t{} ({})'.format(entity.text, entity.category))

            # -----------------------------
            # Get linked entities
            # -----------------------------
            #linked_entities = ai_client.recognize_linked_entities(documents=[text])[0].entities
            #if linked_entities:
              #  print("\nLinked Entities:")
               # for linked_entity in linked_entities:
                #    print('\t{} ({})'.format(linked_entity.name, linked_entity.url))

    except Exception as ex:
        print("Error:", ex)


if __name__ == "__main__":
    main()
