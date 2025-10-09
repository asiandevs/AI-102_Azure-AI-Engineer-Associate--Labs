from dotenv import load_dotenv
import os
import sys
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# Import namespaces
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel, FaceAttributeTypeDetection01
from azure.core.credentials import AzureKeyCredential

def main():

    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        # Get Configuration Settings
        load_dotenv()
        cog_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        cog_key = os.getenv('AI_SERVICE_KEY')

        # Get image
        image_file = 'images/face1.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Authenticate Face client
        face_client = FaceClient(
            endpoint=cog_endpoint,
            credential=AzureKeyCredential(cog_key)
        )

        # Specify facial features to be retrieved
        features = [
            FaceAttributeTypeDetection01.HEAD_POSE,
            FaceAttributeTypeDetection01.OCCLUSION,
            FaceAttributeTypeDetection01.ACCESSORIES
        ]

        # Get faces
        with open(image_file, mode="rb") as image_data:
            detected_faces = face_client.detect(
                image_content=image_data.read(),
                detection_model=FaceDetectionModel.DETECTION01,
                recognition_model=FaceRecognitionModel.RECOGNITION01,
                return_face_id=False,
                return_face_attributes=features
            )

            face_count = 0
            if len(detected_faces) > 0:
                print(len(detected_faces), 'faces detected')
                for face in detected_faces:
                    face_count += 1
                    print('\nFace number {}'.format(face_count))
                    print(' - Head Pose (Yaw): {}'.format(face.face_attributes.head_pose.yaw))
                    print(' - Head Pose (Pitch): {}'.format(face.face_attributes.head_pose.pitch))
                    print(' - Head Pose (Roll): {}'.format(face.face_attributes.head_pose.roll))
                    print(' - Forehead occluded?: {}'.format(face.face_attributes.occlusion["foreheadOccluded"]))
                    print(' - Eye occluded?: {}'.format(face.face_attributes.occlusion["eyeOccluded"]))
                    print(' - Mouth occluded?: {}'.format(face.face_attributes.occlusion["mouthOccluded"]))
                    print(' - ACCESSORIES:')
                    for accessory in face.face_attributes.accessories:
                        print('   - {}'.format(accessory.type))
                
                # Annotate faces in the image
                annotate_faces(image_file, detected_faces)

    except Exception as ex:
        print(ex)

def annotate_faces(image_file, detected_faces):
    print('\nAnnotating faces in image...')

    # Prepare image for drawing
    fig = plt.figure(figsize=(8, 6))
    plt.axis('off')
    image = Image.open(image_file)
    draw = ImageDraw.Draw(image)
    color = 'lightgreen'

    # Annotate each face in the image
    face_count = 0
    for face in detected_faces:
        face_count += 1
        r = face.face_rectangle
        bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
        draw.rectangle(bounding_box, outline=color, width=5)
        annotation = 'Face number {}'.format(face_count)
        plt.annotate(annotation, (r.left, r.top), backgroundcolor=color)

    # Save annotated image
    plt.imshow(image)
    outputfile = 'detected_faces.jpg'
    fig.savefig(outputfile)
    print(f'  Results saved in {outputfile}\n')

if __name__ == "__main__":
    main()
