from google.oauth2 import service_account
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload

import io
from PIL import Image
import os
import tempfile

from datetime import datetime


class Uploader:
    def __init__(self):
        # Set the path to your service account credentials JSON file
        self.creds_path = 'credentials/credentials.json'

        # Set the ID of the folder where you want to store the uploaded images
        self.folder_id = '1p9i5Vwkdj-tvDqcafcUi56e-u4YAb1iK'

        # Authenticate your application with the Google Drive API
        self.creds = None
        self.authenticate()

        # Create a service object for interacting with the Google Drive API
        self.drive_service = build('drive', 'v3', credentials=self.creds)

    def authenticate(self):
        # Set the scopes required for Google Drive access
        scopes = ['https://www.googleapis.com/auth/drive']

        # Check if token file exists
        flow = InstalledAppFlow.from_client_secrets_file(
            credentials_file, scopes)
        credentials = flow.run_local_server(port=0)

    def create_image_dir(self, dirname):
        """
        creates a folder in the google drive
        """
        file_metadata = {
            'name': dirname,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [self.folder_id]
        }
        file = self.drive_service.files().create(
            body=file_metadata, fields='id').execute()

        self.child_folder_id = file.get('id')
        return file

    def upload_image(self, file_path):
        # Open the image file
        with Image.open(file_path) as img:
            # convert image to RGB format
            img = img.convert('RGB')

            # Resize the image
            max_size = 500
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.LANCZOS)

            # Save the resized image to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg') as f:
                img.save(f, format='JPEG')
                f.seek(0)

                # Upload the resized image file to the Google Drive folder
                file_metadata = {'name': os.path.basename(file_path),
                                 'parents': [self.child_folder_id]}
                media = MediaFileUpload(f.name, mimetype='image/jpeg')
                try:
                    file = self.drive_service.files().create(
                        body=file_metadata, media_body=media, fields='id').execute()
                except HttpError as error:
                    raise error
            return file

    def upload_model_arch(self, model_json_path):
        """
        uploads a json file which describes the model architecture
        """
        # TODO : upload training parameters
        with open(model_json_path, "r") as json_file:
            file_contents = json_file.read()
            file_metadata = {'name': os.path.basename(model_json_path),
                             'parents': [self.folder_id]}

            media = MediaIoBaseUpload(io.BytesIO(file_contents.encode()),
                                      mimetype='application/json')
            try:
                file = self.drive_service.files().create(
                    body=file_metadata, media_body=media, fields='id').execute()
                print(4)
            except HttpError as error:
                raise error
            return file

    def upload_class(self, class_path):
        """
        uploads a txt file which describes the classifications of image data
        """
        with open(class_path, "r") as txt_file:
            file_contents = txt_file.read()
            file_metadata = {'name': os.path.basename(class_path),
                             'parents': [self.folder_id]}

            media = MediaIoBaseUpload(io.BytesIO(file_contents.encode()),
                                      mimetype='text/plain')
            try:
                file = self.drive_service.files().create(
                    body=file_metadata, media_body=media, fields='id').execute()
            except HttpError as error:
                raise error
            return file
