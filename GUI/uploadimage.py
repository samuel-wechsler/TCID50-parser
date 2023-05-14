from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload

from concurrent.futures import ThreadPoolExecutor

import io
from PIL import Image
import os
import tempfile

from datetime import datetime


class Uploader:
    def __init__(self):
        # Set the path to your service account credentials JSON file
        self.creds_path = 'credentials.json'

        # Set the ID of the folder where you want to store the uploaded images
        self.folder_id = '1p9i5Vwkdj-tvDqcafcUi56e-u4YAb1iK'

        # Authenticate your application with the Google Drive API
        self.creds = None
        if os.path.exists(self.creds_path):
            self.creds = service_account.Credentials.from_service_account_file(
                self.creds_path,
                scopes=['https://www.googleapis.com/auth/drive']
            )

        # Create a service object for interacting with the Google Drive API
        self.drive_service = build('drive', 'v3', credentials=self.creds)

    def upload_image(self, file_path):
        # Open the image file
        with Image.open(file_path) as img:

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
                                 'parents': [self.folder_id]}
                media = MediaFileUpload(f.name, mimetype='image/jpeg')
                try:
                    file = self.drive_service.files().create(
                        body=file_metadata, media_body=media, fields='id').execute()
                except HttpError as error:
                    raise error
            return file
