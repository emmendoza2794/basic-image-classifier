from PIL import Image
import os
from huggingface_hub import hf_hub_download


class Utils:

    def clean_corrupt_images(self, root_dir: str):
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(subdir, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except (IOError, SyntaxError):
                    print(f'corrupt image: {file_path}')
                    os.remove(file_path)

    def download_models(self):

        repo_id = "emmendoza2794/basic-image-classifier"

        download_files = [
            "classes_model_cat_dog.json",
            "model_cat_dog.pth",
            "classes_model_flowers.json",
            "model_flowers.pth"
        ]

        for file in download_files:
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir="models",
                local_dir_use_symlinks=False,
            )
