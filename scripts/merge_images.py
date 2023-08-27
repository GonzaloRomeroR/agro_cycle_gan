import os
from PIL import Image
from pathlib import Path


def merge_images(
    data_folder_A: str, data_folder_B: str, output_folder: str, img_resize=(512, 512)
):
    """
    Merge to images in the same output image. This can be used to have a quick
    comparation between images
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    dirs = () if not os.path.isdir(data_folder_A) else os.listdir(data_folder_A)

    for item in dirs:

        if (
            not item.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
            )
            or not os.path.isfile(f"{data_folder_A}/{item}")
            or not os.path.isfile(f"{data_folder_B}/{item}")
        ):
            continue

        imageA = Image.open(f"{data_folder_A}/{item}")
        imageB = Image.open(f"{data_folder_B}/{item}")

        imageA = imageA.resize(img_resize)
        imageB = imageB.resize(img_resize)

        new_image = Image.new("RGB", (2 * img_resize[0], img_resize[1]))
        new_image.paste(imageA, (0, 0))
        new_image.paste(imageB, (img_resize[0], 0))

        new_image.save(f"{output_folder}/merged_{item}", "JPEG")


if __name__ == "__main__":
    data_folder_A = ""
    data_folder_B = ""
    output_folder = ""
    merge_images(data_folder_A, data_folder_B, output_folder)
