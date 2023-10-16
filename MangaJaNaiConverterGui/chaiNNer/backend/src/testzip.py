import zipfile
from PIL import Image
import io

def _read_pil(im) -> np.ndarray | None:
    if get_ext(path) not in get_pil_formats():
        # not supported
        return None

    im = Image.open(path)
    img = np.array(im)
    _, _, c = get_h_w_c(img)
    if c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif c == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    return img

# Define the paths for the input and output zip files
input_zip_path = r"D:\file\同人誌\(C102) [どらやきや (井上たくや)] ヒカリちゃんのもっとえっち本 (ゼノブレイド2).zip"
output_zip_path = r"D:\file\同人誌\(C102) [どらやきや (井上たくや)] ヒカリちゃんのもっとえっち本 (ゼノブレイド2)-test.zip"

# Open the input zip file in read mode
with zipfile.ZipFile(input_zip_path, 'r') as input_zip:
    # Create a new zip file in write mode for the resized images
    with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
        # Iterate through the files in the input zip
        for file_name in input_zip.namelist():
            # Open the file inside the input zip
            with input_zip.open(file_name) as file_in_zip:
                # Read the image data
                image_data = file_in_zip.read()

                # Open the image using Pillow (PIL)
                with Image.open(io.BytesIO(image_data)) as img:
                    # Resize the image to 50% scale
                    width, height = img.size
                    new_width = width // 2
                    new_height = height // 2
                    resized_img = img.resize((new_width, new_height))

                    # Convert the resized image back to bytes
                    output_buffer = io.BytesIO()
                    resized_img.save(output_buffer, format="JPEG")
                    resized_image_data = output_buffer.getvalue()

                    # Add the resized image to the output zip
                    output_zip.writestr(file_name, resized_image_data)

print(f'Resized images saved to {output_zip_path}')
