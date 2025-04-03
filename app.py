import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

def load_image(image_file):
    """Load an image from an uploaded file."""
    image = Image.open(image_file)
    return image

def process_image(image, method, params):
    """Process the image using the selected method and parameters."""
    # Convert PIL image to a NumPy array
    image_np = np.array(image)
    
    if method == 'Grayscale':
        processed = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    elif method == 'Blur':
        ksize = params.get('kernel_size', 15)
        if ksize % 2 == 0:
            ksize += 1
        processed = cv2.GaussianBlur(image_np, (ksize, ksize), 0)
    elif method == 'Canny Edge Detection':
        threshold1 = params.get('threshold1', 100)
        threshold2 = params.get('threshold2', 200)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        processed = cv2.Canny(gray, threshold1, threshold2)
    elif method == 'Sketch':
        blur_kernel = params.get('sketch_blur_kernel', 21)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        inv_gray = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv_gray, (blur_kernel, blur_kernel), sigmaX=0, sigmaY=0)
        processed = cv2.divide(gray, 255 - blur, scale=256)
    elif method == 'Histogram':
        plt.figure()
        if len(image_np.shape) == 2:
            hist = cv2.calcHist([image_np], [0], None, [256], [0, 256])
            plt.plot(hist, color='black')
            plt.title("Grayscale Histogram")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.xlim([0, 256])
        else:
            for i, color in enumerate(('r', 'g', 'b')):
                hist = cv2.calcHist([image_np], [i], None, [256], [0, 256])
                plt.plot(hist, color=color)
            plt.title("Color Histogram")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.xlim([0, 256])
        processed = plt.gcf()
        plt.close(processed)
    elif method == 'Crop':
        # Get crop coordinates from params
        left = params.get('crop_left', 0)
        top = params.get('crop_top', 0)
        right = params.get('crop_right', image_np.shape[1])
        bottom = params.get('crop_bottom', image_np.shape[0])
        # Ensure valid coordinates
        if left < right and top < bottom:
            processed = image_np[top:bottom, left:right]
        else:
            processed = image_np
    else:
        processed = image_np
        
    return processed

def convert_image_to_bytes(img):
    """Convert processed image (as a numpy array) to PNG format bytes."""
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = img
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def main():
    st.title("Image Processing App")
    st.write("Apply image processing techniques to your images!")
    
    image_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if image_file is not None:
        image = load_image(image_file)
        image_np = np.array(image)
        height, width = image_np.shape[:2]
        
        st.sidebar.header("Processing Options")
        method = st.sidebar.selectbox("Select Processing Method", 
                                      ["Original", "Grayscale", "Blur", "Canny Edge Detection", "Sketch", "Histogram", "Crop"])
        
        params = {}
        if method == 'Blur':
            params['kernel_size'] = st.sidebar.slider("Kernel Size", min_value=1, max_value=31, value=15, step=2)
        elif method == 'Canny Edge Detection':
            params['threshold1'] = st.sidebar.slider("Threshold 1", min_value=0, max_value=255, value=100)
            params['threshold2'] = st.sidebar.slider("Threshold 2", min_value=0, max_value=255, value=200)
        elif method == 'Sketch':
            params['sketch_blur_kernel'] = st.sidebar.slider("Sketch Blur Kernel", min_value=3, max_value=51, value=21, step=2)
        elif method == 'Crop':
            st.sidebar.write("Set crop boundaries:")
            params['crop_left'] = st.sidebar.slider("Left", min_value=0, max_value=width-1, value=0)
            params['crop_right'] = st.sidebar.slider("Right", min_value=1, max_value=width, value=width)
            params['crop_top'] = st.sidebar.slider("Top", min_value=0, max_value=height-1, value=0)
            params['crop_bottom'] = st.sidebar.slider("Bottom", min_value=1, max_value=height, value=height)
        
        if st.sidebar.button("Process"):
            processed_image = process_image(image, method, params)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                if method == "Histogram":
                    st.pyplot(processed_image)
                else:
                    st.image(processed_image, caption=f"{method} Image", use_container_width=True)
            
            if method != "Histogram":
                img_bytes = convert_image_to_bytes(processed_image)
                st.download_button(
                    label="Download Processed Image",
                    data=img_bytes,
                    file_name=f"processed_image.png",
                    mime="image/png"
                )

if __name__ == '__main__':
    main()
