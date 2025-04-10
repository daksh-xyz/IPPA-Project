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
    elif method == 'Cartoonify':
        K = params.get('k', 8)
        processed = cartoonify(image_np, K)
    elif method == 'Sketch':
        blur_kernel = params.get('sketch_blur_kernel', 21)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (blur_kernel, blur_kernel), 8)
        inverted_blur = 255 - blurred
        sketch = cv2.divide(gray, inverted_blur, scale=256.0)
        edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
        processed = cv2.bitwise_and(sketch, 255 - edges)
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

def cartoonify(img_np, k):
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # Apply some Gaussian blur on the image
    img_gb = cv2.GaussianBlur(img, (7, 7) ,0)
    # Apply some Median blur on the image
    img_mb = cv2.medianBlur(img_gb, 5)
    # Apply a bilateral filer on the image
    img_bf = cv2.bilateralFilter(img_mb, 5, 80, 80)
    # Use the laplace filter to detect edges
    img_lp_im = cv2.Laplacian(img, cv2.CV_8U, ksize=5)
    img_lp_gb = cv2.Laplacian(img_gb, cv2.CV_8U, ksize=5)
    img_lp_mb = cv2.Laplacian(img_mb, cv2.CV_8U, ksize=5)
    img_lp_al = cv2.Laplacian(img_bf, cv2.CV_8U, ksize=5)
    # Convert the image to greyscale (1D)
    img_lp_im_grey = cv2.cvtColor(img_lp_im, cv2.COLOR_BGR2GRAY)
    img_lp_gb_grey = cv2.cvtColor(img_lp_gb, cv2.COLOR_BGR2GRAY)
    img_lp_mb_grey = cv2.cvtColor(img_lp_mb, cv2.COLOR_BGR2GRAY)
    img_lp_al_grey = cv2.cvtColor(img_lp_al, cv2.COLOR_BGR2GRAY)
    _, EdgeImage = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    blur_im = cv2.GaussianBlur(img_lp_im_grey, (5, 5), 0)
    blur_gb = cv2.GaussianBlur(img_lp_gb_grey, (5, 5), 0)
    blur_mb = cv2.GaussianBlur(img_lp_mb_grey, (5, 5), 0)
    blur_al = cv2.GaussianBlur(img_lp_al_grey, (5, 5), 0)# Apply a threshold (Otsu)
    _, tresh_im = cv2.threshold(blur_im, 245, 255,cv2.THRESH_BINARY +  cv2.THRESH_OTSU)
    _, tresh_gb = cv2.threshold(blur_gb, 245, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, tresh_mb = cv2.threshold(blur_mb, 245, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, tresh_al = cv2.threshold(blur_al, 245, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_original = cv2.subtract(255, tresh_im)
    inverted_GaussianBlur = cv2.subtract(255, tresh_gb)
    inverted_MedianBlur = cv2.subtract(255, tresh_mb)
    inverted_Bilateral = cv2.subtract(255, tresh_al)
    img_reshaped = img.reshape((-1,3))
    # convert to np.float32
    img_reshaped = np.float32(img_reshaped)
    # Set the Kmeans criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set the amount of K (colors)
    K = k
    # Apply Kmeans
    _, label, center = cv2.kmeans(img_reshaped, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)# Covert it back to np.int8
    center = np.uint8(center)
    res = center[label.flatten()]
    # Reshape it back to an image
    img_Kmeans = res.reshape((img.shape))
    # Reduce the colors of the original image
    div = 64
    img_bins = img // div * div + div // 2
    # Convert the mask image back to color 
    inverted_Bilateral = cv2.cvtColor(inverted_Bilateral, cv2.COLOR_GRAY2RGB)
    # Combine the edge image and the binned image
    cartoon_Bilateral = cv2.bitwise_and(inverted_Bilateral, img_bins)# Save the image
    cartoon_rgb = cv2.cvtColor(cartoon_Bilateral, cv2.COLOR_BGR2RGB)
    return cartoon_rgb

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
                                      ["Original", "Grayscale", "Cartoonify", "Blur", "Canny Edge Detection", "Sketch", "Histogram", "Crop"])
        
        params = {}
        if method == 'Blur':
            params['kernel_size'] = st.sidebar.slider("Gaussian Blur Kernel Size", min_value=1, max_value=31, value=15, step=2)
        elif method == 'Canny Edge Detection':
            params['threshold1'] = st.sidebar.slider("Threshold 1", min_value=0, max_value=255, value=100)
            params['threshold2'] = st.sidebar.slider("Threshold 2", min_value=0, max_value=255, value=200)
        elif method == 'Cartoonify':
            params['K'] = st.sidebar.slider("k", min_value=0, max_value=16, value=8)
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
