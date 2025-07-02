import streamlit as st
import cv2
import numpy as np
from streamlit_image_comparison import image_comparison
from utils import center_align_images  # Assuming you move it into a separate utils.py

st.set_page_config(layout="wide")

# === Sidebar Upload ===
uploaded_files = st.sidebar.file_uploader(
    "Upload one or two images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# === Sidebar Filter Selection ===
filter_type = st.sidebar.radio(
    "Select Filter",
    [
        "FFT Magnitude",
        "Saturation Map",
        "Edge Detection (Sobel)",
        "Laplacian Edge Detection",
        "Custom 3x3 Kernel"
    ]
)

# === Filter Functions ===
def apply_fft(image, eps=1.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + eps)

    h, w = gray.shape
    center_x, center_y = w // 2, h // 2

    x_offset = st.slider("Horizontal Frequency Offset", -center_x, center_x, 0)
    y_offset = st.slider("Vertical Frequency Offset", -center_y, center_y, 0)
    window_size = st.slider("Frequency Window Size", 1, 15, 5, step=2)

    # Convert offset to actual FFT coordinates
    x = center_x + x_offset
    y = center_y + y_offset

    # Create a new FFT with only a small frequency region around (x, y)
    fshift_masked = np.zeros_like(fshift, dtype=complex)
    half_win = window_size // 2
    fshift_masked[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1] = \
        fshift[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1]

    # Inverse FFT to reconstruct the image contribution of selected frequency
    f_ishift = np.fft.ifftshift(fshift_masked)
    reconstructed = np.fft.ifft2(f_ishift)
    reconstructed_img = np.abs(reconstructed)
    reconstructed_img = cv2.normalize(reconstructed_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_saturation_map(image, colormap="JET"):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    if colormap == "None":
        return saturation
    cmap_code = getattr(cv2, f"COLORMAP_{colormap}")
    return cv2.applyColorMap(saturation, cmap_code)

def apply_sobel(image, ksize=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(sobelx**2 + sobely**2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_laplacian(image, ksize=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(lap)

def apply_custom_kernel(image, kernel_vals):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array(kernel_vals, dtype=np.float32)
    filtered = cv2.filter2D(gray, -1, kernel)
    return filtered

# === UI Parameter Control ===
def get_filter_parameters():
    if filter_type == "FFT Magnitude":
        eps = st.slider("Log scale epsilon (to avoid log(0))", 0.1, 10.0, 1.0)
        return {"eps": eps}

    elif filter_type == "Saturation Map":
        colormap = st.selectbox("Colormap", ["JET", "BONE", "None"], index=0)
        return {"colormap": colormap}

    elif filter_type == "Edge Detection (Sobel)":
        ksize = st.slider("Kernel Size (odd numbers only)", 1, 31, 3, step=2)
        return {"ksize": ksize}

    elif filter_type == "Laplacian Edge Detection":
        ksize = st.slider("Kernel Size (odd numbers only)", 1, 31, 3, step=2)
        return {"ksize": ksize}

    elif filter_type == "Custom 3x3 Kernel":
        st.write("Custom 3×3 Kernel")
        kernel_vals = []
        cols = st.columns(3)
        default_kernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        for i in range(3):
            row = []
            for j in range(3):
                val = cols[j].number_input(
                    "", value=default_kernel[i][j], step=1, key=f"k_{i}_{j}"
                )
                row.append(val)
            kernel_vals.append(row)
        return {"kernel_vals": kernel_vals}

    return {}

# === Filtering Logic ===
def apply_selected_filter(image, params):
    if filter_type == "FFT Magnitude":
        return apply_fft(image, **params)
    elif filter_type == "Saturation Map":
        return apply_saturation_map(image, **params)
    elif filter_type == "Edge Detection (Sobel)":
        return apply_sobel(image, **params)
    elif filter_type == "Laplacian Edge Detection":
        return apply_laplacian(image, **params)
    elif filter_type == "Custom 3x3 Kernel":
        return apply_custom_kernel(image, **params)
    else:
        return image

# === Main Display ===
if len(uploaded_files) == 0:
    st.info("👈 Upload at least one image to begin.")
else:
    images = []
    for file in uploaded_files[:2]:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        images.append(img)

    st.sidebar.image(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB), caption="Image 1", use_container_width=True)
    if len(images) > 1:
        st.sidebar.image(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB), caption="Image 2", use_container_width=True)

    # Collect UI parameters once
    filter_params = get_filter_parameters()

    # Apply filter to each uploaded image
    processed_images = [apply_selected_filter(img, filter_params) for img in images]

    # Show comparison or single image
    if len(processed_images) == 2:
        img1_aligned, img2_aligned = center_align_images(processed_images[0], processed_images[1])
        image_comparison(
            img1=img1_aligned,
            img2=img2_aligned,
            label1="Image 1",
            label2="Image 2",
            width=1400
        )
    else:
        st.markdown("### Filtered Image")
        st.image(processed_images[0], use_container_width=True, clamp=True)
