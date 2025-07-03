import gradio as gr
import cv2
import numpy as np
import os

def ensure_grayscale(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def to_rgb(image):
    if image is None:
        return None
    if len(image.shape) == 2:  # grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply FFT transform with optional individual frequency inspection
def apply_fft(image, eps, mode, min_radius_ratio, max_radius_ratio):
    gray = ensure_grayscale(image)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    if mode == "Full FFT":
        mag = 20 * np.log(np.abs(fshift) + eps)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    else:
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        max_dist = np.sqrt((cx)**2 + (cy)**2)  # max radius from center to corner

        # Convert ratios to pixel distances
        min_r = min_radius_ratio * max_dist
        max_r = max_radius_ratio * max_dist

        # Create a circular band-pass mask
        mask = (dist >= min_r) & (dist <= max_r)
        fshift_masked = np.zeros_like(fshift, dtype=complex)
        fshift_masked[mask] = fshift[mask]

        f_ishift = np.fft.ifftshift(fshift_masked)
        reconstructed = np.fft.ifft2(f_ishift)
        reconstructed_img = np.abs(reconstructed)
        return cv2.normalize(reconstructed_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_saturation_map(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1]

def apply_blur(image):
    return cv2.GaussianBlur(image, (9, 9), 0)

def apply_hue_map(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 0]

def apply_sobel(image, ksize):
    gray = ensure_grayscale(image)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(sobelx**2 + sobely**2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_laplacian(image, ksize):
    gray = ensure_grayscale(image)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(lap)

def apply_custom_kernel(image, kernel_vals):
    gray = ensure_grayscale(image)
    kernel = np.array(kernel_vals, dtype=np.float32)
    return cv2.filter2D(gray, -1, kernel)

def analyze(img1, img2, transform_type, filter_type, eps, mode, min_radius_ratio, max_radius_ratio, ksize, kernel_vals, display_choice):
    def transform(img):
        if transform_type == "Saturation Map":
            return apply_saturation_map(img)
        elif transform_type == "Blur":
            return apply_blur(img)
        elif transform_type == "Hue Map":
            return apply_hue_map(img)
        return img

    def apply_filter(img):
        if filter_type == "FFT Magnitude":
            return apply_fft(img, eps, mode, min_radius_ratio, max_radius_ratio)
        elif filter_type == "Edge Detection (Sobel)":
            return apply_sobel(img, ksize)
        elif filter_type == "Laplacian Edge Detection":
            return apply_laplacian(img, ksize)
        elif filter_type == "Custom 3x3 Kernel":
            return apply_custom_kernel(img, kernel_vals)
        return img

    out1 = out2 = None
    if img1 is not None:
        img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
        out1 = apply_filter(transform(img1))

    if img2 is not None:
        img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
        out2 = apply_filter(transform(img2))

    out1 = to_rgb(out1)
    out2 = to_rgb(out2)
    if display_choice == "Image 1":
        return out1, None
    elif display_choice == "Image 2":
        return out2, None
    elif display_choice == "Comparison":
        if out1 is not None and out2 is not None:
            return None, (out1, out2)
    return None, None

# Gradio interface setup
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion(visible=True, label="Input Images"):
                with gr.Row():
                    img1 = gr.Image(type="numpy", label="Image 1", height=200)
                    img2 = gr.Image(type="numpy", label="Image 2 (optional)", height=200)
            transform_type = gr.Radio([
                "None", "Saturation Map", "Blur", "Hue Map"
            ], value="None", label="Transform")

            filter_type = gr.Radio([
                "None",
                "FFT Magnitude",
                "Edge Detection (Sobel)",
                "Laplacian Edge Detection",
                "Custom 3x3 Kernel"
            ], value="FFT Magnitude", label="Filter Type")

            with gr.Accordion(visible=True, label="FFT options") as fft_group:
                mode = gr.Radio(["Full FFT", "Individual Frequency"], value="Full FFT", label="FFT Mode")
                eps = gr.Slider(0.1, 10.0, value=1.0, label="Log scale epsilon")
                min_radius_ratio = gr.Slider(0.0, 1.0, step=0.001, value=0.0, label="Min Radius Ratio (Inner)")
                max_radius_ratio = gr.Slider(0.0, 1.0, step=0.001, value=0.2, label="Max Radius Ratio (Outer)")

            with gr.Accordion(visible=False, label="Kernel Options") as ksize_group:
                ksize = gr.Slider(1, 31, step=2, value=3, label="Kernel Size")

            with gr.Accordion(visible=False, label="Custom Kernel Options") as custom_group:
                kernel_vals = gr.Dataframe([[0,0,0],[0,3,0],[0,0,0]], type="numpy", label="3x3 Kernel", show_row_numbers=False)

        with gr.Column(scale=3):
            out_display = gr.Image(type="numpy", label="Filtered Image 1", visible=True)
            slider_display = gr.ImageSlider(label="Display Filtered Output", visible=False)
            display_choice = gr.Radio([
                "Image 1", "Image 2", "Comparison"
            ], label="Display Option", value="Image 1")

    def toggle_visibility(filter_type):
        return (
            gr.update(visible=filter_type == "FFT Magnitude"),
            gr.update(visible=filter_type in ["Edge Detection (Sobel)", "Laplacian Edge Detection"]),
            gr.update(visible=filter_type == "Custom 3x3 Kernel")
        )

    def update_display(display_choice):
        if display_choice == "Image 1":
            return gr.update(visible=True), gr.update(visible=False)
        elif display_choice == "Image 2":
            return gr.update(visible=True), gr.update(visible=False)
        elif display_choice == "Comparison":
            return gr.update(visible=False), gr.update(visible=True)

    filter_type.change(fn=toggle_visibility, inputs=filter_type, outputs=[fft_group, ksize_group, custom_group])
    inputs = [img1, img2, transform_type, filter_type, eps, mode, min_radius_ratio, max_radius_ratio, ksize, kernel_vals, display_choice]
    outputs = [out_display, slider_display]
    for inp in inputs:
        inp.change(fn=analyze, inputs=inputs, outputs=outputs)

    display_choice.change(fn=update_display, inputs=[display_choice], outputs=[out_display, slider_display])

if __name__ == "__main__":
    demo.launch()
