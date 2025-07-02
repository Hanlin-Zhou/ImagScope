import gradio as gr
import cv2
import numpy as np
import os

# Helper function to center-align two images by padding them to the same size
def center_align_images(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h, w = max(h1, h2), max(w1, w2)

    def center_pad(img, target_h, target_w):
        top = (target_h - img.shape[0]) // 2
        bottom = target_h - img.shape[0] - top
        left = (target_w - img.shape[1]) // 2
        right = target_w - img.shape[1] - left
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    return center_pad(img1, h, w), center_pad(img2, h, w)

# Apply FFT transform with optional individual frequency inspection
def apply_fft(image, eps, mode, min_radius_ratio, max_radius_ratio):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

def apply_saturation_map(image, colormap):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    if colormap == "None":
        return saturation
    cmap_code = getattr(cv2, f"COLORMAP_{colormap}")
    return cv2.applyColorMap(saturation, cmap_code)

def apply_sobel(image, ksize):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(sobelx**2 + sobely**2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_laplacian(image, ksize):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(lap)

def apply_custom_kernel(image, kernel_vals):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array(kernel_vals, dtype=np.float32)
    return cv2.filter2D(gray, -1, kernel)

def analyze(img1, img2, filter_type, eps, mode, min_radius_ratio, max_radius_ratio, colormap, ksize, kernel_vals, display_choice):
    def apply_filter(img):
        if filter_type == "FFT Magnitude":
            return apply_fft(img, eps, mode, min_radius_ratio, max_radius_ratio,)
        elif filter_type == "Saturation Map":
            return apply_saturation_map(img, colormap)
        elif filter_type == "Edge Detection (Sobel)":
            return apply_sobel(img, ksize)
        elif filter_type == "Laplacian Edge Detection":
            return apply_laplacian(img, ksize)
        elif filter_type == "Custom 3x3 Kernel":
            return apply_custom_kernel(img, kernel_vals)
        return img

    out1 = out2 = None
    print("Analyzing...")
    if img1 is not None:
        img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
        out1 = apply_filter(img1)

    if img2 is not None:
        img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
        out2 = apply_filter(img2)

    if display_choice == "Image 1":
        return out1, None
    elif display_choice == "Image 2":
        return out2, None
    elif display_choice == "Comparison":
        if out1 is not None and out2 is not None:
            # aligned1, aligned2 = center_align_images(out1, out2)
            return None, (out1, out2)
    return None, None

# Gradio interface setup
with gr.Blocks() as demo:
    # gr.Markdown("# Greetings from Gradio!")

    # Define states to hold outputs
    

    with gr.Row():
        # Left column: upload + filter controls
        with gr.Column(scale=3):
            with gr.Row():
                img1 = gr.Image(type="numpy", label="Image 1")
                img2 = gr.Image(type="numpy", label="Image 2 (optional)")
            filter_type = gr.Radio([
                "FFT Magnitude",
                "Saturation Map",
                "Edge Detection (Sobel)",
                "Laplacian Edge Detection",
                "Custom 3x3 Kernel"
            ], value="FFT Magnitude", 
            label="Filter Type")

            with gr.Column(visible=True) as fft_group:
                mode = gr.Radio(["Full FFT", "Individual Frequency"], value="Full FFT", label="FFT Mode")
                eps = gr.Slider(0.1, 10.0, value=1.0, label="Log scale epsilon")
                min_radius_ratio = gr.Slider(0.0, 1.0, step=0.001, value=0.0, label="Min Radius Ratio (Inner)")
                max_radius_ratio = gr.Slider(0.0, 1.0, step=0.001, value=0.2, label="Max Radius Ratio (Outer)")

            with gr.Column(visible=False) as sat_group:
                colormap = gr.Dropdown(["JET", "BONE", "None"], value="JET", label="Colormap")

            with gr.Column(visible=False) as ksize_group:
                ksize = gr.Slider(1, 31, step=2, value=3, label="Kernel Size")

            with gr.Column(visible=False) as custom_group:
                kernel_vals = gr.Dataframe([[0,0,0],[0,3,0],[0,0,0]], type="numpy", label="3x3 Kernel", show_row_numbers=False)


        # Right column: outputs
        with gr.Column(scale=3):
            out_display = gr.Image(type="numpy", label="Filtered Image 1", visible=True)
            slider_display = gr.ImageSlider(label="Display Filtered Output", visible=False)
            display_choice = gr.Radio(
            ["Image 1", "Image 2", "Comparison"], label="Display Option", value="Image 1"
        )

    def toggle_visibility(filter_type):
        return (
            gr.update(visible=filter_type == "FFT Magnitude"),
            gr.update(visible=filter_type == "Saturation Map"),
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


    filter_type.change(fn=toggle_visibility, inputs=filter_type, outputs=[fft_group, sat_group, ksize_group, custom_group])
    inputs = [img1, img2, filter_type, eps, mode, min_radius_ratio, max_radius_ratio, colormap, ksize, kernel_vals, display_choice]
    outputs = [out_display, slider_display]
    for inp in inputs:
        inp.change(fn=analyze, inputs=inputs, outputs=outputs)
    # out1.change(fn=update_display, inputs=[display_choice, out1, out2], outputs=[out_display, slider_display])
    # out2.change(fn=update_display, inputs=[display_choice, out1, out2], outputs=[out_display, slider_display])
    display_choice.change(fn=update_display, inputs=[display_choice], outputs=[out_display, slider_display])


if __name__ == "__main__":
    demo.launch()
