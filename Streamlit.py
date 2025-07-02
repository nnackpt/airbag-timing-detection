import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

# Import Segment Anything components
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    st.error("The 'segment_anything' library is not found. Please install it using:")
    st.code("pip install git+https://github.com/facebookresearch/segment-anything.git")
    st.stop()
    print("ERROR: segment_anything library not found.") # Debug print

print("Script started.") # Debug print

# --- Configuration ---
YOLO_MODEL_PATH = "Model/best.pt"
SAM_CHECKPOINT = "Model/sam_vit_b_01ec64.pth"
SAM_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}") # Debug print

# --- Streamlit App Setup ---
st.set_page_config(
    page_title="Airbag and Circle Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚗 Airbag and Circle Detection from Video")
st.markdown("""
This application processes a video to first segment objects (like airbags) using YOLO and Segment Anything,
and then detects circles in the segmented output.
""")

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_yolo_model(path):
    """Loads the YOLO model."""
    print(f"Attempting to load YOLO model from: {path}") # Debug print
    try:
        model = YOLO(path)
        print("YOLO model loaded successfully.") # Debug print
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model from {path}: {e}")
        print(f"ERROR: Failed to load YOLO model: {e}") # Debug print
        return None

@st.cache_resource
def load_sam_predictor(sam_type, checkpoint_path, device):
    """Loads the SAM model and predictor."""
    print(f"Attempting to load SAM model from: {checkpoint_path} with type {sam_type} on device {device}") # Debug print
    try:
        sam = sam_model_registry[sam_type](checkpoint=checkpoint_path).to(device)
        predictor = SamPredictor(sam)
        print("SAM model loaded successfully.") # Debug print
        return predictor
    except Exception as e:
        st.error(f"Error loading SAM model from {checkpoint_path}: {e}")
        st.info("Please ensure the SAM checkpoint file is correctly downloaded and placed in the 'Model' directory.")
        print(f"ERROR: Failed to load SAM model: {e}") # Debug print
        return None

# Load models at the start
yolo_model = load_yolo_model(YOLO_MODEL_PATH)
sam_predictor = load_sam_predictor(SAM_TYPE, SAM_CHECKPOINT, DEVICE)

if yolo_model is None or sam_predictor is None:
    st.warning("Please resolve the model loading errors to proceed.")
    st.stop()
    print("ERROR: Model loading failed, stopping app.") # Debug print
else:
    print("All models loaded. Proceeding with UI setup.") # Debug print


# --- Video Processing Functions ---

def segment_video(input_video_path, output_video_path, yolo_model, sam_predictor, confidence_threshold):
    """
    Processes the video to perform object segmentation using YOLO and SAM.
    """
    print(f"Starting segmentation for {input_video_path}") # Debug print
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file {input_video_path}")
        print(f"ERROR: Could not open video file {input_video_path}") # Debug print
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_text = "Segmentation in progress. Please wait."
    progress_bar = st.progress(0, text=progress_text)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # print(f"Processing frame {frame_count}/{total_frames} for segmentation...") # Too verbose for CMD
        progress_bar.progress(frame_count / total_frames, text=f"Processing frame {frame_count}/{total_frames} for segmentation...")

        image_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

        # Run YOLO to find bounding boxes
        results = yolo_model.predict(
            source=frame,
            conf=confidence_threshold,
            show=False,
            save=False,
            stream=False,
            verbose=False,
            device=DEVICE # Ensure YOLO uses the specified device
        )

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        # Send image to SAM
        sam_predictor.set_image(image_rgb)

        overlay = frame.copy()
        for box in boxes:
            x1, y1, x2, y2 = box
            input_box = np.array([x1, y1, x2, y2])

            masks, scores, _ = sam_predictor.predict(
                box=input_box[None, :],
                multimask_output=True
            )
            if masks.size > 0: # Check if masks were found
                best_mask = masks[np.argmax(scores)]

                # Draw Transparent Mask (Green)
                green_mask = np.zeros_like(frame, dtype=np.uint8)
                green_mask[best_mask] = (0, 255, 0)
                overlay = cv2.addWeighted(overlay, 1.0, green_mask, 0.5, 0)

        out.write(overlay)

    cap.release()
    out.release()
    progress_bar.empty() # Clear the progress bar
    print("Segmentation complete.") # Debug print
    return True

def detect_circles(input_video_path, output_video_path):
    """
    Detects circles in the processed video frames.
    """
    print(f"Starting circle detection for {input_video_path}") # Debug print
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file {input_video_path}")
        print(f"ERROR: Could not open video file {input_video_path}") # Debug print
        return False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_video_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (frame_width, frame_height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_text = "Circle detection in progress. Please wait."
    progress_bar = st.progress(0, text=progress_text)


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # print(f"Processing frame {frame_count}/{total_frames} for circle detection...") # Too verbose for CMD
        progress_bar.progress(frame_count / total_frames, text=f"Processing frame {frame_count}/{total_frames} for circle detection...")

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # Detect Circle
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=25,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=30)

        # Draw Circle if found
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw Circle
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw Centroid
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

        out.write(frame)

    cap.release()
    out.release()
    progress_bar.empty() # Clear the progress bar
    print("Circle detection complete.") # Debug print
    return True


# --- Streamlit UI ---

print("Setting up sidebar.") # Debug print
st.sidebar.header("Configuration")
confidence_threshold = st.sidebar.slider(
    "YOLO Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Minimum confidence score for YOLO to detect an object."
)

st.sidebar.markdown(f"**Using Device:** `{DEVICE}`")
if DEVICE == "cpu":
    st.sidebar.warning("Running on CPU. For better performance, consider using a GPU if available.")

print("Waiting for file upload.") # Debug print
uploaded_file = st.file_uploader("Upload a video file (e.g., .mp4, .avi)", type=["mp4", "avi"])

if uploaded_file is not None:
    print("File uploaded.") # Debug print
    st.video(uploaded_file, format="video/mp4", start_time=0)

    # Create temporary files for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_video:
        temp_input_video.write(uploaded_file.read())
        input_video_path = temp_input_video.name
    print(f"Input video temp path: {input_video_path}") # Debug print

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_segmented_video:
        segmented_video_path = temp_segmented_video.name
    print(f"Segmented video temp path: {segmented_video_path}") # Debug print

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output_video:
        output_video_path = temp_output_video.name
    print(f"Output video temp path: {output_video_path}") # Debug print

    st.info("Video uploaded. Starting processing...")

    try:
        print("Starting segmentation process.") # Debug print
        with st.spinner("Step 1/2: Performing object segmentation (YOLO + SAM)..."):
            segmentation_success = segment_video(
                input_video_path,
                segmented_video_path,
                yolo_model,
                sam_predictor,
                confidence_threshold
            )

        if segmentation_success:
            st.success("Step 1: Segmentation complete!")
            st.video(segmented_video_path, format="video/mp4", start_time=0)
            print("Segmentation successful, starting circle detection.") # Debug print

            with st.spinner("Step 2/2: Detecting circles in the segmented video..."):
                circle_detection_success = detect_circles(
                    segmented_video_path,
                    output_video_path
                )

            if circle_detection_success:
                st.success("Step 2: Circle detection complete!")
                st.subheader("Processed Video Output")
                st.video(output_video_path, format="video/mp4", start_time=0)
                print("Circle detection successful.") # Debug print

                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="Download Processed Video",
                        data=file.read(),
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
            else:
                st.error("Circle detection failed.")
                print("ERROR: Circle detection failed.") # Debug print
        else:
            st.error("Segmentation failed.")
            print("ERROR: Segmentation failed.") # Debug print

    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")
        st.exception(e) # Display full traceback for debugging
        print(f"CRITICAL ERROR: {e}") # Debug print
    finally:
        # Clean up temporary files
        print("Cleaning up temporary files.") # Debug print
        if os.path.exists(input_video_path):
            os.remove(input_video_path)
        if os.path.exists(segmented_video_path):
            os.remove(segmented_video_path)
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        st.info("Temporary files cleaned up.")

print("Script finished.") # Debug print
st.markdown("---")
st.markdown("""
**Instructions for setting up:**
1.  **Save this code** as `Streamlit.py` (หรือชื่อที่คุณใช้).
2.  **Create a folder** named `Model` in the same directory as your script.
3.  **Download your models** (`best.pt` for YOLO and `sam_vit_b_01ec64.pth` for SAM) and place them inside the `Model` folder.
4.  **Install necessary libraries**:
    ```bash
    pip install streamlit opencv-python numpy ultralytics torch
    pip install git+[https://github.com/facebookresearch/segment-anything.git](https://github.com/facebookresearch/segment-anything.git)
    ```
5.  **Run the app** from your terminal in the same directory:
    ```bash
    streamlit run Streamlit.py
    ```
""")
