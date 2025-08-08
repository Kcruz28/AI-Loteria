import gradio as gr
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import threading
import time
from PIL import Image
import io


class LoteriaDetector:
    def __init__(self):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Load model
        try:
            self.model = YOLO("best.pt")
            self.model.to(self.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        # Game state
        self.class_color = {}
        self.green_cards = set()
        self.total_cards = 16

        # Tabla classes (board names - these shouldn't be detected as cards)
        self.tablas = [
            "tabla 1",
            "tabla 2",
            "tabla 3",
            "tabla 4",
            "tabla 5",
            "tabla 6",
            "tabla 7",
            "tabla 8",
            "tabla 9",
            "tabla 10",
        ]

    def reset_game(self):
        """Reset the game state"""
        self.class_color.clear()
        self.green_cards.clear()
        return "Game reset! All cards cleared."

    def process_image(self, image, confidence_threshold=0.7):
        """Process a single image and return annotated result"""
        if self.model is None:
            return image, "Model not loaded!"

        if image is None:
            return None, "No image provided"

        try:
            # Convert PIL to OpenCV format if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Run detection
            results = self.model(image, conf=confidence_threshold)
            annotated_frame = results[0].plot()

            # Process detections
            detected_classes = set()
            status_text = ""

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x_mid = int((x1 + x2) / 2)
                        y_mid = int((y1 + y2) / 2)
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())

                        detected_classes.add(cls)

                        # Skip tabla classes
                        if cls not in self.tablas:
                            color = (0, 0, 255)  # Red for unmatched
                            if cls in self.green_cards:
                                color = (0, 255, 0)  # Green for matched

                            # Draw detection marker
                            cv2.circle(annotated_frame, (x_mid, y_mid), 20, color, -1)
                            cv2.putText(
                                annotated_frame,
                                f"Class {cls} ({conf:.2f})",
                                (x_mid + 25, y_mid),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2,
                            )

            # Update game status
            status_text = f"Detected cards: {len(detected_classes)}\n"
            status_text += (
                f"Matched cards: {len(self.green_cards)}/{self.total_cards}\n"
            )

            # Check win condition
            if len(self.green_cards) >= self.total_cards:
                cv2.putText(
                    annotated_frame,
                    "LOTERIA!",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 0),
                    5,
                )
                status_text += "🎉 LOTERIA! You won! 🎉"
            else:
                status_text += f"Need {self.total_cards - len(self.green_cards)} more cards to win!"

            # Convert back to RGB for Gradio
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            return annotated_frame, status_text

        except Exception as e:
            return image, f"Error processing image: {str(e)}"

    def match_cards(self, image1, image2, confidence_threshold=0.7):
        """Process two images and match common detected cards"""
        if self.model is None:
            return None, None, "Model not loaded!"

        if image1 is None or image2 is None:
            return image1, image2, "Please provide both images"

        try:
            # Process both images
            def get_detections(img):
                if isinstance(img, Image.Image):
                    img = np.array(img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                results = self.model(img, conf=confidence_threshold)
                annotated = results[0].plot()

                detected_classes = set()
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0].item())
                            if cls not in self.tablas:  # Skip tabla classes
                                detected_classes.add(cls)

                return annotated, detected_classes

            # Get detections from both images
            annotated1, classes1 = get_detections(image1)
            annotated2, classes2 = get_detections(image2)

            # Find matching classes
            matching_classes = classes1.intersection(classes2)

            # Update green cards with matches
            for cls in matching_classes:
                self.green_cards.add(cls)
                self.class_color[cls] = True

            # Add green circles for matched cards
            def add_match_markers(annotated, detected_classes):
                for result in self.model(
                    cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                    conf=confidence_threshold,
                ):
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0].item())
                            if cls in matching_classes and cls not in self.tablas:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                x_mid = int((x1 + x2) / 2)
                                y_mid = int((y1 + y2) / 2)
                                cv2.circle(
                                    annotated, (x_mid, y_mid), 25, (0, 255, 0), 5
                                )

                return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Convert back to RGB and add match markers
            annotated1 = cv2.cvtColor(annotated1, cv2.COLOR_BGR2RGB)
            annotated2 = cv2.cvtColor(annotated2, cv2.COLOR_BGR2RGB)

            # Status message
            status = f"Camera 1 detected: {len(classes1)} cards\n"
            status += f"Camera 2 detected: {len(classes2)} cards\n"
            status += f"Matching cards: {len(matching_classes)}\n"
            status += f"Total matched: {len(self.green_cards)}/{self.total_cards}\n"

            if len(self.green_cards) >= self.total_cards:
                status += "🎉 LOTERIA! You won! 🎉"
                # Add LOTERIA text to both images
                cv2.putText(
                    annotated1,
                    "LOTERIA!",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    4,
                )
                cv2.putText(
                    annotated2,
                    "LOTERIA!",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    4,
                )
            else:
                status += f"Need {self.total_cards - len(self.green_cards)} more matches to win!"

            return annotated1, annotated2, status

        except Exception as e:
            return image1, image2, f"Error processing images: {str(e)}"


# Initialize detector
detector = LoteriaDetector()

#  Gradio interface
with gr.Blocks(title="🎲 AI Loteria Game", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
    # 🎲 AI Loteria Card Detection Game
    
    Upload images from your cameras to detect Loteria cards and play the game!
    
    **How to play:**
    1. **Single Image Mode**: Upload one image to see detected cards
    2. **Dual Camera Mode**: Upload images from both cameras to find matching cards
    3. Match all 16 cards to win LOTERIA! 🎉
    """
    )

    with gr.Tabs():
        # Single image tab
        with gr.TabItem("📷 Single Image Detection"):
            with gr.Row():
                with gr.Column():
                    single_input = gr.Image(label="Upload Image", type="pil")
                    single_confidence = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.05,
                        label="Confidence Threshold",
                    )
                    single_process_btn = gr.Button("🔍 Detect Cards", variant="primary")

                with gr.Column():
                    single_output = gr.Image(label="Detection Results")
                    single_status = gr.Textbox(label="Detection Status", lines=4)

        # Dual camera tab
        with gr.TabItem("📷📷 Dual Camera Matching"):
            with gr.Row():
                with gr.Column():
                    dual_input1 = gr.Image(label="Camera 1 Image", type="pil")
                    dual_input2 = gr.Image(label="Camera 2 Image", type="pil")
                    dual_confidence = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.05,
                        label="Confidence Threshold",
                    )
                    dual_process_btn = gr.Button("🎯 Match Cards", variant="primary")

                with gr.Column():
                    dual_output1 = gr.Image(label="Camera 1 Results")
                    dual_output2 = gr.Image(label="Camera 2 Results")
                    dual_status = gr.Textbox(label="Matching Status", lines=5)

    # Game controls
    with gr.Row():
        reset_btn = gr.Button("🔄 Reset Game", variant="secondary")
        reset_output = gr.Textbox(label="Reset Status", visible=False)

    # Game statistics
    gr.Markdown("### 📊 Game Statistics")
    with gr.Row():
        gr.Markdown("- **Goal**: Match 16 cards between cameras")
        gr.Markdown("- **Red circles**: Unmatched cards")
        gr.Markdown("- **Green circles**: Matched cards")

    # Event handlers
    single_process_btn.click(
        fn=detector.process_image,
        inputs=[single_input, single_confidence],
        outputs=[single_output, single_status],
    )

    dual_process_btn.click(
        fn=detector.match_cards,
        inputs=[dual_input1, dual_input2, dual_confidence],
        outputs=[dual_output1, dual_output2, dual_status],
    )

    reset_btn.click(fn=detector.reset_game, outputs=[reset_output])

if __name__ == "__main__":
    print("🎲 Starting AI Loteria Gradio App...")
    print("📱 Open the URL below in your browser to play!")

    demo.launch(
        server_name="0.0.0.0",  
        server_port=7860,
        share=True, 
        show_error=True,
    )
