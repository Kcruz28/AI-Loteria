# 🎲 AI Loteria Card Detection

A computer vision-powered Loteria game using YOLO object detection and a modern Gradio web interface. Detect, match, and play Loteria cards in real time using your webcam or uploaded images!

---

## 🚀 Features

- **Dual Camera Matching:** Detect and match Loteria cards from two camera feeds or images.
- **Single Image Detection:** Test card detection on any image.
- **Real-Time Feedback:** Visualize detected cards with red (unmatched) and green (matched) circles.
- **Progress Tracking:** See your matched cards count and win status.
- **Modern Web UI:** Clean, mobile-friendly Gradio interface with tabs and controls.
- **Game Controls:** Reset game, adjust detection confidence, and more.
- **Public Sharing:** Share your game with a public Gradio link.

---

## 🖥️ Demo

You can try the AI Loteria Card Detection app instantly online via Hugging Face Spaces:

👉 **[Launch Online Demo](https://huggingface.co/spaces/LaFlame10/AI_Loteria_Detection_YOLO8V)**

---

## 📦 Requirements

- Python 3.9+
- macOS (MPS), CUDA, or CPU
- Two webcams (for full dual camera experience)
- `best.pt` YOLO model file (place in project root)

Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🏁 Quick Start

### 1. **Run the Gradio Web App**

```bash
source venv/bin/activate
python gradio_app.py
```

- Open the local or public URL printed in the terminal.
- Upload images or use camera snapshots to play!

### 2. **Run the Classic Camera App**

```bash
source venv/bin/activate
python app.py
```

- Uses two webcams for real-time card detection and matching.
- Press `r` to reset, `q` to quit.

---

## 🗂️ Project Structure

```
AI Loteria/
├── app.py            # Classic dual-camera Loteria app
├── gradio_app.py     # Gradio web interface
├── requirements.txt  # Python dependencies
├── best.pt           # YOLO model weights (not included)
└── README.md         # Project documentation
```

---

## ⚙️ Configuration

- **Model:** Place your trained YOLO `best.pt` file in the project root.
- **Cameras:** The classic app uses camera 0 and 1 by default. Adjust in `app.py` if needed.
- **Detection Threshold:** Adjustable in the Gradio UI.

---

## 🧠 How It Works

- Uses YOLO (via Ultralytics) for object detection.
- Matches detected cards between two camera feeds or images.
- Tracks matched cards and displays win status ("LOTERIA!").
- Gradio app provides a user-friendly web interface for uploads and results.

---

## 🤝 Credits

- Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [Gradio](https://gradio.app/)
- Inspired by the classic Mexican game Lotería

---

## 💡 Tips

- For best results, use clear, well-lit images of Loteria cards.
- You can deploy the Gradio app to Hugging Face Spaces for free hosting.
- Extend the app with sound effects, leaderboards, or more game modes!

---

## 📝 Contact

For questions, suggestions, or collaborations, open an issue or reach out via GitHub.
