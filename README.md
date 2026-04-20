# Covete Detector

A computer vision system designed to detect *covetes* (egg trays) in a production line using OpenCV and feature matching techniques.

---

## 📌 Project Overview

This project performs automatic detection and classification of objects using video input.

Main features:

* Motion detection in a Region of Interest (ROI)
* Best frame selection based on:

  * Object centrality
  * Object size
  * Image sharpness
* Classification using ORB feature matching
* Automatic dataset generation

---

## ⚙️ Setup

Clone the repository:

```bash
git clone https://github.com/RodrigoH17/covete-detector.git
cd covete-detector
```

### Create virtual environment

```bash
python -m venv venv
```

### Activate environment

**Windows:**

```bash
venv\Scripts\activate
```
Se der erro colcoar primeiro `
```bash

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

1. Place your video inside the `videos/` folder
2. Update the path in the script:

```python
VIDEO_PATH = "videos/your_video.mp4"
```

3. Run:

```bash
python scripts/detected.py
```
---

## 🧠 Technologies

* Python
* OpenCV
* ORB Feature Matching

---

## 📌 Notes

* Large files (videos, datasets) are ignored via `.gitignore`
* Templates are required for detection
* The `videos/` folder is intentionally empty

---

## 👨‍💻 Author

Rodrigo Henriques
GitHub: https://github.com/RodrigoH17

---
