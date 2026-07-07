# MeowMirror

A face-tracking mirror app using your webcam. Detects faces in real time and draws a box around them.

## Requirements

- Python 3.11 or 3.12 (Python 3.13 is **not** recommended — recent versions of `mediapipe` have a bug where `mp.solutions` fails to load on 3.13)

## Setup

1. **Clone the repo:**
   ```
   git clone https://github.com/pusheen5000000/MeowMirror.git
   cd MeowMirror
   ```

2. **Create a virtual environment (recommended):**
   ```
   python -m venv mp_env
   ```

3. **Activate it:**
   - Windows:
     ```
     .\mp_env\Scripts\activate
     ```
   - Mac/Linux:
     ```
     source mp_env/bin/activate
     ```

4. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

5. **Run it:**
   ```
   python meowmirror.py
   ```

   Press `q` to quit.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'mediapipe'` or `'cv2'`** — make sure your virtual environment is activated and you installed via `requirements.txt`, not just `pip install mediapipe` globally.
- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`** — this means a broken/incompatible mediapipe version got installed. Make sure you're on Python 3.11 or 3.12, then run:
  ```
  pip uninstall mediapipe -y
  pip install mediapipe==0.10.14
  ```
- **No webcam window opens** — check that no other app is using your webcam, and that your webcam permissions are enabled for your terminal/editor.
