## 🚀 Project Overview

The system captures video from two **ESP32-CAM** modules, processes the data using Python (OpenCV), calculates the 3D coordinates ($X, Y, Z$) of the glove, and streams this data to Unity via UDP in real-time.

### Key Features

* **Stereoscopic Depth:** Uses triangulation to calculate true 3D depth, not just 2D position.

* **Lag-Free Threading:** Custom `CameraStream` classes run video fetching in background threads to prevent network latency from blocking the processing loop.

* **Tilt Correction:** Mathematically rotates the 3D world to account for the cameras being mounted at a 50° angle.

* **Visualizer:** Includes a real-time 3D bar chart to debug tracking boundaries without needing to look at Unity.

## 🛠 Hardware Setup

1. **Cameras:** 2x **ESP32-CAM** modules (configured as MJPEG streamers).

2. **Mounting:** Ceiling or high-rig mount.

   * **Position:** Top-down, angled \~50° downwards towards the play area.

   * **Alignment:** Cameras should be roughly parallel.

3. **Controller:** A colored boxing glove.

4. **Network:** A dedicated 2.4GHz WiFi hotspot is recommended for low latency.

## 📂 Project Structure

| Location | Description |
| :---- | :---- |
| main.py | **The Main Engine.** Run this to start the tracking system. |
| core/ | **Library Logic.** Contains the camera threads (camera.py), math (vision.py), and settings (config.py). |
| tools/align\_cameras.py | **Step 1:** Visual aid to physically aim the cameras so they overlap correctly. |
| tools/calibrate.py | **Step 2:** Uses a checkerboard to calculate lens distortion and stereo geometry. |
| tools/color\_tuner.py | **Step 3:** A tool with sliders to find the perfect HSV color values for your glove. |
| data/stereo\_calib.yml | **Generated File.** Stores the intrinsic and extrinsic camera matrices (created by Step 2). |

## ⚙️ Installation & Requirements

### Python Dependencies

Ensure you have Python 3.8+ installed. Install the required libraries:

```bash
pip install -r requirements.txt
```

### Configuration

Open `core/config.py` to change settings. **You must update the IP addresses here:**

```python
# core/config.py
CAM_IP_1 = "192.168.1.101" 
CAM_IP_2 = "192.168.1.102"
```

## ⚡ Usage Workflow

Follow these steps in order to set up the system.

### 1. Physical Alignment

Run the alignment tool to see both camera feeds side-by-side with crosshairs.

```bash
python tools/align_cameras.py
```

* **Goal:** Adjust your physical camera mounts until the crosshairs of both cameras point to the exact same spot in the center of your play area.

### 2. Stereo Calibration

Print a checkerboard pattern (configured for 9x6 inner corners, 25mm squares).

```bash
python tools/calibrate.py
```

* Hold the board visible to both cameras.

* Press **[Space]** to capture pairs of images (aim for >15 pairs).

* Press **[c]** to calculate math.

* **Result:** This will save a `stereo_calib.yml` file into the `data/` folder.

### 3. Color Tuning

Run the tuner to isolate your glove from the background.

```bash
python tools/color_tuner.py
```

* Click on your glove in the "Picker" window.

* Adjust the HSV sliders until the glove is white and the background is black in the "Mask" window.

* **Press [s]** to print the values. Copy these into `core/config.py`.

### 4. Start Tracking

Run the main engine.

```bash
python main.py
```

* This script connects to the cameras, applies the calibration from `data/`, tracks the glove, and sends UDP packets to `127.0.0.1:5005`.

* **Visualizer:** Watch the "3D Data Visualizer" window to see the interpreted X, Y, Z positions.

## 🎮 Unity Integration

The system broadcasts a string via UDP in the format:
`"X,Y,Z"`

To receive this in Unity:

1. Create a C# script (e.g., `UDPReceiver.cs`).

2. Use `UdpClient` to listen on Port **5005**.

3. Parse the string and map it to your game object.

*Note: The Python script swaps axes to match Unity standards:*

* **Python Y** (Distance from camera) → **Unity Z** (Forward/Back)

* **Python Z** (Depth/Height) → **Unity Y** (Up/Down)

## 🔧 Troubleshooting & Logic

### Coordinate Rotation

Because the cameras are angled down at 50°, a raw Z-depth calculation would result in the coordinate moving "down" as you punch "forward". The `apply_tilt_correction` function (in `core/vision.py`) uses a rotation matrix to correct this:

$$
y_{new} = y \cdot \cos(\theta) - z \cdot \sin(\theta)
$$

$$
z_{new} = y \cdot \sin(\theta) + z \cdot \cos(\theta)
$$

### Network Lag

If video feeds lag behind reality:

1. Ensure you are using the `CameraStream` threaded class (default in `core/camera.py`).

2. Lower the `framesize` in `core/config.py` (e.g., use `framesize=8` for VGA instead of UXGA).

3. Ensure your PC is connected to the same WiFi router via Ethernet cable if possible.

### Calibration Errors

If the "Rectified" images look warped or swirled:

* Ensure the checkerboard was perfectly flat during calibration.

* Ensure you captured the board at the edges of the frame, not just the center.