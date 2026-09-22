# 🌸 FlowerBouquet

### *A bouquet that grows with your hands.*

> **Pinch. Open. Grow. Bloom.**
>
> A real-time interactive flower garden built with **Python, OpenCV, MediaPipe, and procedural graphics** — where your hands become the controls for a living bouquet.

<br>

<p align="center">

🌱 **Pinch** → Closed buds
🌷 **Open** → Bloom
🌿 **Left hand** → Grow the stems
🌸 **Right hand** → Control the bloom

</p>

---

## ✦ What is this?

**FlowerBouquet** is an interactive computer-vision experiment where a webcam becomes the interface.

Instead of clicking buttons or using a mouse, you control a procedural bouquet using your hands.

The flowers begin as tiny buds gathered together. As you interact with them, organic curved stalks grow outward, the flowers open, sway naturally, breathe independently, and move together like a real hand-tied bouquet.

No image assets.

No pre-rendered flowers.

Everything is generated **in real time**.

---

## 🌷 The Interaction

| Your hand | Gesture        | Effect                  |
| --------- | -------------- | ----------------------- |
| 🤚 Left   | Pinch          | Stems shrink            |
| 🤚 Left   | Open           | Stems grow              |
| 🤚 Right  | Pinch          | Flowers close into buds |
| 🤚 Right  | Open           | Flowers bloom           |
| 🫶 Both   | Move naturally | Bouquet responds        |

The controls are continuous rather than binary, so the bouquet responds gradually to your hand position.

---

## ✨ What makes it different?

This isn't just a webcam filter.

The bouquet is procedurally constructed and animated.

### 🌱 Organic growth

The stems don't simply appear as straight lines.

Each stem has its own:

* curvature
* length
* direction
* bend
* sway
* depth

They originate from a shared gathering point to create the appearance of a **hand-tied bouquet**.

### 🌸 Procedural flowers

The flowers are drawn mathematically rather than loaded from PNGs or sprites.

Their geometry is generated dynamically using OpenCV primitives and curves.

### 🌬️ Natural movement

The bouquet continuously responds to a subtle simulated breeze.

Movement is strongest toward the flower heads while the gathering point remains relatively stable.

### 💗 Breathing flowers

Each flower has an independent breathing cycle.

The result is intentionally subtle:

> not an animation playing on top of the flowers, but a bouquet that feels slightly alive.

### 🎀 Hand-tied bouquet

A procedural ribbon wraps around the gathered stems.

The stems partially disappear beneath the ribbon to create a small sense of physical depth and occlusion.

---

# 🧠 How it works

```text
                 WEBCAM
                    │
                    ▼
            ┌───────────────┐
            │    OpenCV     │
            │ Video Capture │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │   MediaPipe   │
            │ Hand Tracking │
            └───────┬───────┘
                    │
             Hand Landmarks
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
      LEFT HAND           RIGHT HAND
          │                   │
          ▼                   ▼
     Stem Growth          Bloom Control
          │                   │
          └─────────┬─────────┘
                    ▼
             ┌─────────────┐
             │   Flower    │
             │   Engine    │
             └──────┬──────┘
                    │
                    ▼
            Procedural Bouquet
                    │
                    ▼
             Real-time Render
```

---

# 🛠️ Built With

### Computer Vision

* **Python**
* **OpenCV**
* **MediaPipe**

### Mathematics & Rendering

* **NumPy**
* Procedural geometry
* Curve interpolation
* Parametric animation
* Real-time compositing
* Alpha blending

### Interface

* Webcam
* Hand landmark tracking
* Gesture-based interaction

---

# 📁 Project Structure

```text
FlowerBouquet/
│
├── main.py
├── flower.py
├── hand_tracker.py
├── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

### `main.py`

The application entry point.

Responsible for:

* webcam capture
* application loop
* hand input
* bouquet updates
* rendering

### `hand_tracker.py`

Handles:

* MediaPipe initialization
* hand landmark detection
* gesture estimation
* left/right hand identification
* pinch/open measurements

### `flower.py`

The heart of the visual system.

Responsible for:

* flower geometry
* buds
* blooming
* stalk curves
* sway
* breathing
* colors
* bouquet behavior
* ribbon rendering

### `utils.py`

Contains reusable helpers for:

* smoothing
* interpolation
* coordinate conversion
* drawing
* animation utilities

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/Akshaya-Sruti/FlowerBouquet.git
cd FlowerBouquet
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## 4. Run

```powershell
python main.py
```

Make sure your webcam is available.

---

# 🎮 Controls

### Left Hand 🌿

```text
PINCH ───────────────► Short / hidden stems

        ↕ continuous

OPEN ────────────────► Long stems
```

### Right Hand 🌸

```text
PINCH ───────────────► Buds

        ↕ continuous

OPEN ────────────────► Full bloom
```

### Keyboard

```text
Q / ESC  →  Exit
```

---

# 🌸 Design Philosophy

FlowerBouquet was built around a simple idea:

> **What if a computer-generated flower could feel less like a graphic and more like something alive?**

Instead of trying to make the most realistic botanical simulation possible, the project focuses on **perceived naturalness**.

Small imperfections are intentional.

Different stems curve differently.

Flowers breathe at different phases.

The bouquet sways rather than rotating rigidly.

The stems gather rather than forming a perfect radial pattern.

The ribbon hides a small portion of the stems.

These tiny details are what make the procedural system feel organic.

---

# 🧩 Procedural, not pre-rendered

There are **no flower PNGs or sprite sheets** required for the bouquet.

The visual system generates the flowers dynamically.

That means:

```text
Geometry
   +
Mathematics
   +
Motion
   +
Hand Input
   ↓
Living Bouquet
```

The same system can therefore produce variations without requiring new image assets.

---

# 🔮 Possible Future Experiments

This project can evolve far beyond a single bouquet.

Some ideas:

* 🌹 Different flower species
* 🌻 Seasonal bouquets
* 🌈 Gesture-controlled color palettes
* 🌧️ Rain interaction
* ☀️ Light-sensitive flowers
* 🌙 Night-mode gardens
* 🦋 Procedural butterflies
* 🎵 Music-reactive blooming
* 👥 Two-person interaction
* 🪴 Multiple bouquet types
* 🧠 Gesture learning
* 🥽 VR / XR version

---

# 💭 Why I Built This

Most computer-vision projects end with:

> detect something → display something.

I wanted to explore something different:

> **detect something → create an experience.**

FlowerBouquet is an experiment in combining:

**Computer Vision × Procedural Graphics × Interaction Design**

into something playful.

---

## 🌷 Made with curiosity, code & a little bit of magic.

<p align="center">

**Python · OpenCV · MediaPipe · NumPy**

🌱 → 🌿 → 🌷 → 🌸

</p>

---

<p align="center">
  <i>Touch nothing. Just move your hands.</i>
</p>
