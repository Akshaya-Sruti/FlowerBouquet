# 🌸 FlowerBouquet

### A little garden that listens.

> Some things are better grown quietly.

FlowerBouquet is a small interactive experiment built around flowers, movement, and the little details that make something feel alive.

It started as a simple hand-controlled OpenCV project and slowly became something softer: a bouquet that grows, sways, blooms, and responds to you.

---

## ✿ The Idea

The bouquet is controlled through hand gestures.

Instead of simply displaying flowers, the project treats the bouquet as a living little system.

* 🌱 **Left hand** controls the height of the stems
* 🌸 **Right hand** controls how much the flowers bloom
* 🤏 **Pinch** makes them smaller
* 🖐️ **Open hand** lets them grow and bloom
* 🌿 Every stem moves slightly differently
* 🌬️ The bouquet has a gentle natural sway
* ✨ Each flower breathes independently

The goal is not perfect symmetry.

It is supposed to feel a little alive.

---

## 🌷 The Bouquet

Every flower is generated procedurally.

There are no pre-rendered flower images.

Each flower is built from shapes, curves, layers, and carefully controlled motion.

The stems follow organic curved paths rather than perfectly straight lines, while the flowers have their own small variations in:

* curvature
* angle
* length
* depth
* sway
* bloom timing
* breathing motion

Together, these small differences make the bouquet feel less like a collection of identical objects and more like one gathered bunch of flowers.

---

## 🎀 One Small Detail

The flowers come together around a shared base and are held with a soft procedural ribbon.

The ribbon has:

* subtle folds
* gentle highlights
* natural curvature
* slightly uneven tails
* depth-aware layering

A small portion of the stems disappears underneath it, so the bouquet feels tied together rather than assembled separately.

---

## 🌱 From Bud to Bloom

When the project starts, the flowers appear as tiny buds.

There are no visible stems at first.

As the left hand opens, the stems gradually grow outward along their predefined curves.

At the same time, the right hand controls the flowers themselves.

The transition is continuous:

**bud → swelling bud → partial bloom → full bloom**

Everything is interpolated smoothly rather than switching between fixed states.

---

## ✨ Procedural by Design

The entire bouquet is drawn dynamically.

No flower PNGs.

No pre-rendered animation.

No fixed movement paths.

The visual system uses:

* Python
* OpenCV
* MediaPipe
* NumPy
* procedural geometry
* Bézier-style curves
* layered drawing
* real-time animation

The result is generated frame by frame.

---

## 🖐️ Interaction

| Gesture             | Interaction                           |
| ------------------- | ------------------------------------- |
| 🤏 Right-hand pinch | Flowers close into buds               |
| 🖐️ Right-hand open | Flowers bloom                         |
| 🤏 Left-hand pinch  | Stems become short                    |
| 🖐️ Left-hand open  | Stems grow                            |
| ✋ Movement          | Adds natural variation to the bouquet |

The two controls are intentionally independent.

One hand grows the bouquet.

The other lets it bloom.

---

## 🛠️ Tech Stack

**Core**

* Python
* OpenCV
* NumPy
* MediaPipe

**Computer Vision**

* Hand landmark detection
* Gesture recognition
* Real-time camera input

**Rendering**

* Procedural geometry
* Curved paths
* Layered shapes
* Real-time animation

---

## 📁 Project Structure

```text
FlowerBouquet/
│
├── main.py
├── flower.py
├── hand_tracker.py
├── utils.py
│
├── .gitignore
└── README.md
```

### `main.py`

Runs the application and connects the camera, hand tracking, controls, and rendering system.

### `flower.py`

Contains the procedural flower and bouquet rendering logic.

### `hand_tracker.py`

Handles MediaPipe hand tracking and extracts the required landmarks and gestures.

### `utils.py`

Contains supporting utility functions used throughout the project.

---

## 🚀 Running Locally

Clone the repository and move into the project directory:

```bash
git clone https://github.com/Akshaya-Sruti/FlowerBouquet.git
cd FlowerBouquet
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

Make sure your camera is available before starting the application.

---

## 🌿 Design Philosophy

The project is intentionally soft.

The movement is not perfectly synchronized.

The flowers are not perfectly identical.

The stems are not perfectly straight.

The ribbon is not perfectly symmetrical.

The small imperfections are part of the design.

Because sometimes the nicest things are the ones that don't look completely calculated.

---

## 🌸 What's Next

The project can eventually grow into a browser-based interactive experience with:

* Web-based procedural rendering
* Browser hand tracking
* Mobile support
* More flower varieties
* More natural hand interaction
* Interactive environments
* A small whimsical garden experience

For now, it is just a little bouquet.

Quietly growing.

---

<p align="center">

**made with love ♡**

*For the one who notices the little things.*

`soft launch · quietly growing`

</p>
