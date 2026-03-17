# 🚀 VisionMeasure - Object Detection & Dimension Estimation System

VisionMeasure is a computer vision-based system that detects objects in an image and calculates their real-world dimensions with high accuracy. It combines image processing, contour detection, and geometric analysis to measure object width and height dynamically.

---

## 🧠 Project Overview

Measuring objects manually can be inefficient and error-prone. VisionMeasure automates this process by:

* Detecting objects in an image
* Identifying boundaries using contours
* Calculating width and height of each object
* Displaying dimensions directly on the screen

This system is useful for applications where **automated measurement and inspection** are required.

---

## ⚙️ Key Features

* 📦 **Automatic Object Detection**
* 📏 **Real-Time Dimension Calculation (Width & Height)**
* 🎯 **Detects Multiple Objects in a Single Image**
* 🧠 **Contour-Based Detection (No Hardcoding)**
* ⚡ **Fast Image Processing Pipeline**
* 🖥️ **Visual Output with Bounding Boxes & Measurements**
* 🔄 **Scalable for Real-Time Video Processing**

---

## 🏗️ Tech Stack

* **Language:** C++
* **Framework:** Qt (for UI display)
* **Image Processing:** OpenCV
* **Detection Technique:** Contour Detection & Bounding Rectangles
* **Mathematics:** Pixel-to-real-world scaling

---

## 🔄 Workflow

1. Input image or video frame
2. Preprocess image (grayscale, blur, edge detection)
3. Detect contours (object boundaries)
4. Filter valid objects (remove noise)
5. Draw bounding boxes around objects
6. Calculate width & height in pixels
7. Convert pixels to real-world dimensions using scaling
8. Display dimensions on screen

---

## 📐 Dimension Calculation Logic

The system calculates object size using pixel measurements and converts them into real-world units.

### Steps:

* Detect object boundary using contours
* Compute bounding rectangle
* Extract:

  * Width (pixels)
  * Height (pixels)
* Apply scaling factor:

```id="dimcalc1"
Real Width  = Pixel Width  × Scale Factor  
Real Height = Pixel Height × Scale Factor
```

### Notes:

* Scale factor is derived using a reference object or calibration
* Ensures accurate real-world measurement

---

## 📂 Project Structure

```plaintext id="struc1"
VisionMeasure/
│
├── src/
│   ├── main.cpp
│   ├── imageprocessor.cpp
│   ├── detection.cpp
│
├── include/
│   ├── imageprocessor.h
│   ├── detection.h
│
├── assets/
│   ├── test_images/
│
├── output/
│   ├── processed_images/
│
└── README.md
```

---

## 🚀 How to Run

### Prerequisites

* Install OpenCV
* Install Qt Creator

### Steps

```bash id="run1"
# Clone the repository
git clone https://github.com/cmohan312002/Object-Detection_with-Dimensions.git

# Open in Qt Creator
# Build and run the project
```

---

## 🎯 Use Cases

* 📦 Industrial Object Measurement
* 🏭 Quality Inspection Systems
* 📏 Automated Size Detection
* 📸 Image-Based Measurement Tools
* 🤖 Computer Vision Applications

---

## 💡 What Makes This Project Special?

Unlike basic object detection systems, VisionMeasure:

* Focuses on **dimension estimation**, not just detection
* Uses **geometric analysis for real-world measurements**
* Supports **multiple object detection in one frame**
* Provides **visual and measurable output simultaneously**

---

## 🧪 Future Enhancements

* 🔥 Deep Learning-based object detection (YOLO / SSD)
* 🎥 Real-time video stream processing
* 📱 Mobile app integration
* 📊 Measurement analytics dashboard
* 📐 Automatic calibration system

---

## 👨‍💻 Author

**Mohan Chilivery**
Programmer

---

## ⭐ Support

If you found this project useful:

* Star ⭐ the repo
* Share it with others
* Contribute to improve it

---

## 📜 License

This project is open-source and available under the MIT License.
