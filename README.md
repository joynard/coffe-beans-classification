# Multi Model Coffee Bean Classifier

Computer vision project designed to classify coffee bean images into specific varietals using a selection of deep learning models. This repository contains the complete web application, inference pipeline, and pre-trained model weights. 

The application is used to deployed on Railway for real-world testing and evaluation (depricated since 7, July 2025)

---

## Project Overview

This system provides a web interface where users can upload images of coffee beans and dynamically select from multiple pre-trained deep learning architectures to compare their predictions. The classification pipeline is strictly trained to identify three primary taxonomic groups of coffee:

* **Arabica**: Highly valued for its complex acidity and sweet flavor profiles.
* **Robusta (Canephora)**: Known for its high caffeine content, strong body, and deep bitter notes.
* **Liberica**: A distinct, larger bean varietal with an asymmetric shape and unique smoky characteristics.

### Out-of-Distribution Handling
To ensure robustness against invalid inputs (such as non-coffee objects or damaged specimens), the system incorporates a confidence threshold filter. If an uploaded image does not match the features of the three target classes with high confidence, the system outputs **"Tidak Dikenali"** (Unrecognized) instead of forcing a false positive classification.

---

## Core Features

### Dynamic Model Selector
Users can swap the active inference engine on the fly. The project includes several pre-trained models packaged directly within the repository assets, eliminating the need to download heavy external weights during setup.

### Embedded Deep Learning Weights
All model architectures and trained weights are version-controlled and embedded directly inside the project structure. This guarantees immediate deployment compatibility and consistent local execution.

### Cloud Deployment
The system is hosted on Railway for a one-month testing period. This deployment serves as a public staging environment to gather real-world inference metrics and evaluate runtime performance.

---

## Technical Architecture

The application is built using a modern AI-web stack:
* **Frontend**: Responsive web interface optimized for both desktop and mobile image uploads.
* **Inference Engine**: PyTorch / TensorFlow runtime (depending on your specific framework) optimized for rapid CPU inference.
* **Deployment Platform**: Railway cloud infrastructure with automated Docker builds.

---

Production Deployment on Railway
This project is configured for automated builds on Railway. The deployment includes a custom Dockerfile or a start script mapping to Railway's dynamic port binding variables.

Environment Requirements
To maintain a high-performance deployment on Railway's micro-containers, the runtime environment is configured to run lightweight CPU-optimized libraries to keep memory utilization under the cloud provider's limits.

---

Developed by Alexander Fabiano Joynard Lapod | GitHub: https://github.com/joynard
