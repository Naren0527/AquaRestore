# AquaRestore 🌊

### Underwater Image Enhancement, Quality Assessment & Visual Analysis

AquaRestore is a Python-based underwater image enhancement project designed to improve the visual quality of underwater photographs through image processing, quality diagnostics, and comparative analysis.

Underwater images often suffer from color distortion, reduced contrast, haziness, and loss of fine details due to light absorption and scattering. AquaRestore provides an interactive platform for enhancing images, inspecting quality metrics, and evaluating processing results through a user-friendly interface.

The project combines image enhancement with diagnostic tools and experimental analysis to support systematic evaluation of underwater image restoration techniques.

---

## ✨ Key Features

### 🖼️ Interactive Image Enhancement

* Upload and process underwater images through a Streamlit interface.
* View original and enhanced images for visual comparison.
* Inspect image quality metrics alongside processing results.
* Export processed images and available evaluation results.

### 🔬 Image Diagnostics

* Bilateral-filter-based image denoising.
* Basic image quality assessment.
* Diagnostic utilities for examining image characteristics.
* Structured outputs for downstream analysis.

### 📊 Experimental Evaluation

* Run enhancement experiments on collections of input images.
* Compare original images with processed outputs.
* Record available quality metrics and processing results.
* Export evaluation results to CSV for further analysis.

### 📈 Visualization & Analysis

* Generate quality comparison plots.
* Compare processing runtimes.
* Visualize paired differences between enhancement methods.
* Generate contact sheets for convenient visual inspection.
* Summarize experimental results using analysis utilities.

### 🧪 Automated Testing

* Unit tests for diagnostics and evaluation functionality.
* Tests for result analysis and visualization utilities.
* Regression checks to help maintain reliable project behavior.

---

## 🛠️ Technology Stack

| Technology | Purpose                        |
| ---------- | ------------------------------ |
| Python     | Core implementation            |
| OpenCV     | Image processing and denoising |
| NumPy      | Numerical operations           |
| Pillow     | Image handling                 |
| Matplotlib | Visualization and plotting     |
| Streamlit  | Interactive web interface      |
| Pytest     | Automated testing              |

---

## 🏗️ Project Structure

```text
AquaRestore/
│
├── app_new.py
├── requirements.txt
│
├── aquarestore/
│   ├── contracts.py
│   │
│   ├── diagnostics/
│   │   ├── denoising.py
│   │   └── quality.py
│   │
│   ├── evaluation/
│   │   ├── experiments.py
│   │   ├── analyze_results.py
│   │   └── visualize_results.py
│   │
│   ├── visualization/
│   │   └── plots.py
│   │
│   └── enhancement/
│       └── pipeline.py
│
└── tests/
    ├── test_analysis.py
    ├── test_diagnostics.py
    ├── test_evaluation.py
    ├── test_plots.py
    └── test_visualization.py
```

*Note: This is a high-level overview of the main project files, not an exhaustive listing of every repository file.*

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or later recommended.
* Git.
* A terminal or command prompt.

### 1. Clone the repository

```bash
git clone https://github.com/Naren0527/AquaRestore.git
cd AquaRestore
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the application

```bash
python -m streamlit run app_new.py
```

To launch the application on a different port:

```bash
python -m streamlit run app_new.py --server.port 8600
```

Streamlit will provide a local URL that you can open in your browser.

---

## 📷 Using AquaRestore

1. Launch the Streamlit application.
2. Upload an underwater image using the interface.
3. Run the available enhancement pipeline.
4. Compare the original image with the enhanced result.
5. Inspect the available quality metrics and diagnostic information.
6. Export the supported outputs for further use.

The available controls and metrics depend on the implemented enhancement and diagnostic functionality.

---

## 📊 Evaluation Workflow

AquaRestore includes utilities for conducting repeatable image-processing experiments and analyzing their results.

The evaluation workflow supports:

* Processing collections of images.
* Comparing original and enhanced outputs.
* Recording available quality metrics.
* Saving experiment results in CSV format.
* Generating contact sheets and comparison plots.
* Summarizing paired experimental results.

Generated files are intended for local analysis and can be used to compare enhancement behavior across a dataset.

**Note:** Evaluation results depend on the input dataset, image-processing configuration, and available metrics. Visual improvement should not automatically be interpreted as proof of superior perceptual quality.

---

## 🧪 Running the Tests

From the project root, activate your virtual environment and run:

```bash
python -m pytest -q
```

The full automated test suite has previously completed successfully with **86 passing tests**.

Test results may vary if the code, dependencies, or environment change.

---

## 🎯 Project Objectives

AquaRestore aims to:

* Improve the visual appearance of underwater images.
* Provide accessible image-processing tools through an interactive interface.
* Integrate image diagnostics with enhancement workflows.
* Support repeatable experiments and structured result analysis.
* Make image comparisons easier through visualizations and exported results.
* Establish a modular foundation for future underwater image restoration research.

---

## 🔭 Future Scope

Potential directions for further development include:

* Integration of additional underwater image quality metrics.
* More comprehensive comparisons across enhancement algorithms.
* Expanded evaluation datasets and benchmarking.
* Improved robustness across different underwater environments.
* Additional visualization and reporting capabilities.
* Further refinement of the interactive user experience.

These are potential extensions rather than claims about currently implemented features.

---

## 👥 Contributors

AquaRestore is developed collaboratively, with responsibilities divided across enhancement pipeline development, diagnostics, evaluation, visualization, and interface development.

Contributions and improvements are welcome through issues and pull requests.

---

## 📄 License

Please refer to the repository's license information. If no license has been added, the project's reuse and redistribution terms should be established before presenting it as an open-source project.

---

<p align="center">
  <b>AquaRestore — Making underwater image enhancement easier to explore, evaluate, and understand.</b>
</p>
