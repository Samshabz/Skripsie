# Synthesis and Evaluation of an Image-Based Redundancy System for UAV Navigation

This project contains code to implement and evaluate an image-based redundancy system for UAV navigation. The core system leverages computer vision techniques to ensure reliable navigation in GPS-denied environments, with the primary functionality centered on detecting and matching image features for UAV positional inference.

## Project Overview

- **Title**: Synthesis and Evaluation of an Image-Based Redundancy System for UAV Navigation
- **Main Functionality**: Implements image-based methods using feature extractors and matchers to infer UAV position, allowing the system to navigate and return to known positions without GPS.

## Key Files and Modules

- The **`GoogleEarth`** folder contains all relevant design code. 
- **`NAVSYSTEM.py`**: The main program file, executing the core navigation algorithm and calling all necessary modules and dependencies.
  - Utilizes **Local_Matchers.py** for feature matching.
  - Applies **Feature_Extractors.py** for feature extraction.
  - Employs **shiftdetectors.py** for transformation inference.
  - Applies **translation_methods.py** for translation methods.

> **Note**: Some files are old or report-related and can be ignored. Only `NAVSYSTEM.py` and the files it references are necessary for running the system.

## Dependencies

The system relies on the following non-standard libraries:
- **SuperPoint** and **LightGlue** for feature detection and matching.
- **OpenCV** for general image processing.
- **NumPy** and **scikit-learn** for auxiliary mathematical functions.
- Other more common lib

## Running the Code

1. Clone this repository.
2. Install dependencies as listed in `NAVSYSTEM.py`.
3. Run `NAVSYSTEM.py` to initialize the navigation system and begin testing.

## License

[Property of Stellenbosch University, Lantern Engineering, And Sameer Shaboodien]

---

This README provides a high-level overview; refer to the code in `NAVSYSTEM.py` for detailed implementation logic.

