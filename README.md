# Out of Label-Hazards in Autonomous Driving

![fig4](https://github.com/user-attachments/assets/efddd282-a838-4d4a-8fb5-42228663b270)

This project processes videos to identify hazards and detect changes in driver state using object detection, image captioning, and LSTM-based analysis. The code implements a pipeline that integrates YOLO, BLIP, and additional techniques to process videos and output results in a structured format.

# Hazard Detection and Driver State Analysis in Autonomous Driving

## **Features**
1. **Object Detection**: Detect objects in video frames using a pre-trained YOLO model.
2. **Hazard Identification**: Identify the most probable hazard and generate captions for it using the BLIP model.
3. **Driver State Analysis**: Analyse driver behavior using LSTM and frame-based distance metrics to detect changes in driver state.
4. **Parallel Processing**: Utilise multi-threading for efficient processing of multiple videos simultaneously.
5. **Results Logging**: Outputs structured results to a CSV file for further analysis.


## **Prerequisites**
The following libraries and tools are required to run the code:
- Python 3.x
- NumPy
- OpenCV
- PyTorch
- Transformers
- Ultralyitcs YOLO
- PIL (Pillow)

### **Dataset and Annotations**
1. The videos are located at `/kaggle/input/coool-benchmark/COOOL Benchmark`.
2. The annotations file (`annotations_public.pkl`) must be in the directory `/kaggle/input/annotations-public-pkl/`.


## **Code Workflow**
1. **Initialization**:
   - Load the video files and annotations.
   - Initialize models for object detection (YOLO) and image captioning (BLIP).
   - Prepare the output file (`results.csv`) for logging results.

2. **Video Processing**:
   - For each video:
     - Open and read frames using OpenCV.
     - Use YOLO to detect objects and extract bounding boxes and labels.
     - Track object centroids to analyze movement patterns and detect potential hazards.
     - Identify and caption probable hazards using BLIP.
     - Use LSTM to analyze driver state changes based on distance metrics.

3. **Driver State Detection**:
   - Analyze frame-by-frame centroid distances.
   - Use a rolling window of 5 frames to generate input sequences for an LSTM model.
   - Predict slowing or anomalous driver behavior using an LSTM threshold.
<img width="866" alt="fig1" src="https://github.com/user-attachments/assets/32eb2c3e-186c-4f2c-bd6e-9b567b86c313" />

4. **Results**:
   - Log the following in `results.csv`:
     - Frame ID
     - Driver state change flag
     - Probable hazard track ID
     - Hazard description
     - Placeholder fields for additional hazard tracks and descriptions.

5. **Parallel Execution**:
   - Videos are processed in parallel using a thread pool for efficiency.

---

## Output
The results are saved to a CSV file named `results.csv`, with the following format:

```
ID, Driver_State_Changed, Hazard_Track_0, Hazard_Name_0, ..., Hazard_Track_22, Hazard_Name_22
```

---

## How to Run
1. Ensure the required paths (`VIDEO_ROOT` and `ANNOTATION_PATH`) are set correctly.
2. Place the pre-trained YOLO and BLIP models in the specified locations.
3. Run the script in an environment with all dependencies installed:
   ```
   python process_videos.py
   ```
4. The results will be saved in the `results.csv` file.


## Limitations
- The LSTM model (`lstm_model`) for driver state detection must be pre-trained and loaded. The script assumes its existence but does not include the model training code.
- Object detection is performed on a CPU for compatibility, but using a GPU is recommended for better performance.
- BLIP captioning uses a CPU, which may slow down processing for large-scale datasets.


## Future Improvements
- Integrate GPU support for YOLO and BLIP to accelerate processing.
- Add training and fine-tuning scripts for the LSTM driver state model.
- Enhance hazard detection logic by incorporating temporal tracking of objects.
- Expand the dataset to include more scenarios and edge cases.


## **Acknowledgments**
This project uses the following models and frameworks:
- [YOLO](https://github.com/ultralytics/ultralytics): Pre-trained object detection.
- [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base): Image captioning.

