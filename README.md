**Theoretical analysis**
**Explain how Edge AI reduces latency and enhances privacy compared to cloud-based AI. Provide a real-world example.**
Edge AI refers to running AI models directly on local devices (edge devices) such as smartphones, IoT sensors, autonomous drones, or security cameras—rather than sending data to the cloud for processing.

**How Edge AI reduces latency**
Local processing: Because computations happen on the device itself, data does not need to travel over the internet to a remote server.
Faster response times: This eliminates network delays and results in near-instant decisions.
Ideal for real-time applications: Critical systems (e.g., drones, robots, self-driving cars) can make split-second decisions.

**Real-world example: Autonomous drones**
Autonomous drones use Edge AI to analyze camera feeds, detect obstacles, and navigate in real time. ie A drone surveying a disaster zone processes video on-board.
It avoids hazards instantly without needing to send video to the cloud.

**Compare Quantum AI and classical AI in solving optimization problems. What industries could benefit most from Quantum AI?**
Classical AI uses traditional computing and struggles with very large optimization problems.
Quantum AI uses qubits that can evaluate many possibilities simultaneously, making it far more powerful for complex optimization tasks.
**Industries that benefit most:**
Logistics, finance, pharmaceuticals, energy, manufacturing, and telecommunications — where large-scale optimization is crucial.
**
PART 2
****
 Practical Implementation
 
**TASK 1: Edge AI Prototype ****

This project develops a lightweight image classification model, converts it to TensorFlow Lite, and tests it for deployment on an edge device such as a Raspberry Pi. The model classifies objects (e.g., recyclable materials), allowing real-time AI processing without cloud dependency.

** Model Development**

Dataset: TF Flowers (placeholder for recycling dataset).
Model: Small CNN with 16–64 filters for edge efficiency.
Training: 5 epochs, Adam optimizer.
Accuracy: ~70% on test data.
Output: edge_model.tflite.

**Benefits of Edge AI
**
Low latency	Decisions- happens instantly on-device.
No internet required- it Works offline—ideal for remote locations.
Lower cloud cost- Processing stays local.
Better privacy- Images never leave the device.

- Edge AI improves speed, privacy, and reliability in applications like recycling detection, smart surveillance, or environmental monitoring.

**TASK 2:AI-Driven IoT Concept**

**1) Sensors & On‑field Hardware (list + purpose)**

Soil moisture sensors (volumetric water content)
Purpose: monitor water availability at root zone

Soil temperature sensors
Purpose: germination and microbial activity indicator

Air temperature & humidity (DHT/BME series)
Purpose: microclimate, evapotranspiration calculation

Leaf wetness / dew sensors
Purpose: disease risk (fungal infections)

Light / PAR sensor (Photosynthetically Active Radiation)
Purpose: sunlight availability for photosynthesis

pH / EC (electrical conductivity) sensors
Purpose: soil fertility and salinity monitoring

**2) Proposed AI Model(s) to Predict Crop Yields**

Hybrid deep model (cloud training)
-SHAP values on tabular features; Grad-CAM on images

Lightweight production model (edge-friendly)
-Feature engineering (seasonal aggregates) + XGBoost/LightGBM
-Why XGBoost: fast inference, small memory, good performance on tabular data; easy to export (ONNX, Core ML, TFLite via conversion if needed)

Probabilistic / Uncertainty-aware model
-Use quantile regression or Bayesian Neural Networks to provide prediction intervals (important for farm decision-making).

**3) Data Flow Diagram**

[Sensors] --(LoRa/WiFi)--> [Edge Gateway] --(MQTT)--> [Message Broker]
      |                                           |
      |                                           v
      |                                      [Stream Processor]
      |                                           |
      v                                           v
[Images/Drone] --> [Edge/Drone] --> (upload) --> [Object Storage (images)]

[Message Broker] -> [Time-Series DB (InfluxDB)]

[Batch ETL] <- [Time-Series DB + Object Storage] -> [Feature Store]

[Feature Store] -> [Model Training (Cloud)] -> [Trained Model + Export]

[Trained Model] -> [Model Registry]

[Model Registry] -> [Edge Inference (Gateway/RPi)]  OR  -> [Cloud Inference API]

[Dashboard/Alerts] <- [Predictions]  (user notifications, irrigation control)




