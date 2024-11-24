# Glass Defender

> 1. <b>[Project Overview](https://github.com/devellybutton/Glass-Defender-AloT-Project?tab=readme-ov-file#1-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B0%9C%EC%9A%94)</b>
>    - Topic and Background
>    - Acceleration of Automation in the Food Service Industry
>    - Problem Situation and Expected Effects
>    - Project Structure
>    - Tools and Equipment Used
> 2. <b>[Project Team Composition and Roles](https://github.com/devellybutton/Glass-Defender-AloT-Project?tab=readme-ov-file#2-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%ED%8C%80-%EA%B5%AC%EC%84%B1-%EB%B0%8F-%EC%97%AD%ED%95%A0)</b>
> 3. <b>[Project Execution Process](https://github.com/devellybutton/Glass-Defender-AloT-Project?tab=readme-ov-file#3-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%88%98%ED%96%89-%EA%B3%BC%EC%A0%95)</b>
> 4. <b>[Project Results](https://github.com/devellybutton/Glass-Defender-AloT-Project?tab=readme-ov-file#4-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%88%98%ED%96%89-%EA%B2%B0%EA%B3%BC)</b>
>    - Overview
>    - Demo Video
> 5. <b>[Project Evaluation](https://github.com/devellybutton/Glass-Defender-AloT-Project?tab=readme-ov-file#5-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%ED%8F%89%EA%B0%80)</b>
>    - Limitations and Improvements
>    - Future Plans and Suggestions

------

# 1. Project Overview
## 1) Topic and Background

- <b>Topic</b> : Preventing the Falling of Drinking Glasses and Bottles Using a Tablet PC Camera in Restaurants
- <b>Background</b> : This topic was chosen based on a personal experience of witnessing a broken liquor bottle incident in a restaurant. The aim is to develop a system that uses the camera on a tablet PC to prevent the falling of drinking glasses and bottles.


## 2) Acceleration of Automation in the Food Service Industry

![무인화가속화흰색](https://github.com/user-attachments/assets/a7b7dc00-2c3f-4e99-af4d-9945d9720de8)

- There are two main reasons for the expansion of unmanned services in the foodservice industry:

  - Increased demand for contactless ordering
  - Rising labor costs due to the increase in minimum wage <br><i>(from 6,470 KRW in 2017 to 9,620 KRW in 2023, a rise of about 50%)</i>

- Key changes and current status due to this trend:

    - A rapid increase in the number of kiosk installations <i>(from 5,500 units in 2019 to 90,000 units in 2022)</i>
    - Introduction of various unmanned devices <i>(tablet PCs, serving robots, delivery robots, unmanned pickup systems, etc.)</i>
    - Monthly rental costs for devices are cheaper compared to labor costs <i>(tablet PCs around 10,000 KRW, kiosks around 100,000 KRW, serving robots under 1,000,000 KRW)</i>

- A successful case is the table order startup 'MenuIt', whose annual transaction volume grew rapidly from 300 million KRW in 2017 to 478 billion KRW in 2022.
- This trend of automation is expected to contribute to <b>reduction in operating costs</b>, <b>improvement in customer convenience</b>, and <b>enhancement of employee work efficiency</b>.

## 3) Problem Situation and Expected Effects
- <b>Problem Situation</b>
  - Liquor bottles and glasses are slippery and can easily break.
  - Customers, especially those who are intoxicated, are more likely to cause accidents due to carelessness.
- <b>Solution</b> 
  - <b>`'Glass Defender' system`</b>
    - Detects hazardous situations using the camera of the ordering tablet PC.
    - Alerts customers to potential dangers with LED warning lights.
- <b>Expected Effects</b>
  - Technical Aspects
    - Real-time hazard detection using Object Detection and deep learning.
    - Accident prevention analysis through data collection.
  - Practical Aspects
    - Minimization of implementation costs by utilizing existing ordering tablet PCs.
    - Cost savings from accident prevention.
    - Overall improvement in store safety.

## 4) Project Structure

### Overall Project Structure

![기획구현그림001](https://github.com/user-attachments/assets/3a6657ad-fbfb-44c3-aed7-1d57da65b98b)

<br>

#### 🔶 **Planning**

> - The <b>camera</b> mounted on the order tablet PC captures images of the table.  <br>
> →  <b>The captured images</b> are sent to the <b>server</b> where potential objects at risk of falling are detected. <br>
> → If the object's center of gravity is in a risky position, the <b>LED lights</b> up to alert users.

1. The camera mounted on the order tablet PC captures images of the table.
2. The captured image is sent to the server.
3. The server program determines if there is a risk and sends the results to the LED or other indicators.
4. If there is a danger of a glass or bottle tipping off the edge of the table, the LED lights are activated.

<br>

#### 🔶 Implementation (Demonstration)

> A <b>webcam</b> connected to a <b>Raspberry Pi</b> captures the table, and the captured images are sent to the laptop. <br> → The <b>laptop</b> evaluates the potential risk. <br> → The <b>Raspberry Pi</b> receives the result and activates the <b>LED</b> if a risky situation occurs.

1. The webcam connected to the Raspberry Pi captures the table.
2. The captured image is transmitted as a stream to the Raspberry Pi.
3. The image sent to the Raspberry Pi is forwarded to the server PC (laptop used) via a REST API.
4. The program on the laptop determines whether there is a risk, then sends the result (data) to the Raspberry Pi.
5. Based on the received data, the Raspberry Pi turns on the LED lights if a dangerous situation is detected.

<br>

### 저장소 폴더 구조

  ```
project_root/
│
├── README.md                   # Project description in Korean
├── README_en.md                # Project description in English
├── project_structure.png       # Project structure diagram
│
├── image_crawling/             # Scripts for image crawling
│   ├── crawl_google.ipynb      # Google image crawling script
│   └── crawl_naver.ipynb       # Naver image crawling script
│
├── models/                     # Scripts and files related to models
│   ├── mobilenetv2_final.ipynb # MobileNetV2 model training script
│   ├── mobilenetv2_test.py     # Model testing script
│   ├── models_comparison.py    # Model comparison script
│   └── trained_model_final.pt  # Final trained model file
│
├── scripts/                    # Other utility scripts
  ├── desk_edge_save.py         # Edge saving script
  ├── edge_detection.py         # Edge detection script
  └── glass_defender.py         # Full integration script
  ```

## 5) Tools and Equipment Used

- <b>Development Environment</b> : PyCharm Community Edition 2023.2.3, Python 3.9.10, Google Colab, Raspbian
- <b>Libraries</b> : Open CV, Numpy, Matplotlib, requests, PIL(Pillow), time, os, torchvision
- <b>Frameworks</b> : Pytorch, Flask
- <b>Equipment</b>
    - PC for server (used laptop)
    - LED lights
    - Raspberry Pi
    - Webcam
- <b>Collaboration</b> : Notion, Google Docs, Google Presentation

------

# 2. Project Team member and Roles

| Jaehyun Kwon | [Garin Lee](https://github.com/devellybutton) |
|--------|--------|
| Computer Vision (cv2) Usage       | Data Collection (Crawling, Capturing) |
| YOLOv5 Model Conversion               | Data Cleaning                            |
| YOLOv5 Model, MobileNetV2 Simultaneous Operation | MobileNetV2 Model Transfer Learning |
| Sensor Connection and Operation      | Model Evaluation                         |
| Server, Client Connection and Operation | Result Compilation and Report Writing    |

------

# 3. Project Execution Process

- Daily plan and progress

  <div style="width: 80%;">

  ![진행한내용최종](https://github.com/user-attachments/assets/cf288ea0-d9ad-41e2-b157-221999a1cd3a)

  </div>

------

# 4. Project Results
## 1) Overview

> #### 🔶 AI Model Training
> [Data Collection(①)](https://github.com/devellybutton/Glass-Defender-AloT-Project/blob/main/image_crawling/README.md#1-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%88%98%EC%A7%91) → [Data Processing(②)](https://github.com/devellybutton/Glass-Defender-AloT-Project/blob/main/image_crawling/README.md#2-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%B2%98%EB%A6%AC) → [Model Selection(③)](https://github.com/devellybutton/Glass-Defender-AloT-Project/blob/main/models/README.md#3-%EB%AA%A8%EB%8D%B8-%EC%84%A0%EC%A0%95) → [Model Training(④)](https://github.com/devellybutton/Glass-Defender-AloT-Project/blob/main/models/README.md#4-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5) → [Model Evaluation(⑤)](https://github.com/devellybutton/Glass-Defender-AloT-Project/blob/main/models/README.md#5-%EB%AA%A8%EB%8D%B8-%ED%8F%89%EA%B0%80)

> #### 🔶 Object Detection and LED Integrated Control Using Video Processing
> [Desk Outline and RedLine Detection with OpenCV2(⑥)](https://github.com/devellybutton/Glass-Defender-AloT-Project/blob/main/scripts/README.md#6-mobilenetv2%EC%99%80-yolov5-%ED%99%98%EA%B2%BD-%ED%86%B5%ED%95%A9) <br>
> → [Running MobileNetV2 in YOLOv5 Environment(⑦)](https://github.com/devellybutton/Glass-Defender-AloT-Project/blob/main/scripts/README.md#7-%ED%85%8C%EC%9D%B4%EB%B8%94-%EC%9C%A4%EA%B3%BD%EC%84%A0%EA%B3%BC-redline-%ED%99%94%EB%A9%B4-%EC%B6%9C%EB%A0%A5) <br>
> → [Server-Client Connection and LED Control Implementation(⑧)](https://github.com/devellybutton/Glass-Defender-AloT-Project/blob/main/scripts/README.md#8-%EC%84%9C%EB%B2%84%EB%85%B8%ED%8A%B8%EB%B6%81-%ED%81%B4%EB%9D%BC%EC%9D%B4%EC%96%B8%ED%8A%B8%EB%9D%BC%EC%A6%88%EB%B2%A0%EB%A6%AC%ED%8C%8C%EC%9D%B4-%EC%97%B0%EA%B2%B0-%EB%B0%8F-%EC%B5%9C%EC%A2%85-%EA%B5%AC%EB%8F%99) <br>

* The progress of each step is documented in the README file of each folder.

## 2) Demonstration Video
[![Video Label](http://img.youtube.com/vi/sdjRR-CV2RM/0.jpg)](https://youtu.be/sdjRR-CV2RM)

-------

# 5. Project Evaluation

## 1) Limitations and Areas for Improvement

### Dataset-related
- Data shortage issue due to web crawling.
- Insufficient recognition accuracy for various object shapes, such as cups and beer glasses. <br>
→ Direct data collection and acquisition of diverse data types are needed.

### Optimization of Execution Environment

- Simulation in actual restaurant environments (lighting, table sizes, etc.) is necessary.
- Consideration of implementing depth cameras for protecting privacy rights.
- Improvement of notification methods based on user (owner/customer) feedback is needed.

### Project Constraints

- Limited completeness due to the short development period of 2 weeks.
- Time required for learning how to use various libraries.


## 2) Future Plans and Recommendations

- Currently, the system has been implemented with basic accident prevention features.
    - Areas that need improvement:
    - Technological innovation
    - Incorporation of user feedback
    - Market trend analysis
    - Compliance with relevant regulations
- This project has confirmed the practical feasibility, and it is expected to be an innovative starting point for preventing safety accidents within restaurants.