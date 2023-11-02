# Glass_Defender_pj
**<h3>[AIot Project]</h3>** 

**Team name** : Glass Defender

**Team member** : jahyeon Kwon, Garin Lee

**Topic** :
- Using the camera of the tablet PC in the restaurant "to prevent glass and glass bottles from falling"
  
   (Modified **YOLOv5** and **MobileNetV2** with transfer learning, Utilizing **RaspberryPi**)

![project structure](project_structure.png)

**image_crawling** <br>

• crawling_google&naver(JupyterNotebook) : Crawling codes that collects images from Google and Naver <br>
  (cups, beer cups, soju cups) <br>
<br>

**models** <br>

• models_comparision (PyCharm) : Compare inference speeds and resource usage for three CNN models <br>
  (VGG-16, ResNet-50, MobileNetV2) <br>
  
• MobileNetV2 (Colab) : Transfer Learning Using MobileNetV2 <br>

• MobileNetV2_test (PyCharm) : Test the learning model to assess accuracy and loss <br>
<br>

**desk_edge_save** <br>

• Extracting table contours with OpenCV (cv2) <br>
<br>

**edge_detection** <br>

• Output to table contours and RedLine screens <br>
<br>

**glass_defender** <br>

• Real-time object detection (YOLOv5) and classification (MobileNetV2) are performed to control LEDs based on detection of fragile objects <br>
<br>

**request_exm** <br>

• Running a server <br>
