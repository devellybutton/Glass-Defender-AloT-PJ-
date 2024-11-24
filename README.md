# Glass Defender

> 1. <b>[프로젝트 개요](https://github.com/devellybutton/Glass-Defender-AloT-Project?tab=readme-ov-file#1-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B0%9C%EC%9A%94)</b>
>    - 주제 및 선정 배경
>    - 외식업계의 무인화 가속화 현황
>    - 문제상황 및 기대효과
>    - 프로젝트 구조
>    - 활용 도구 및 장비
> 2. <b>[프로젝트 팀 구성 및 역할](https://github.com/devellybutton/Glass-Defender-AloT-Project?tab=readme-ov-file#2-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%ED%8C%80-%EA%B5%AC%EC%84%B1-%EB%B0%8F-%EC%97%AD%ED%95%A0)</b>
> 3. <b>[프로젝트 수행 과정](https://github.com/devellybutton/Glass-Defender-AloT-Project?tab=readme-ov-file#3-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%88%98%ED%96%89-%EA%B3%BC%EC%A0%95)</b>
> 4. <b>[프로젝트 수행 결과](https://github.com/devellybutton/Glass-Defender-AloT-Project?tab=readme-ov-file#4-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%88%98%ED%96%89-%EA%B2%B0%EA%B3%BC)</b>
>    - 개요
>    - 시연 영상
> 5. <b>[프로젝트 평가](https://github.com/devellybutton/Glass-Defender-AloT-Project?tab=readme-ov-file#5-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%ED%8F%89%EA%B0%80)</b>
>    - 한계점과 개선점
>    - 향후 계획 및 제언

------

# 1. 프로젝트 개요
## 1) 주제 및 선정 배경

- <b>주제</b> : 식당 내 태블릿PC의 카메라를 활용한 음주잔 및 음주병 낙하 방지
- <b>배경</b> : 식당에서 술병이 넘어져 깨지는 사고를 직접 목격한 경험을 계기로, 태블릿PC의 카메라를 활용해 음주잔과 음주병의 낙하를 방지하는 시스템을 개발하고자 이 주제를 선정함.

## 2) 외식업계의 무인화 가속화 현황

![무인화가속화흰색](https://github.com/user-attachments/assets/a7b7dc00-2c3f-4e99-af4d-9945d9720de8)

- 외식업계가 무인 서비스를 확대하게 된 주요 배경은 크게 두 가지가 있음.

  - 비대면 주문 수요 증가
  - 최저임금 상승으로 인한 노동비용 부담 <br><i>(2017년 6,470원 → 2023년 9,620원, 약 50% 상승)</i>

- 이로 인한 주요 변화와 현황:

  - 키오스크 설치 대수가 급증 <i>(2019년 5,500대 → 2022년 9만대)</i>
  - 다양한 무인화 기기 도입 <i>(태블릿 PC, 서빙로봇, 배달로봇, 무인 픽업 시스템 등)</i>
  - 기기별 월 렌탈비용이 인건비 대비 저렴 <i>(태블릿 PC 1만원대, 키오스크 10만원대, 서빙로봇 100만원 이하)</i>

- 성공 사례로는 테이블오더 스타트업 '메뉴잇'이 있으며, 연간 거래액이 2017년 3억원에서 2022년 4,780억원으로 급증하였음.
- 이러한 무인화는 <b>`매장 운영비용 절감`</b>과 <b>`고객 편의성 향상`</b>, 그리고 <b>`직원들의 업무 효율 개선`</b>에 기여할 것으로 전망됨.

## 3) 문제상황 및 기대효과
- <b>문제상황</b>
  - 술병과 잔이 미끄러워 쉽게 깨질 수 있음
  - 특히 음주 상태의 고객이 부주의로 인해 사고 유발 가능성 높음
- <b>해결방안</b> 
  - <b>`'Glass Defender' 시스템`</b>
    - 주문용 태블릿PC의 카메라로 위험 상황 감지
    - LED 경고등으로 고객에게 위험 상황 알림
- <b>기대효과</b>
  - 기술적 측면
    - Object Detection과 딥러닝으로 실시간 위험 감지
    - 데이터 수집을 통한 사고 예방 분석 가능
  - 실용적 측면
    - 기존 주문용 태블릿PC 활용으로 도입 비용 최소화
    - 안전사고 예방으로 인한 비용 절감
    - 전반적인 매장 안전도 향상

## 4) 프로젝트 구조

### 전체 프로젝트 구조

![기획구현그림001](https://github.com/user-attachments/assets/3a6657ad-fbfb-44c3-aed7-1d57da65b98b)

<br>

#### 🔶 **기획**

> - 주문용 태블릿PC에 장착된 **카메라**가 테이블을 촬영함. <br>
> → **촬영된 이미지**가 **서버**에 전송되어 깨질 위험이 있는 물체를 감지함. <br>
> → 해당 물체의 밑 무게 중심이 위험한 위치에 있으면 **LED**에 불이 들어옴.

1. 주문용 태블릿PC에 장착된 카메라로 테이블을 촬영한다.
2. 카메라로 캡쳐된 이미지는 서버로 전송된다.
3. 서버 내 프로그램은 위험 상황 여부를 판단하고, 결과를 LED 등으로 전송한다.
4. 유리 잔이나 병이 테이블 모서리를 넘어가려고 하는 위험 상황인 경우 LED 등이 켜진다.

<br>

#### 🔶 **구현 (시연)**

> - **라즈베리파이**와 연결된 **웹캠**이 테이블을 촬영하여 캡쳐된 이미지를 노트북으로 전송함. <br> 
> → **노트북**에서 위험 여부 판단함. <br>
> → **라즈베리파이**에 결과를 전달하여 위험 상황 시 **LED**를 켬.

1. 라즈베리파이와 연결된 웹캠이 테이블을 촬영한다.
2. 웹캠에서 캡쳐된 이미지는 스트림으로 라즈베리파이에 전송된다.
3. 라즈베리파이로 전송된 이미지는 REST API를 통해 연결된 서버용 pc (노트북 활용)으로 전달된다.
4. 노트북 내 프로그램은 위험여부를 판단하고, 결과(데이터)를 라즈베리파이에 전송한다.
5. 라즈베리파이는 전달받은 데이터에 따라 위험상황일 경우에 LED등을 켠다.

<br>

### 저장소 폴더 구조

  ```
  project_root/
  │
  ├── README.md                   # 한국어 버전의 프로젝트 설명
  ├── README_en.md                # 영어 버전의 프로젝트 설명
  ├── project_structure.png       # 프로젝트 구조 다이어그램
  │
  ├── image_crawling/             # 이미지 크롤링 관련 스크립트들
  │   ├── crawl_google.ipynb      # 구글 이미지 크롤링
  │   └── crawl_naver.ipynb       # 네이버 이미지 크롤링
  │
  ├── models/                     # 모델 관련 스크립트 및 파일
  │   ├── mobilenetv2_final.ipynb # MobileNetV2 모델 학습 스크립트
  │   ├── mobilenetv2_test.py     # 모델 테스트 스크립트
  │   ├── models_comparison.py    # 모델 비교 스크립트
  │   └── trained_model_final.pt  # 최종 학습된 모델 파일
  │
  ├── scripts/                    # 기타 유틸리티 스크립트들
    ├── desk_edge_save.py         # 엣지 저장 스크립트
    ├── edge_detection.py         # 경계선 감지 스크립트
    └── glass_defender.py         # 전체 통합 스크립트
  ```

## 5) 활용 도구 및 장비

- <b>개발환경</b> : PyCharm Community Edition 2023.2.3, Python 3.9.10, Google Colab, Raspbian
- <b>라이브러리</b> : Open CV, Numpy, Matplotlib, requests, PIL(Pillow), time, os, torchvision
- <b>프레임워크</b> : Pytorch, Flask
- <b>장비</b>
    - 서버용 pc <i>(노트북 활용)</i>
    - LED
    - 라즈베리파이
    - 웹캠
- <b>협업</b> : Notion, Google Docs, Google Presentation

------

# 2. 프로젝트 팀 구성 및 역할

| 권재현 | [이가린](https://github.com/devellybutton) |
|--------|--------|
| 컴퓨터 비전(cv2) 활용 | 데이터 수집 (크롤링, 촬영) |
| YOLOv5 모델 변환 | 데이터 정제 |
| YOLOv5 모델, MobileNetV2 동시 구동 | MobileNetV2 모델 전이학습 |
| 센서 연결 및 구동 | 학습 결과 평가
| 서버, 클라이언트 연결 및 구동 | 결과 정리 및 보고서 작성 |

------

# 3. 프로젝트 수행 과정

- 일자별 계획 및 진행한 내용

  <div style="width: 80%;">

  ![진행한내용최종](https://github.com/user-attachments/assets/cf288ea0-d9ad-41e2-b157-221999a1cd3a)

  </div>

------

# 4. 프로젝트 수행 결과
## 1) 개요

> #### 🔶 인공지능 모델 학습
> 데이터 수집(①) → 데이터 처리(②) → 모델 선정(③) → 모델 학습(④) → 모델 평가(⑤)

> #### 🔶 영상처리를 활용한 객체 탐지 및 LED 통합 제어
> OpenCV2로 책상 윤곽선과 RedLine 검출(⑥) <br>
> → MobileNetV2를 YOLOv5환경에서 실행(⑦) <br>
> → 서버-클라이언트 연결 및 LED 제어 구현(⑧) <br>

* 단계별 진행 내용은 각 폴더 README 문서에 기재해 두었음.

## 2) 시연 영상

[![Video Label](http://img.youtube.com/vi/sdjRR-CV2RM/0.jpg)](https://youtu.be/sdjRR-CV2RM)

-------

# 5. 프로젝트 평가

## 1) 한계점과 개선점

### 데이터셋 관련
- 크롤링으로 인한 데이터 부족 문제 발생
- 컵/맥주잔 등 다양한 형태의 객체에 대한 인식 정확도 부족 <br>
→ 직접 데이터 수집 및 다양한 형태의 데이터 확보 필요

### 실행 환경 최적화

- 실제 식당 환경(조명, 테이블 크기 등)에서의 시뮬레이션 필요
- 초상권 보호를 위한 깊이 카메라 도입 검토
- 사용자(업주/손님) 피드백 기반 알림 방식 개선 필요

### 프로젝트 제약

- 2주라는 짧은 개발 기간으로 인한 완성도 제한
- 다양한 라이브러리 사용에 대한 학습 시간 필요


## 2) 향후 계획 및 제언

- 현재는 기본적인 사고 예방 시스템을 구현한 단계
- 개선이 필요한 부분들:
  - 기술 혁신
  - 사용자 피드백 반영
  - 시장 동향 분석
  - 관련 법규 준수
- 이번 프로젝트를 통해 실제 구현 가능성을 확인했으며, 이는 식당 내 안전사고 예방을 위한 혁신적인 시작점이 될 것으로 기대됨.