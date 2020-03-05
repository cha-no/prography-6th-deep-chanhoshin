프로그라피 6기 딥러닝 사전과제
==========================

사전 과제
1. tensorflow모듈을 이용해 vgg-16모델을 로드하고 cv2모듈을 이용해 mnist 데이터를 rgb채널로 변경했습니다.
2. my_prography.py 파일의 model_initialization, model_inference 함수로 구현했습니다. model_inference에서 한 층을 더 추가했습니다.
3. vgg-16모델의 Conv2_1의 입력을 첫번째 Dense 입력에 추가해주는 구조를 추가했습니다
4. 2,3번의 과제를 반영해 my_vgg 모델을 만들어 20번 학습했습니다.
5. 4번에서 학습한 model을 저장한 후 python test.py 파일을 실행하면 test 셋에 대해 정확도를 출력하도록 했습니다.
6. 정확도는 대략 0.951300입니다.

버전
* tensorflow 2.0.0
* keras 2.3.1
