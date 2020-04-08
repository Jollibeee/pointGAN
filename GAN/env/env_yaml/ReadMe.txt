

가상환경 저장

 conda env export > environment.yaml



yaml 파일을 사용하여 가상환경 만들기

 conda env create -f environment.yaml




가상환경에서 패키지 목록 만들기

 pip freeze > requirements.txt


패키지 목록으로 부터 패키지 설치하기

 pip install -r requirements.txt
