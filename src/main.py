# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv


def __main__():
    # API KEY 정보로드
    # True가 출력되어야 정상 
    load_dotenv()