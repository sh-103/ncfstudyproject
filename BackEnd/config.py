import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Settings:
    def __init__(self):
        self.DATABASE_URL = self._get_env("DATABASE_URL")
        self.MODEL_PATH = self._get_env("MODEL_PATH", "./default_model.pkl")
        self.SECRET_KEY = self._get_env("SECRET_KEY")

    def _get_env(self, key: str, default=None):
        value = os.getenv(key, default)
        if value is None:
            # 서버 구동 시 필수값이 없으면 즉시 에러 발생 (Fail-Safe)
            raise ValueError(f"환경 변수 {key}가 누락되었습니다.")
        return value

settings = Settings()