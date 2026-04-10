# 계층: 인프라 계층 (Core)
# 역할: JWT 토큰 생성/검증과 비밀번호 해싱을 한 곳에서 관리
#       개별 엔드포인트에서 보안 로직을 직접 구현하지 않고,
#       이 모듈을 DI로 주입받아 사용하는 것이 MVA 원칙이다.
# 의존: app.core.config (SECRET_KEY, ALGORITHM, TOKEN 만료 설정) 
# MVA 원칙: 보안 모듈 격리 - 보안 로직이 여러 파일에 흩어지는 것을 방지
"""
보안 모듈

JWT 토큰 생성/검증 및 비밀번호 해싱을 담당
"""

# UTC 시간 기반 토큰 만료 처리
from datetime import datetime, timedelta, timezone

# python-jose: JWT(JSON Web Token) 생성 및 검증 라이브러리
# JWTError: 토큰 디코딩 실패 시 발생하는 예외 (만료, 변조, 형식 오류 등)
from jose import JWTError, jwt
# passlib: 비밀번호 해싱 라이브러리 (bcrypt 알고리즘 사용)
# CryptContext: 해싱/검증 알고리즘을 관리하는 컨텍스트 객체
from passlib.context import CryptContext

# SCCRET_KEY, ALGORITHM 등 보안 설정
from app.core.config import settings

# --------------------------------------------------------------
# 비밀번호 해싱 컨텍스트
# --------------------------------------------------------------
# bcrypt: 업계 표준 단방향 해싱 알고리즘
# deprecated="auto": 향후 알고리즘 변경 시 기존 해시도 자동으로 재해싱
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """
    평문 비밀번호를 bcrypt로 해싱

    결과 예시: "$2b$12$LH3m4ys..."
    같은 비밀번호라도 매번 다른 해시값이 생성됨 (salt가 자동 포함)
    해싱된 값으로는 원본 비밀번호를 복원할 수 없음 (단방향)
    """
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    """
    평문 비밀번호와 해싱된 비밀번호를 비교 검증

    Args:
        plain: 사용자가 입력한 평문 비밀번호
        hashed: DB에 저장된 해싱됨 비밀번호

    Returns:
        True: 비밀번호 일치, False: 불일치
    """
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    JWT 액세스 토큰 생성.

    Args:
        data: 토큰에 담을 정보 (ex> {"sub": "user_id_123"})
        expires_delta: 만료 시간 (기본값: config의 ACCESS_TOKEN_EXPIRE_MINUTES)

    Returns:
        JWT 문자열 (ex> "eyJhbGciOiJIUzI1NiIs...")
        
    토큰 구조:
        Header: {"alg": "HS256", "typ": "JWT"}
        Payload: {"sub": "user_id"}
        Signature: HMAC-SHA256(header + payload, SECRET_KEY)
    """
    to_encode = data.copy()                 # 원본 dict를 변경하지 않기 위해 복사
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    to_encode.update({"exp": expire})       # 만료 시간을 payload에 추가
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def decode_access_token(token: str) -> dict | None:
    """
    JWT 토큰 디코딩 및 검증

    Args:
        token: 클라이언트가 보낸 JWT 문자열

    Returns:
        성공 시: payload dict (ex> {"sub": "user_id", "exp": ...})
        실패 시: None (만료, 변조, 형식 오류)

    실패 케이스:
        - 토큰 만료 (exp < 현재시간)
        - 서명 불일치 (SECRET_KEY가 다름 = 변조 의심)
        - 형식 오류 (JWT 구조가 아님)
    """
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return None                         # 예외를 None으로 변환하여 호출부에서 간결하게 처리 가능