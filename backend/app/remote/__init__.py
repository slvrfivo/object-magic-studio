"""원격 GPU 추론 (Phase 5B-2).

- worker.py   : GPU 컨테이너 안에서 실행되는 워커 (기존 Sam2Engine / Sam2ImageService / VideoTracker 재사용)
- transport.py: 로컬 서버 ↔ 워커 통신 (InProcess = 테스트·개발용, Modal = 서버리스 GPU)
- service.py  : 로컬 FastAPI 쪽 클라이언트 (영상 업로드, 분할, 추적 작업, 취소, 사용량·비용 한도)
- usage.py    : 원격 GPU 사용량 기록과 예산 확인

모든 원격 호출의 인자와 결과는 기본 자료형(dict / list / bytes / str / int / float)만 사용합니다.
"""
