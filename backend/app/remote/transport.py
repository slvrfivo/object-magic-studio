"""로컬 서버 ↔ 원격 GPU 워커 전송 계층.

- InProcessTransport: 같은 프로세스의 GpuWorker 를 직접 호출합니다 (테스트·개발용, GPU 불필요).
  인자와 결과를 pickle 로 한 번 왕복시켜 원격 호출처럼 직렬화 가능한 값만 오가는지 확인합니다.
- ModalTransport: Modal 에 배포된 클래스(remote_gpu/modal_app.py)를 Modal Python SDK 로 호출합니다.
  인증은 Modal 토큰(MODAL_TOKEN_ID / MODAL_TOKEN_SECRET 또는 ~/.modal.toml)으로 하며,
  공개 웹 엔드포인트를 만들지 않으므로 토큰이 없는 사람은 GPU 를 호출할 수 없습니다.

워커 메서드는 오류를 {"ok": False, "error": ...} 로 반환합니다. 전송 자체의 실패(네트워크, 인증, 배포 없음)는
RemoteTransportError 로 올립니다.
"""

from __future__ import annotations

import pickle
import threading
from collections.abc import Iterator
from typing import Any


class RemoteTransportError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class Transport:
    provider = "none"

    def call(self, method: str, **kwargs) -> dict:
        raise NotImplementedError

    def stream(self, method: str, **kwargs) -> Iterator[dict]:
        raise NotImplementedError

    def request_cancel(self, key: str) -> None:
        raise NotImplementedError

    def clear_cancel(self, key: str) -> None:
        raise NotImplementedError


class InProcessTransport(Transport):
    """GpuWorker(또는 같은 메서드를 가진 객체)를 같은 프로세스에서 호출."""

    provider = "inprocess"

    def __init__(self, worker, roundtrip_pickle: bool = True):
        self.worker = worker
        self.roundtrip = roundtrip_pickle
        self._cancel: set[str] = set()
        self._lock = threading.Lock()

    def _rt(self, value: Any) -> Any:
        return pickle.loads(pickle.dumps(value)) if self.roundtrip else value

    def call(self, method: str, **kwargs) -> dict:
        return self._rt(getattr(self.worker, method)(**self._rt(kwargs)))

    def stream(self, method: str, **kwargs) -> Iterator[dict]:
        kwargs = self._rt(kwargs)
        key = kwargs.pop("cancel_key", None)
        gen = getattr(self.worker, method)(**kwargs, should_cancel=lambda: self._is_cancelled(key))
        for ev in gen:
            yield self._rt(ev)

    def _is_cancelled(self, key: str | None) -> bool:
        with self._lock:
            return key is not None and key in self._cancel

    def request_cancel(self, key: str) -> None:
        with self._lock:
            self._cancel.add(key)

    def clear_cancel(self, key: str) -> None:
        with self._lock:
            self._cancel.discard(key)


class ModalTransport(Transport):
    """Modal 에 배포된 앱 호출. `modal` 패키지와 토큰이 필요합니다 (처음 호출할 때 확인)."""

    provider = "modal"

    def __init__(self, app_name: str, cls_name: str, cancel_dict: str):
        self.app_name = app_name
        self.cls_name = cls_name
        self.cancel_dict_name = cancel_dict
        self._obj = None
        self._dict = None
        self._lock = threading.Lock()

    def _modal(self):
        try:
            import modal  # noqa: PLC0415 — 원격 모드에서만 필요
        except ImportError as e:
            raise RemoteTransportError("remote_unavailable", "The 'modal' Python package is not installed on the server (pip install modal).") from e
        return modal

    def _instance(self):
        with self._lock:
            if self._obj is None:
                modal = self._modal()
                try:
                    cls = modal.Cls.from_name(self.app_name, self.cls_name)
                    self._obj = cls()
                    self._dict = modal.Dict.from_name(self.cancel_dict_name, create_if_missing=True)
                except Exception as e:  # noqa: BLE001
                    raise self._map(e) from e
            return self._obj

    @staticmethod
    def _map(e: Exception) -> RemoteTransportError:
        name = type(e).__name__
        text = str(e) or name
        if "token" in text.lower() or name in ("AuthError", "PermissionError"):
            return RemoteTransportError("remote_unavailable", f"Modal authentication failed: {text}. Run `modal token new` or set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET.")
        if name in ("NotFoundError",):
            return RemoteTransportError("remote_unavailable", f"The GPU app is not deployed on Modal ({text}). Deploy remote_gpu/modal_app.py first.")
        if name in ("ConnectionError", "TimeoutError", "ClientClosed", "OSError", "StreamTerminatedError", "gaierror", "ConnectionRefusedError", "ConnectionResetError") or isinstance(e.__cause__, OSError):
            # 예: DNS 조회 실패(gaierror) — 실제 T4 점검 중 한 번 발생 (일시적 네트워크 문제)
            return RemoteTransportError("remote_unavailable", f"Could not reach Modal: {text}")
        return RemoteTransportError("remote_failed", f"Remote GPU call failed: {name}: {text}")

    def call(self, method: str, **kwargs) -> dict:
        obj = self._instance()
        try:
            return getattr(obj, method).remote(**kwargs)
        except RemoteTransportError:
            raise
        except Exception as e:  # noqa: BLE001
            raise self._map(e) from e

    def stream(self, method: str, **kwargs) -> Iterator[dict]:
        obj = self._instance()
        try:
            yield from getattr(obj, method).remote_gen(**kwargs)
        except RemoteTransportError:
            raise
        except Exception as e:  # noqa: BLE001
            raise self._map(e) from e

    def request_cancel(self, key: str) -> None:
        self._instance()
        try:
            self._dict.put(key, True)
        except Exception as e:  # noqa: BLE001
            raise self._map(e) from e

    def clear_cancel(self, key: str) -> None:
        if self._dict is None:
            return
        try:
            self._dict.pop(key, None)
        except Exception:  # noqa: BLE001 — 정리 실패는 무시 (키는 작업마다 고유)
            pass
