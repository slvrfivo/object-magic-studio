"""원격 GPU 사용량 기록과 예산 확인 (로컬 JSON 파일).

서버리스 GPU 요금은 컨테이너가 켜져 있는 시간만큼 나옵니다:
  콜드 스타트(컨테이너 시작 + 모델 로드) + 실제 호출 시간 + 마지막 호출 뒤 대기 시간(scaledown window)
이 모듈은 로컬에서 측정한 호출 시간에 위 규칙을 적용해 **청구 시간을 보수적으로 추정**합니다.
실제 청구액은 Modal 대시보드가 기준이며, 이 값은 예산을 넘지 않도록 막는 안전장치입니다.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class BudgetExceeded(Exception):
    pass


class UsageLedger:
    def __init__(self, path: Path, *, cost_per_s: float, idle_s: float, cold_start_s: float, daily_limit_s: float, monthly_limit_usd: float, clock=time.time):
        self.path = Path(path)
        self.cost_per_s = cost_per_s
        self.idle_s = idle_s
        self.cold_start_s = cold_start_s
        self.daily_limit_s = daily_limit_s
        self.monthly_limit_usd = monthly_limit_usd
        self.clock = clock
        self._lock = threading.Lock()
        self._data = self._load()

    # ------------------------------------------------------------------ 저장
    def _load(self) -> dict:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(d, dict) and isinstance(d.get("days"), dict):
                return d
        except (OSError, ValueError):
            pass
        return {"days": {}, "warmUntil": 0.0, "lastEnd": 0.0}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def _day(ts: float) -> str:
        return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")

    def _bucket(self, ts: float) -> dict:
        return self._data["days"].setdefault(self._day(ts), {"billableS": 0.0, "calls": 0, "segments": 0, "tracks": 0, "trackedFrames": 0, "coldStarts": 0})

    # ------------------------------------------------------------------ 조회
    def totals(self) -> dict:
        now = self.clock()
        with self._lock:
            today = self._data["days"].get(self._day(now), {})
            month = self._day(now)[:7]
            month_s = sum(v.get("billableS", 0.0) for k, v in self._data["days"].items() if k.startswith(month))
            warm = self._data.get("warmUntil", 0.0) > now
        return {
            "todayBillableS": round(today.get("billableS", 0.0), 1),
            "todayCalls": today.get("calls", 0),
            "todayTrackedFrames": today.get("trackedFrames", 0),
            "monthBillableS": round(month_s, 1),
            "monthEstimatedUSD": round(month_s * self.cost_per_s, 4),
            "dailyLimitS": self.daily_limit_s,
            "monthlyLimitUSD": self.monthly_limit_usd,
            "costPerSecondUSD": self.cost_per_s,
            "idleWindowS": self.idle_s,
            "coldStartEstimateS": self.cold_start_s,
            "likelyWarm": warm,
        }

    def estimate_call_s(self, active_s: float) -> float:
        """이 호출에 대해 새로 청구될 시간 추정 (콜드 스타트 + 호출 + 대기)."""
        now = self.clock()
        warm = self._data.get("warmUntil", 0.0) > now
        return (0.0 if warm else self.cold_start_s) + active_s + self.idle_s

    def check(self, expected_active_s: float = 0.0) -> None:
        t = self.totals()
        extra = self.estimate_call_s(expected_active_s)
        if self.daily_limit_s > 0 and t["todayBillableS"] + extra > self.daily_limit_s:
            raise BudgetExceeded(
                f"Daily remote GPU limit reached ({t['todayBillableS']:.0f}s of {self.daily_limit_s:.0f}s used; this job needs about {extra:.0f}s). "
                "Use Local CPU, or raise OMS_REMOTE_DAILY_GPU_SECONDS."
            )
        if self.monthly_limit_usd > 0 and (t["monthBillableS"] + extra) * self.cost_per_s > self.monthly_limit_usd:
            raise BudgetExceeded(
                f"Monthly remote GPU budget reached (about ${t['monthEstimatedUSD']:.2f} of ${self.monthly_limit_usd:.2f}). "
                "Use Local CPU, or raise OMS_REMOTE_MONTHLY_USD."
            )

    # ------------------------------------------------------------------ 기록
    def record(self, kind: str, started: float, ended: float, *, tracked_frames: int = 0) -> dict:
        """호출 1회 기록. 반환: 이번 호출로 추가된 추정 청구 시간."""
        with self._lock:
            warm_until = self._data.get("warmUntil", 0.0)
            last_end = self._data.get("lastEnd", 0.0)
            added = 0.0
            cold = started >= warm_until
            if cold:
                added += self.cold_start_s
            elif last_end:
                # 직전 호출 뒤 대기 시간은 이미 idle_s 만큼 미리 더해 두었으므로, 실제 간격이 더 짧았으면 차이를 돌려받음
                gap = max(0.0, started - last_end)
                added -= max(0.0, self.idle_s - gap)
            # 겹치는 호출(동시 실행)은 컨테이너 1개를 같이 쓰므로 겹친 부분을 중복 청구하지 않음
            active_start = max(started, last_end) if not cold else started
            added += max(0.0, ended - active_start) + self.idle_s
            b = self._bucket(started)
            b["billableS"] = round(b["billableS"] + added, 3)
            b["calls"] += 1
            b["coldStarts"] += int(cold)
            if kind == "segment":
                b["segments"] += 1
            elif kind == "track":
                b["tracks"] += 1
                b["trackedFrames"] += tracked_frames
            self._data["lastEnd"] = max(last_end, ended)
            self._data["warmUntil"] = max(warm_until, ended + self.idle_s)
            self._save()
        return {"billableS": round(added, 2), "estimatedUSD": round(added * self.cost_per_s, 5), "coldStart": cold}
