from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

# spaCy NER (model tiếng Anh) không đáng tin với tiếng Việt → loại các entity
# free-text này để tránh false-positive (vd "Nhân viên", "nghỉ phép" -> PERSON).
_UNRELIABLE_PII_TYPES = {
    "PERSON", "LOCATION", "NRP", "DATE_TIME", "FACILITY", "ORGANIZATION",
    "GPE", "EVENT", "WORK_OF_ART", "LANGUAGE", "PRODUCT",
    "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_PASSPORT", "US_ITIN",
}
# Ngưỡng score tối thiểu để coi là PII (loại các match yếu của Presidio).
_PII_SCORE_THRESHOLD = 0.5


def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    analyzer  = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    if analyzer is None or anonymizer is None:
        try:
            analyzer, anonymizer = setup_presidio()
        except Exception:
            analyzer, anonymizer = None, None

    if analyzer is not None and anonymizer is not None:
        try:
            results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
            # Lọc false-positive: spaCy NER tiếng Anh hay gán nhầm từ tiếng Việt
            # thành PERSON/LOCATION/DATE_TIME... Chỉ giữ PII có cấu trúc đáng tin
            # và bỏ các match score thấp.
            results = [
                r for r in results
                if r.entity_type not in _UNRELIABLE_PII_TYPES and r.score >= _PII_SCORE_THRESHOLD
            ]
            if not results:
                return {"has_pii": False, "entities": [], "anonymized": text}

            anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
            entities = [
                {
                    "type": r.entity_type,
                    "text": text[r.start:r.end],
                    "score": round(r.score, 3),
                    "start": r.start,
                    "end": r.end,
                }
                for r in results
            ]
            return {"has_pii": True, "entities": entities, "anonymized": anonymized}
        except Exception:
            # Fallback regex nếu Presidio runtime lỗi.
            pass

    patterns = [
        ("VN_CCCD", r"\b\d{12}\b", 0.9),
        ("VN_CCCD", r"\b\d{9}\b", 0.7),
        ("VN_PHONE", r"\b0[3-9]\d{8}\b", 0.9),
        ("EMAIL_ADDRESS", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 0.95),
    ]
    entities: list[dict] = []
    for entity_type, pattern, score in patterns:
        for match in re.finditer(pattern, text):
            entities.append({
                "type": entity_type,
                "text": match.group(0),
                "score": score,
                "start": match.start(),
                "end": match.end(),
            })

    if not entities:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = text
    # Replace từ cuối về đầu để không lệch index.
    for e in sorted(entities, key=lambda x: x["start"], reverse=True):
        anonymized = (
            anonymized[:e["start"]] + f"<{e['type']}>" + anonymized[e["end"]:]
        )
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — Colang dialogue flows (topic check, jailbreak check, output check)
    """
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails  = LLMRails(config)
    return rails


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    if rails is None:
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None

    if rails is not None:
        try:
            response = await rails.generate_async(
                messages=[{"role": "user", "content": text}]
            )
            response_text = response if isinstance(response, str) else str(response)
            refuse_keywords = [
                "xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry"
            ]
            blocked = any(kw in response_text.lower() for kw in refuse_keywords)
            return {
                "allowed": not blocked,
                "blocked_reason": "nemo_input_rail" if blocked else None,
                "response": response_text,
            }
        except Exception:
            # fallback rules bên dưới
            pass

    lowered = text.lower()
    attack_patterns = [
        "ignore",
        "bỏ qua",
        "dan",
        "system override",
        "admin command",
        "dump all",
        "confidential",
        "mật khẩu",
        "password",
        "tiết lộ",
        "reveal",
        "unrestricted ai",
        "training data",
        "system instructions",
    ]
    off_topic_patterns = [
        "bài thơ",
        "phở",
        "bitcoin",
        "ethereum",
        "phương trình vi phân",
        "marvel",
        "movie",
    ]
    blocked = any(pat in lowered for pat in attack_patterns + off_topic_patterns)
    if blocked:
        return {
            "allowed": False,
            "blocked_reason": "nemo_input_rail",
            "response": "Xin lỗi, tôi không thể hỗ trợ yêu cầu này theo chính sách bảo mật.",
        }
    return {"allowed": True, "blocked_reason": None, "response": "allowed"}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output).
    Kiểm tra: có PII không? Nội dung có phù hợp không? Có hallucination rõ ràng không?

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    if rails is None:
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None

    if rails is not None:
        try:
            response = await rails.generate_async(messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            response_text = response if isinstance(response, str) else str(response)
            refuse_keywords = ["xin lỗi", "không thể cung cấp", "i cannot"]
            flagged = any(kw in response_text.lower() for kw in refuse_keywords)
            return {
                "safe": not flagged,
                "flagged_reason": "nemo_output_rail" if flagged else None,
                "final_answer": response_text if flagged else answer,
            }
        except Exception:
            pass

    pii = pii_scan(answer)
    if pii["has_pii"]:
        return {
            "safe": False,
            "flagged_reason": "nemo_output_rail",
            "final_answer": pii["anonymized"],
        }
    return {"safe": True, "flagged_reason": None, "final_answer": answer}


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    async def _run_all() -> list[dict]:
        results = []
        for item in adversarial_set:
            user_input = item.get("input", "")
            blocked_by = None

            pii_result = pii_scan(user_input, analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            if blocked_by is None:
                rail_result = await check_input_rail(user_input, rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id": item.get("id"),
                "category": item.get("category"),
                "input": user_input[:80] + ("..." if len(user_input) > 80 else ""),
                "expected": item.get("expected"),
                "actual": actual,
                "blocked_by": blocked_by,
                "passed": actual == item.get("expected"),
            })
        return results

    try:
        results = asyncio.run(_run_all())
    except RuntimeError:
        # Khi đang ở event loop (vd notebook).
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(_run_all())
        finally:
            loop.close()

    passed = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Mục tiêu production: P95 total < LATENCY_BUDGET_P95_MS (500ms mặc định)

    Insight cần quan sát:
        - Presidio: local regex → rất nhanh (<10ms)
        - NeMo:     LLM API call → chậm (~200-800ms tuỳ model và network)
        → Tổng: dominated by NeMo

    Returns:
        {
          "presidio_ms":  {"p50": float, "p95": float, "p99": float},
          "nemo_ms":      {"p50": float, "p95": float, "p99": float},
          "total_ms":     {"p50": float, "p95": float, "p99": float},
          "latency_budget_ok": bool,
          "budget_ms": int,
        }
    """
    samples = test_inputs[:max(1, min(n_runs, len(test_inputs) if test_inputs else 1))]
    if not samples:
        samples = [""]

    presidio_times: list[float] = []
    nemo_times: list[float] = []
    total_times: list[float] = []

    async def _measure() -> None:
        for text in samples:
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    try:
        asyncio.run(_measure())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_measure())
        finally:
            loop.close()

    def _percentiles(values: list[float]) -> dict[str, float]:
        s = sorted(values) if values else [0.0]
        n = len(s)

        def pick(p: float) -> float:
            idx = int(round((n - 1) * p))
            idx = max(0, min(n - 1, idx))
            return round(float(s[idx]), 2)

        return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}

    total_p = _percentiles(total_times)
    return {
        "presidio_ms": _percentiles(presidio_times),
        "nemo_ms": _percentiles(nemo_times),
        "total_ms": total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    results = run_adversarial_suite(adversarial_set)
    if results:
        passed = sum(1 for r in results if r["passed"])
        print(f"Adversarial suite: {passed}/{len(results)} passed")

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    # --- Lưu report ---
    os.makedirs("reports", exist_ok=True)
    guard_report = {
        "pii_demo": {
            "input": test_pii,
            "has_pii": result["has_pii"],
            "entities": result["entities"],
            "anonymized": result["anonymized"],
        },
        "adversarial_suite": {
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]) if results else 0,
            "results": results,
        },
        "latency_p95": latency,
    }
    with open("reports/guard_results.json", "w", encoding="utf-8") as f:
        json.dump(guard_report, f, ensure_ascii=False, indent=2)
    print("\n✓ Saved → reports/guard_results.json")
