# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** *Nguyễn Trần Mạnh Thắng*  
**Ngày:** 30/06/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~?ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~?ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---



## Latency Budget

*(Kết quả thực tế từ Task 12 — measure_p95_latency(), chạy CPU 8GB RAM)*


| Layer            | P50 (ms)                     | P95 (ms)     | P99 (ms) | Budget     |
| ---------------- | ---------------------------- | ------------ | -------- | ---------- |
| Presidio PII     | 6633.91                      | 19714.85     | 19714.85 | <10ms      |
| NeMo Input Rail  | 2.23                         | 17.06        | 17.06    | <300ms     |
| RAG Pipeline     | (không đo trong harness này) | -            | -        | <2000ms    |
| NeMo Output Rail | (gộp trong NeMo)             | -            | -        | <300ms     |
| **Total Guard**  | 6635.14                      | **19722.75** | 19722.75 | **<500ms** |


**Budget OK?** [ ] Yes / [x] No  
**Comment:** Bottleneck là **Presidio PII** do spaCy `en_core_web_lg` chạy trên CPU (máy 8GB RAM), p95 ~~19.7s vượt xa budget. Cách tối ưu: (1) thay model NER nhỏ hơn (~~`en_core_web_sm`~~) hoặc tắt NER engine và chỉ dùng pattern recognizers cho VN_CCCD/VN_PHONE/EMAIL; (2) chạy Presidio trên GPU hoặc batch hóa; (3) cache kết quả. NeMo Input Rail đạt budget tốt (~~17ms p95).

---



## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---



## Monitoring Dashboard (production)


| Metric                            | Alert Threshold | Action                     |
| --------------------------------- | --------------- | -------------------------- |
| RAGAS faithfulness (daily sample) | < 0.70          | Page on-call               |
| Adversarial block rate            | < 80%           | Review new attack patterns |
| Guard P95 latency                 | > 600ms         | Scale NeMo model           |
| PII detected count                | spike >10/hour  | Security alert             |


---



## Kết quả thực tế từ Lab


|                               | Kết quả                                                            |
| ----------------------------- | ------------------------------------------------------------------ |
| RAGAS avg_score (50q)         | ~0.668 (factual 0.742 / multi_hop 0.625 / adversarial 0.604)       |
| Worst metric                  | answer_relevancy (~0.21)                                           |
| Dominant failure distribution | factual                                                            |
| Cohen's κ                     | 0.000 (demo placeholder — chưa nối judge thật vào 10 human labels) |
| Adversarial pass rate         | 20 / 20                                                            |
| Guard P95 latency             | 19722.75 ms (vượt budget 500ms)                                    |


---



## Nhận xét & Cải tiến

> **Hoạt động tốt:** Hybrid search (BM25 + dense bge-m3) + rerank cho context_precision rất cao (~0.95–0.98) ở cả 3 phân phối; guardrail chặn 20/20 adversarial input (off-topic, jailbreak, prompt injection); Presidio + custom recognizer phát hiện chính xác VN_CCCD/VN_PHONE/EMAIL và ẩn danh tốt.
>
> **Cần cải thiện:** `answer_relevancy` thấp (~0.21) là điểm yếu chủ đạo — gợi ý cải thiện prompt template (yêu cầu trả lời bám sát câu hỏi, ngắn gọn, đúng trọng tâm). Nhóm `multi_hop` yếu về faithfulness (0.53) do phải tổng hợp nhiều nguồn. Latency Presidio quá cao trên CPU.
>
> **Nếu deploy production:** (1) Tách lớp guard thành microservice riêng, chạy Presidio với model nhẹ/GPU để đạt budget <500ms; (2) Thêm CI gate chặn merge nếu faithfulness < 0.75 hoặc adversarial pass rate < 75%; (3) Nối judge thật vào human labels để đo Cohen's κ thực tế thay vì placeholder; (4) Monitoring dashboard cảnh báo khi RAGAS faithfulness hoặc block rate suy giảm.

