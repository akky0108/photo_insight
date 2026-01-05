## Restart Checklist

1) Run unit tests:
- pytest -q tests/unit/headers -x
- pytest -q tests/unit/evaluators/test_portrait_quality_acceptance.py -x

2) Next target:
- Add unit tests for `FullBodyDetector.detect()` (mock mediapipe results)
- Ensure empty/small image returns `_empty_result`
- Ensure `detect_full_body()` delegates to `detect()`
