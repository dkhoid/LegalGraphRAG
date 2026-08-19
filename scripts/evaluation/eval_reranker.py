#!/usr/bin/env python3
"""
Evaluate the performance of different CrossEncoder models for Vietnamese legal text.
Usage:
    python scripts/evaluation/eval_reranker.py
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.retriever.reranker import CrossEncoderReranker

# A small validation dataset (Query + Candidate Laws + Expected Law ID)
TEST_CASES = [
    {
        "query": "Bố tôi mất không để lại di chúc. Căn nhà bố tôi để lại sẽ được chia cho các anh em như thế nào?",
        "expected_top_law_id": "BLDS_650",
        "candidates": [
            {
                "id": "BLDS_116",
                "description": "Giao dịch dân sự là hợp đồng hoặc hành vi pháp lý đơn phương làm phát sinh, thay đổi hoặc chấm dứt quyền, nghĩa vụ dân sự.",
            },
            {
                "id": "BLDS_650",
                "description": "Thừa kế theo pháp luật được áp dụng trong trường hợp không có di chúc, di chúc không hợp pháp, hoặc những người thừa kế theo di chúc chết trước.",
            },
            {
                "id": "LLD_36",
                "description": "Người sử dụng lao động có quyền đơn phương chấm dứt hợp đồng lao động trong trường hợp người lao động thường xuyên không hoàn thành công việc.",
            },
            {
                "id": "BLDS_624",
                "description": "Di chúc là sự thể hiện ý chí của cá nhân nhằm chuyển tài sản của mình cho người khác sau khi chết.",
            },
        ],
    },
    {
        "query": "Công ty sa thải tôi ngay lập tức mà không báo trước vì lý do tôi đi trễ 1 ngày. Việc này có đúng luật không?",
        "expected_top_law_id": "LLD_36",
        "candidates": [
            {
                "id": "LLD_36",
                "description": "Quyền đơn phương chấm dứt hợp đồng lao động của người sử dụng lao động. Phải báo trước ít nhất 45 ngày đối với hợp đồng không xác định thời hạn.",
            },
            {
                "id": "BLDS_385",
                "description": "Hợp đồng là sự thỏa thuận giữa các bên về việc xác lập, thay đổi hoặc chấm dứt quyền, nghĩa vụ dân sự.",
            },
            {
                "id": "LLD_125",
                "description": "Hình thức xử lý kỷ luật sa thải được áp dụng trong trường hợp người lao động có hành vi trộm cắp, tham ô, đánh bạc, cố ý gây thương tích.",
            },
            {
                "id": "BLDS_584",
                "description": "Căn cứ phát sinh trách nhiệm bồi thường thiệt hại ngoài hợp đồng.",
            },
        ],
    },
]


def evaluate_model(model_name: str):
    print(f"\n{'='*60}")
    print(f"Evaluating Model: {model_name}")
    print(f"{'='*60}")

    try:
        reranker = CrossEncoderReranker(model_name=model_name, top_k=4)

        start_time = time.time()

        correct = 0
        total = len(TEST_CASES)

        for idx, tc in enumerate(TEST_CASES, 1):
            print(f"\n[Test Case {idx}]")
            print(f"Query: {tc['query']}")

            results = reranker.rerank(tc["query"], tc["candidates"])

            top_id = results[0]["id"] if results else None
            is_correct = top_id == tc["expected_top_law_id"]
            if is_correct:
                correct += 1

            print(f"Expected Top 1: {tc['expected_top_law_id']}")
            print(
                f"Actual Top 1:   {top_id} (Score: {results[0].get('_rerank_score', 'N/A') if results else 'N/A'})"
            )
            print(f"Status:         {'✅ PASS' if is_correct else '❌ FAIL'}")

            print("Rankings:")
            for rank, res in enumerate(results, 1):
                print(f"  {rank}. {res['id']} (Score: {res.get('_rerank_score', 'N/A')})")

        elapsed = time.time() - start_time
        accuracy = (correct / total) * 100
        print(f"\n[Summary for {model_name}]")
        print(f"Accuracy (Top-1): {accuracy:.2f}% ({correct}/{total})")
        print(f"Time Taken:       {elapsed:.2f} seconds")

    except ImportError:
        print("Missing 'sentence-transformers'. Please install it first.")
    except Exception as e:
        print(f"Error evaluating model {model_name}: {e}")


if __name__ == "__main__":
    # Test the default English-centric model
    evaluate_model("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Test the recommended multilingual model
    evaluate_model("nreimers/mmarco-mMiniLMv2-L12-H384-v1")
