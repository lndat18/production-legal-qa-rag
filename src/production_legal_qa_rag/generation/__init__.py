"""Sinh câu trả lời pháp luật có trích dẫn từ các chunk đã truy xuất."""

from production_legal_qa_rag.generation.pipeline import (
    GenerationPipeline,
    answer_stream,
)

__all__ = ["GenerationPipeline", "answer_stream"]
