"""Structured logging with JSON formatter."""

import logging
import json
import time
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a logger with JSON formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_llm_call(
    logger: logging.Logger,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    agent: str = "",
    **kwargs: Any,
) -> None:
    """Log an LLM API call with structured data."""
    extra = {
        "event": "llm_call",
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": round(latency_ms, 1),
        "agent": agent,
        **kwargs,
    }
    logger.info("LLM call completed", extra={"extra_data": extra})


def log_tool_call(
    logger: logging.Logger,
    tool_name: str,
    query: str,
    cached: bool = False,
    latency_ms: float = 0.0,
    **kwargs: Any,
) -> None:
    """Log a tool execution with structured data."""
    extra = {
        "event": "tool_call",
        "tool": tool_name,
        "query": query[:200],
        "cached": cached,
        "latency_ms": round(latency_ms, 1),
        **kwargs,
    }
    logger.info("Tool call completed", extra={"extra_data": extra})


def log_stage_transition(
    logger: logging.Logger,
    stage: str,
    status: str,
    count: int = 0,
    **kwargs: Any,
) -> None:
    """Log a pipeline stage transition."""
    extra = {
        "event": "stage_transition",
        "stage": stage,
        "status": status,
        "item_count": count,
        **kwargs,
    }
    logger.info(f"Stage {stage}: {status}", extra={"extra_data": extra})


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
