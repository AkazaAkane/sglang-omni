# SPDX-License-Identifier: Apache-2.0
"""Shared mechanics for pre-LM encoder services."""

from __future__ import annotations

import concurrent.futures
import contextlib
import queue
import threading
import time
from contextlib import AbstractContextManager
from typing import Any, Generic, TypeVar

ItemT = TypeVar("ItemT")
EncodedT = TypeVar("EncodedT")
EmbeddingT = TypeVar("EmbeddingT")

QueueEntry = tuple[Any, ...]


class PreLMEncoderService(Generic[ItemT, EncodedT, EmbeddingT]):
    """Run model-owned encoder hooks on a queue-backed worker thread."""

    def __init__(self, *, worker_name: str) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker,
            name=worker_name,
            daemon=True,
        )
        self._thread.start()

    def _enqueue(
        self,
        item: ItemT,
        future: concurrent.futures.Future[Any],
    ) -> None:
        self._queue.put((item, future))

    def _submit(
        self,
        item: ItemT,
        future: concurrent.futures.Future[Any] | None = None,
    ) -> concurrent.futures.Future[Any]:
        if future is None:
            future = concurrent.futures.Future()
        self._enqueue(item, future)
        return future

    def _next_batch(self) -> tuple[list[QueueEntry], bool]:
        raise NotImplementedError

    def _batch_context(self) -> AbstractContextManager[Any]:
        return contextlib.nullcontext()

    def encode_batch(self, items: list[ItemT]) -> EncodedT:
        raise NotImplementedError

    def split_embeddings(
        self,
        items: list[ItemT],
        encoded: EncodedT,
    ) -> list[EmbeddingT]:
        raise NotImplementedError

    def attach_embedding(self, item: ItemT, embedding: EmbeddingT) -> None:
        raise NotImplementedError

    def synchronize_batch(self) -> None:
        pass

    def cache_embedding(self, item: ItemT, embedding: EmbeddingT) -> None:
        pass

    def attach_before_synchronize(self) -> bool:
        return True

    def _execute_batch(self, items: list[ItemT]) -> list[EmbeddingT]:
        attach_before_synchronize = self.attach_before_synchronize()
        with self._batch_context():
            encoded = self.encode_batch(items)
            embeddings = self.split_embeddings(items, encoded)
            if attach_before_synchronize:
                for item, embedding in zip(items, embeddings):
                    self.attach_embedding(item, embedding)
        self.synchronize_batch()
        if not attach_before_synchronize:
            for item, embedding in zip(items, embeddings):
                self.attach_embedding(item, embedding)
        for item, embedding in zip(items, embeddings):
            self.cache_embedding(item, embedding)
        return embeddings

    def _handle_batch_failure(
        self,
        batch: list[QueueEntry],
        exc: Exception,
    ) -> Exception:
        return exc

    def _handle_item_failure(self, entry: QueueEntry, exc: Exception) -> Exception:
        return exc

    def _retry_batch(self, batch: list[QueueEntry], exc: Exception) -> bool:
        return False

    def _future_result(self, embedding: EmbeddingT) -> Any:
        return embedding

    def _on_batch_start(self, batch: list[QueueEntry]) -> None:
        pass

    def _on_batch_finished(
        self,
        batch: list[QueueEntry],
        batch_exc: Exception | None,
        retry_recovered: int | None,
        elapsed_s: float,
    ) -> None:
        pass

    def _worker(self) -> None:
        while True:
            batch, shutdown = self._next_batch()
            if not batch:
                return
            self._on_batch_start(batch)
            items = [entry[0] for entry in batch]
            encode_start = time.perf_counter()
            try:
                embeddings = self._execute_batch(items)
            except Exception as batch_exc:
                batch_exc = self._handle_batch_failure(batch, batch_exc)
                if not self._retry_batch(batch, batch_exc):
                    for entry in batch:
                        entry[1].set_exception(batch_exc)
                    self._on_batch_finished(
                        batch,
                        batch_exc,
                        None,
                        time.perf_counter() - encode_start,
                    )
                    if shutdown:
                        return
                    continue
                recovered = 0
                for entry in batch:
                    try:
                        embedding = self._execute_batch([entry[0]])[0]
                        entry[1].set_result(self._future_result(embedding))
                        recovered += 1
                    except Exception as item_exc:
                        item_exc = self._handle_item_failure(entry, item_exc)
                        entry[1].set_exception(item_exc)
                self._on_batch_finished(
                    batch,
                    batch_exc,
                    recovered,
                    time.perf_counter() - encode_start,
                )
                if shutdown:
                    return
                continue
            for entry, embedding in zip(batch, embeddings):
                entry[1].set_result(self._future_result(embedding))
            self._on_batch_finished(
                batch,
                None,
                None,
                time.perf_counter() - encode_start,
            )
            if shutdown:
                return


__all__ = ["PreLMEncoderService", "QueueEntry"]
