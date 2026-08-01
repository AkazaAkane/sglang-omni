# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ThinkerModelRunner async-decode behavior.

lookahead_eligible reads only per-request flags (never other instance state), so it
is exercised on a bare instance built with ``object.__new__`` and stand-in requests.
"""
from __future__ import annotations

import types

import torch

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner


def _runner() -> ThinkerModelRunner:
    return object.__new__(ThinkerModelRunner)


def _sp(**kw):
    d = dict(
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        min_new_tokens=0,
        sampling_seed=None,
        logit_bias=None,
        custom_params=None,
    )
    d.update(kw)
    return types.SimpleNamespace(**d)


def _req(return_logprob=False, stage_payload="text", **sp_kw):
    return types.SimpleNamespace(
        sampling_params=_sp(**sp_kw),
        _omni_data=types.SimpleNamespace(
            return_logprob=return_logprob, stage_payload=stage_payload
        ),
    )


def _batch(*reqs):
    return types.SimpleNamespace(reqs=list(reqs))


def test_plain_greedy_is_eligible():
    assert _runner().lookahead_eligible(_batch(_req(), _req())) is True


def test_empty_batch_is_eligible():
    assert _runner().lookahead_eligible(_batch()) is True


def test_audio_output_is_eligible():
    assert _runner().lookahead_eligible(_batch(_req(stage_payload="audio"))) is True


def test_return_logprob_disables_lookahead():
    assert _runner().lookahead_eligible(_batch(_req(return_logprob=True))) is False


def test_missing_or_none_omni_data_falls_to_sync():
    # request data missing or None cannot be inspected -> fail closed to sync
    # (never raise, never let a possible hidden-capture batch onto async).
    no_data = types.SimpleNamespace(sampling_params=_sp())
    assert _runner().lookahead_eligible(_batch(no_data)) is False
    none_data = types.SimpleNamespace(sampling_params=_sp(), _omni_data=None)
    assert _runner().lookahead_eligible(_batch(none_data)) is False


def test_each_gated_sampling_param_disables_lookahead():
    for kw in (
        dict(repetition_penalty=1.3),
        dict(presence_penalty=0.5),
        dict(frequency_penalty=0.5),
        dict(min_new_tokens=5),
        dict(sampling_seed=42),
        dict(logit_bias={1: 2.0}),
        dict(custom_params={"x": 1}),
    ):
        assert _runner().lookahead_eligible(_batch(_req(**kw))) is False, kw


def test_one_gated_request_disables_whole_batch():
    audio_mix = _batch(_req(), _req(stage_payload="audio"), _req())
    assert _runner().lookahead_eligible(audio_mix) is True
    param_mix = _batch(_req(), _req(repetition_penalty=1.3), _req())
    assert _runner().lookahead_eligible(param_mix) is False


def test_async_speech_capture_is_double_buffered(monkeypatch):
    runner = _runner()
    runner.model = types.SimpleNamespace(_captured_aux_hidden_states=None)
    runner._th_capture_bufs = [None, None]
    runner._th_capture_slot = 0
    monkeypatch.setattr(
        runner,
        "_async_host_buf",
        lambda like, n: torch.empty(n, dtype=like.dtype),
    )

    def launch(token_id, aux_value, stream_value):
        runner.model._captured_aux_hidden_states = [
            torch.tensor([[aux_value]], dtype=torch.float32)
        ]
        result = types.SimpleNamespace(
            next_token_ids=torch.tensor([token_id]),
            logits_output=types.SimpleNamespace(
                hidden_states=torch.tensor([[stream_value]], dtype=torch.float32)
            ),
        )
        state = runner.post_decode_launch(result, object(), [object()])
        return result, state

    result_1, state_1 = launch(11, 1.0, 10.0)
    result_2, state_2 = launch(22, 2.0, 20.0)

    assert state_1.aux_hidden_states[0].data_ptr() != (
        state_2.aux_hidden_states[0].data_ptr()
    )
    runner.post_decode_resolve(state_1, result_1, object(), object(), [object()])
    assert result_1.next_token_ids.tolist() == [11]
    assert runner.model._captured_aux_hidden_states[0].item() == 1.0
    assert result_1.logits_output.hidden_states.item() == 10.0

    runner.model._captured_aux_hidden_states = None
    runner.post_decode_resolve(state_2, result_2, object(), object(), [object()])
    assert result_2.next_token_ids.tolist() == [22]
    assert runner.model._captured_aux_hidden_states[0].item() == 2.0
    assert result_2.logits_output.hidden_states.item() == 20.0
