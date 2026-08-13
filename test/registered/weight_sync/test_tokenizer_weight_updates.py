from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.managers.io_struct import (
    StageWeightUpdateReqOutput,
    UpdateWeightFromCPUReqOutput,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.asyncio
async def test_stage_weight_update_flattens_worker_stats():
    request = object()
    manager = SimpleNamespace(
        auto_create_handle_loop=mock.Mock(),
        stage_weight_update_communicator=mock.AsyncMock(
            return_value=[
                StageWeightUpdateReqOutput(
                    success=True,
                    message="rank 0",
                    rank_stats=[{"rank": 0}],
                ),
                StageWeightUpdateReqOutput(
                    success=True,
                    message="rank 1",
                    rank_stats=[{"rank": 1}],
                ),
            ]
        ),
    )

    success, message, rank_stats = await TokenizerControlMixin.stage_weight_update(
        manager,
        request,
    )

    assert success
    assert message == "rank 0 | rank 1"
    assert rank_stats == [{"rank": 0}, {"rank": 1}]
    manager.stage_weight_update_communicator.assert_awaited_once_with(request)


@pytest.mark.asyncio
async def test_cpu_weight_update_aborts_requests_and_records_version():
    request = SimpleNamespace(target_version=7, abort_all_requests=True)
    manager = SimpleNamespace(
        auto_create_handle_loop=mock.Mock(),
        abort_request=mock.Mock(),
        update_weights_from_cpu_communicator=mock.AsyncMock(
            return_value=[
                UpdateWeightFromCPUReqOutput(
                    success=True,
                    message="committed",
                    rank_stats=[{"rank": 0}],
                )
            ]
        ),
        _update_weight_version_if_provided=mock.Mock(),
    )

    success, message, rank_stats = await TokenizerControlMixin.update_weights_from_cpu(
        manager, request
    )

    assert success
    assert message == "committed Weight version updated to 7."
    assert rank_stats == [{"rank": 0}]
    manager.abort_request.assert_called_once_with(abort_all=True)
    manager._update_weight_version_if_provided.assert_called_once_with("7")


@pytest.mark.asyncio
async def test_failed_cpu_weight_update_does_not_advance_version():
    request = SimpleNamespace(target_version=7, abort_all_requests=False)
    manager = SimpleNamespace(
        auto_create_handle_loop=mock.Mock(),
        abort_request=mock.Mock(),
        update_weights_from_cpu_communicator=mock.AsyncMock(
            return_value=[
                UpdateWeightFromCPUReqOutput(
                    success=False,
                    message="not ready",
                )
            ]
        ),
        _update_weight_version_if_provided=mock.Mock(),
    )

    success, message, rank_stats = await TokenizerControlMixin.update_weights_from_cpu(
        manager, request
    )

    assert not success
    assert message == "not ready"
    assert rank_stats == []
    manager.abort_request.assert_not_called()
    manager._update_weight_version_if_provided.assert_not_called()
