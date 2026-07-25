import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTokenizerManager(TokenizerControlMixin):
    def __init__(self):
        self.weight_sync_lock = asyncio.Lock()
        self.is_pause_cond = asyncio.Condition()
        self.is_pause = True
        self.pull_started = asyncio.Event()
        self.release_pull = asyncio.Event()
        self.cpu_update_started = asyncio.Event()
        self.release_cpu_update = asyncio.Event()
        self.release_cpu_update.set()

    def auto_create_handle_loop(self):
        pass

    async def pull_weights_communicator(self, _obj):
        self.pull_started.set()
        await self.release_pull.wait()
        return [SimpleNamespace(success=True, message="pulled", rank_stats=[])]

    async def update_weights_from_cpu_communicator(self, _obj):
        self.cpu_update_started.set()
        await self.release_cpu_update.wait()
        return [SimpleNamespace(success=True, message="updated", rank_stats=[])]

    def _update_weight_version_if_provided(self, _version):
        pass


class TestWeightSyncLock(unittest.IsolatedAsyncioTestCase):
    async def test_cpu_commit_waits_for_all_pull_responses(self):
        manager = _FakeTokenizerManager()
        request = SimpleNamespace(abort_all_requests=False, target_version=1)

        pull_task = asyncio.create_task(manager.pull_weights(request))
        await asyncio.wait_for(manager.pull_started.wait(), timeout=1)

        update_task = asyncio.create_task(manager.update_weights_from_cpu(request))
        await asyncio.sleep(0)
        self.assertFalse(manager.cpu_update_started.is_set())

        manager.release_pull.set()
        await asyncio.wait_for(pull_task, timeout=1)
        await asyncio.wait_for(update_task, timeout=1)
        self.assertTrue(manager.cpu_update_started.is_set())

    async def test_cancelled_pull_still_blocks_cpu_commit(self):
        manager = _FakeTokenizerManager()
        request = SimpleNamespace(abort_all_requests=False, target_version=1)

        pull_task = asyncio.create_task(manager.pull_weights(request))
        await asyncio.wait_for(manager.pull_started.wait(), timeout=1)
        pull_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pull_task

        update_task = asyncio.create_task(manager.update_weights_from_cpu(request))
        await asyncio.sleep(0)
        self.assertFalse(manager.cpu_update_started.is_set())

        manager.release_pull.set()
        await asyncio.wait_for(update_task, timeout=1)
        self.assertTrue(manager.cpu_update_started.is_set())

    async def test_cancelled_queued_commit_never_reaches_schedulers(self):
        manager = _FakeTokenizerManager()
        request = SimpleNamespace(abort_all_requests=False, target_version=1)

        pull_task = asyncio.create_task(manager.pull_weights(request))
        await asyncio.wait_for(manager.pull_started.wait(), timeout=1)

        update_task = asyncio.create_task(manager.update_weights_from_cpu(request))
        await asyncio.sleep(0)
        update_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await update_task

        manager.release_pull.set()
        await asyncio.wait_for(pull_task, timeout=1)
        await asyncio.sleep(0)
        self.assertFalse(manager.cpu_update_started.is_set())

    async def test_paused_cpu_commit_blocks_unpause(self):
        manager = _FakeTokenizerManager()
        manager.release_cpu_update.clear()
        request = SimpleNamespace(abort_all_requests=False, target_version=1)

        update_task = asyncio.create_task(manager.update_weights_from_cpu(request))
        await asyncio.wait_for(manager.cpu_update_started.wait(), timeout=1)

        unpaused = asyncio.Event()

        async def unpause():
            async with manager.is_pause_cond:
                manager.is_pause = False
                unpaused.set()

        unpause_task = asyncio.create_task(unpause())
        await asyncio.sleep(0)
        self.assertFalse(unpaused.is_set())

        manager.release_cpu_update.set()
        await asyncio.wait_for(update_task, timeout=1)
        await asyncio.wait_for(unpause_task, timeout=1)
        self.assertTrue(unpaused.is_set())


class TestCPUWeightCacheConfiguration(unittest.TestCase):
    def test_cpu_weight_cache_rejects_automatic_eplb(self):
        args = ServerArgs.__new__(ServerArgs)
        args.enable_cpu_weight_cache = True
        args.enable_eplb = True

        with self.assertRaisesRegex(ValueError, "automatic EPLB"):
            args._handle_eplb_and_dispatch()
