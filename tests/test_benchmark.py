import time
from src.utils.benchmark import ResourceMonitor


def test_resource_monitor_lifecycle():
    monitor = ResourceMonitor(interval=1)

    assert monitor.is_running is False
    assert monitor.monitor_thread is None

    monitor.start()
    assert monitor.is_running is True
    assert monitor.monitor_thread is not None
    assert monitor.monitor_thread.is_alive() is True

    time.sleep(1.2)

    monitor.stop()
    assert monitor.is_running is False