"""Exercise the real Qt event loop without opening desktop windows."""
import os
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLBACKEND', 'Agg')

import pytest
from PySide6 import QtWidgets


@pytest.fixture(scope='session')
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


@pytest.fixture
def spin_until(qapp):
    def wait(predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while not predicate():
            qapp.processEvents()
            if time.monotonic() >= deadline:
                pytest.fail('Qt operation timed out')
            time.sleep(.002)
        qapp.processEvents()
    return wait
