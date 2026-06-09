"""
BLE device driver for the BACKY sensor — demo_app edition.

Extracted from backysight.py. Stripped of all Tkinter / heatmap code.
Provides BLESession and BLEWorker which are used by api.py to manage the
device connection in a background daemon thread.

Packet format (Nordic UART Service TX characteristic):
    struct.unpack("<6Hhhh", data)
    → s1..s6  (uint16, raw tactile ADC)
    → ax, ay, az  (int16 / 100.0 → float g)

Column order that matches the model training schema:
    sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, ax, ay, az
"""
from __future__ import annotations

import asyncio
import logging
import math
import struct
import sys
import threading
import time

# Windows-specific: use the selector event loop (required for bleak on Windows)
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from bleak import BleakClient, BleakError, BleakScanner  # type: ignore

# Use the same logger as the UI so messages land in logs/realtime.log
_dlog = logging.getLogger("backy_ui.device")

# ── Nordic UART Service UUIDs ────────────────────────────────────────────────
NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"   # device → host

# Column names in the same order as the model's SENSOR_COLS
SENSOR_COLUMNS = ["sensor1", "sensor2", "sensor3", "sensor4", "sensor5", "sensor6",
                   "ax", "ay", "az"]
CSV_HEADER = ["timestamp"] + SENSOR_COLUMNS

# Packet struct formats (little-endian) — accept both firmware variants
_FMT_FULL    = "<6Hhhh"   # 6 tactile sensors + ax/ay/az
_FMT_SINGLE  = "<Hhhh"    # 1 tactile sensor  + ax/ay/az
_SIZE_FULL   = struct.calcsize(_FMT_FULL)    # 18
_SIZE_SINGLE = struct.calcsize(_FMT_SINGLE)  #  8


def get_signed_tilt_angle(rx: float, ry: float, rz: float) -> float:
    """Compute signed tilt angle (degrees) from raw accelerometer readings."""
    tx, ty, tz = ry, rz, -rx
    mag = math.sqrt(tx * tx + ty * ty + tz * tz) or 1.0
    return -math.degrees(math.asin(ty / mag))


def sanitize_name(name: str) -> str:
    """Make a string safe to use as a filename component."""
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (name or "").strip())
    return cleaned or "device"


# ── Raw BLE session ──────────────────────────────────────────────────────────

class BLESession:
    """Async BLE session: scan → connect → stream notifications."""

    def __init__(self, device_name: str):
        self.device_name = device_name
        self.client: BleakClient | None = None
        self.latest: tuple | None = None   # (s_list, x, y, z, ts)
        self._pkt_count: int = 0
        self._short_pkt_warned: int = 0
        self._disconnect_flag: bool = False    # set by bleak's disconnected_callback

    async def scan_until_found(self, cancel_event: asyncio.Event) -> str | None:
        """Scan repeatedly until the named device is found or cancelled."""
        while not cancel_event.is_set():
            try:
                devices = await BleakScanner.discover(timeout=1.0)
                for d in devices:
                    if (d.name or "").strip() == self.device_name.strip():
                        return d.address
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return None

    async def connect(self, address: str) -> None:
        def _on_disconnect(client):
            self._disconnect_flag = True
            _dlog.warning("[device] bleak disconnected_callback fired  addr=%s", address)

        self.client = BleakClient(address, disconnected_callback=_on_disconnect)
        _dlog.info("[device] connecting to %s …", address)
        await self.client.connect()
        if not self.client.is_connected:
            raise BleakError("Connection failed after BleakClient.connect().")
        _dlog.info("[device] connected  mtu=%s", getattr(self.client, "mtu_size", "?"))
        await self.client.start_notify(NUS_TX_CHAR_UUID, self._on_notify)
        _dlog.info("[device] notifications started on %s", NUS_TX_CHAR_UUID)

    async def disconnect(self) -> None:
        if self.client:
            try:
                if self.client.is_connected:
                    await self.client.stop_notify(NUS_TX_CHAR_UUID)
                    await self.client.disconnect()
            except Exception:
                pass
        self.client = None

    def _on_notify(self, _sender: int, data: bytearray) -> None:
        """Called by bleak on every BLE notification from the device.

        Accepts BOTH known firmware variants:
          • 18-byte full board   "<6Hhhh"   → 6 FSR + ax/ay/az
          •  8-byte single board "<Hhhh"    → 1 FSR + ax/ay/az  (other 5 zero-filled)
        Larger packets are tolerated by reading only the first 18 bytes.
        """
        try:
            n = len(data)
            self._pkt_count += 1
            # log the first packet (so we can confirm format), then every 500th
            if self._pkt_count == 1 or self._pkt_count % 500 == 0:
                _dlog.info("[device] notify #%d  size=%d bytes  raw=%s",
                           self._pkt_count, n, bytes(data[:32]).hex())

            if n >= _SIZE_FULL:
                unpacked = struct.unpack_from(_FMT_FULL, data)
                s = list(unpacked[:6])
                x = unpacked[6] / 100.0
                y = unpacked[7] / 100.0
                z = unpacked[8] / 100.0
            elif n >= _SIZE_SINGLE:
                unpacked = struct.unpack_from(_FMT_SINGLE, data)
                s1 = unpacked[0]
                s  = [s1, 0, 0, 0, 0, 0]      # only sensor1 valid
                x = unpacked[1] / 100.0
                y = unpacked[2] / 100.0
                z = unpacked[3] / 100.0
                if self._pkt_count == 1:
                    _dlog.warning(
                        "[device] single-sensor firmware detected — only sensor1 will have real data"
                    )
            else:
                if self._short_pkt_warned < 3:
                    _dlog.warning("[device] dropped short packet  size=%d bytes  raw=%s",
                                  n, bytes(data).hex())
                    self._short_pkt_warned += 1
                return
            self.latest = (s, x, y, z, time.time())
        except Exception as exc:
            _dlog.error("[device] parse error  size=%d  err=%s  raw=%s",
                        len(data), exc, bytes(data[:32]).hex())

    def read_latest(self) -> tuple | None:
        """Consume and return the latest packet (or None if no new data)."""
        out = self.latest
        self.latest = None
        return out


# ── Thread-based worker ──────────────────────────────────────────────────────

class BLEWorker:
    """
    Runs the BLE event loop in a dedicated daemon thread.

    Callbacks (called from the BLE thread — must be thread-safe):
        on_status(status: str)
            "searching" | "connected" | "idle" | "error:<msg>"
        on_reading(reading: dict)
            Keys: ts, sensor1..sensor6, ax, ay, az, tilt
    """

    def __init__(self, on_status, on_reading):
        self.on_status  = on_status
        self.on_reading = on_reading
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._cancel_event: asyncio.Event | None = None
        self._device_name: str | None = None

    def start(self, device_name: str) -> None:
        """Start scanning and connecting to `device_name` in the background."""
        if self._thread and self._thread.is_alive():
            return
        self._device_name = device_name
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the BLE loop to stop (non-blocking)."""
        if self._loop and self._cancel_event and not self._cancel_event.is_set():
            self._loop.call_soon_threadsafe(self._cancel_event.set)

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── internals ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        if sys.platform.startswith("win"):
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except Exception:
                pass
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._cancel_event = asyncio.Event()
        try:
            self._loop.run_until_complete(self._session_main(self._device_name))
        finally:
            try:
                self._loop.stop()
            except Exception:
                pass

    async def _session_main(self, device_name: str) -> None:
        session = BLESession(device_name)
        self.on_status("searching")
        _dlog.info("[worker] scanning for device '%s'", device_name)

        address = await session.scan_until_found(self._cancel_event)
        if self._cancel_event.is_set() or address is None:
            _dlog.info("[worker] scan ended without finding device (cancelled=%s)",
                       self._cancel_event.is_set())
            self.on_status("idle")
            return

        _dlog.info("[worker] device '%s' found at %s", device_name, address)

        try:
            await session.connect(address)
            self.on_status("connected")
            t_connected = time.time()

            while not self._cancel_event.is_set():
                # detect a disconnect that came in via bleak's callback
                if session._disconnect_flag:
                    raise BleakError("Device disconnected (bleak disconnected_callback)")

                packet = session.read_latest()
                if packet is not None:
                    (s1, s2, s3, s4, s5, s6), x, y, z, ts = packet
                    tilt = get_signed_tilt_angle(x, y, z)
                    self.on_reading({
                        "ts":      ts,
                        "sensor1": s1,
                        "sensor2": s2,
                        "sensor3": s3,
                        "sensor4": s4,
                        "sensor5": s5,
                        "sensor6": s6,
                        "ax":      x,
                        "ay":      y,
                        "az":      z,
                        "tilt":    tilt,
                    })
                await asyncio.sleep(0.01)

            _dlog.info("[worker] cancel requested  uptime=%.1fs  packets=%d",
                       time.time() - t_connected, session._pkt_count)

        except Exception as exc:
            uptime = time.time() - t_connected if 't_connected' in locals() else 0.0
            _dlog.exception(
                "[worker] session ended with exception  uptime=%.1fs  packets=%d  err=%r",
                uptime, session._pkt_count, exc,
            )
            self.on_status(f"error:{exc}")
        finally:
            await session.disconnect()
            _dlog.info("[worker] cleanup done — session closed")
            self.on_status("idle")
