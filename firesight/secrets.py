"""Per-user Windows DPAPI protection for local FireSight camera secrets."""
from __future__ import annotations

import base64
import ctypes
import json
import os
import tempfile
from ctypes import wintypes
from pathlib import Path


class SecretStoreUnavailable(RuntimeError):
    """Raised when the OS cannot provide the configured secret protection."""


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob(value: bytes) -> tuple[_DataBlob, ctypes.Array]:
    buffer = ctypes.create_string_buffer(value)
    return _DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))), buffer


def _dpapi(value: bytes, *, decrypt: bool) -> bytes:
    if os.name != "nt":
        raise SecretStoreUnavailable("Windows DPAPI is unavailable on this operating system.")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [ctypes.POINTER(_DataBlob), wintypes.LPCWSTR, ctypes.POINTER(_DataBlob), wintypes.LPVOID, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(_DataBlob)]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [ctypes.POINTER(_DataBlob), ctypes.POINTER(wintypes.LPWSTR), ctypes.POINTER(_DataBlob), wintypes.LPVOID, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(_DataBlob)]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    source, source_buffer = _blob(value)
    target = _DataBlob()
    entropy, entropy_buffer = _blob(b"FireSight-AI registry key v1")
    if decrypt:
        ok = crypt32.CryptUnprotectData(ctypes.byref(source), None, ctypes.byref(entropy), None, None, 0, ctypes.byref(target))
    else:
        ok = crypt32.CryptProtectData(ctypes.byref(source), "FireSight AI protected value", ctypes.byref(entropy), None, None, 0, ctypes.byref(target))
    # Keep input buffers alive until the native call has returned.
    _ = source_buffer, entropy_buffer
    if not ok:
        raise SecretStoreUnavailable("Windows could not protect or recover the camera secret for this user account.")
    try:
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        kernel32.LocalFree(target.pbData)


class CameraSecretStore:
    """Stores only DPAPI-encrypted URL blobs in a local JSON file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(payload.get("secrets"), dict):
                raise ValueError
            return {
                str(camera_id): _dpapi(base64.b64decode(encoded, validate=True), decrypt=True).decode("utf-8")
                for camera_id, encoded in payload["secrets"].items()
            }
        except SecretStoreUnavailable:
            raise
        except Exception as exc:
            raise SecretStoreUnavailable("The protected camera secret store is unreadable; secrets were not recovered.") from exc

    def save(self, values: dict[str, str]) -> None:
        protected = {
            camera_id: base64.b64encode(_dpapi(value.encode("utf-8"), decrypt=False)).decode("ascii")
            for camera_id, value in values.items()
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, suffix=".tmp", delete=False) as stream:
                temp_path = Path(stream.name)
                json.dump({"version": 1, "secrets": protected}, stream, indent=2)
            os.replace(temp_path, self.path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
