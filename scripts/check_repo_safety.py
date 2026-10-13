"""Fail CI on likely leaked secrets or newly added generated/heavy artifacts.

Only paths changed against the CI base or present in the local working tree are
inspected. Findings report filenames/line numbers, never matched secret values.
"""
from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 25 * 1024 * 1024
MODEL_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".tflite"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
SECRET_QUERY_KEYS = {"password", "passwd", "token", "secret", "api_key", "apikey", "access_token"}
PLACEHOLDER_VALUES = {"example", "replace-me", "replace_with_secret", "your-secret", "changeme"}
TEST_FIXTURE_VALUES = {
    "viewer", "private", "secret", "user", "password", "legacy-user", "legacy-pass", "view@home",
    "view@site", "viewer@site", "p:a ss", "u@ser", "p:ss", "required", "legacy-token", "hidden", "topsecret",
}
RTSP_PATTERN = re.compile(r"(?i)\brtsps?://[^\s\"'<>]+")
SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|password|passwd)\b\s*[:=]\s*['\"]?([A-Za-z0-9_./+=-]{16,})"
)
KNOWN_ARTIFACT_SHA256 = {
    # Existing tracked weights/run metadata are allowed only at these exact contents.
    "models/best.pt": "33f6641e9dba8fd0a419aca1eded81eae7d049f4ee794586bef435600243c10e",
    "models/drowning_best.pt": "9d09950c948131ae6e203a17eae6102e9c51974358bafc6afa7418460c91d3a5",
    "yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "yolo26n.pt": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    "runs/detect/runs/firesight/args.yaml": "74c4b75bd499b3ca9c6d879495ee23a6f37cd0b45c9085554a47ff7868e33bf4",
    "runs/detect/runs/firesight-2/args.yaml": "e37c5480588ef599e7c81d1400f0225c961e295c43aeaf008fd04bf635e51eba",
}


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def changed_paths() -> set[str]:
    base = os.environ.get("BASE_SHA", "").strip()
    if not base or set(base) == {"0"}:
        try:
            base = _git("rev-parse", "HEAD^").decode().strip()
        except subprocess.CalledProcessError:
            base = ""
    if base:
        try:
            _git("cat-file", "-e", f"{base}^{{commit}}")
            paths = set(_git("diff", "--name-only", "--diff-filter=AMR", "-z", f"{base}...HEAD").decode().split("\0"))
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("CI base commit is unavailable; checkout must fetch full history.") from exc
    else:
        paths = set(_git("ls-files", "-z").decode().split("\0"))

    # Include edits made locally so the exact CI safety checks can run before commit.
    paths.update(_git("diff", "--name-only", "--diff-filter=AMR", "-z", "HEAD").decode().split("\0"))
    paths.update(_git("diff", "--cached", "--name-only", "--diff-filter=AMR", "-z").decode().split("\0"))
    paths.update(_git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0"))
    return {path.replace("\\", "/") for path in paths if path}


def _fixture_only_rtsp(path: str, host: str | None, username: str | None, password: str | None, query: list[tuple[str, str]]) -> bool:
    if not path.startswith("tests/") or host not in {"camera", "camera.local", "2001:db8::1"}:
        return False
    values = {unquote(username or "").lower(), unquote(password or "").lower()}
    values.update(value.lower() for key, value in query if key.lower() in SECRET_QUERY_KEYS)
    values.discard("")
    return values <= TEST_FIXTURE_VALUES


def _text_findings(path: str, content: str) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    for number, line in enumerate(content.splitlines(), 1):
        for match in RTSP_PATTERN.finditer(line):
            value = match.group(0).rstrip(".,);]")
            try:
                parts = urlsplit(value)
                query = parse_qsl(parts.query, keep_blank_values=True)
            except ValueError:
                findings.append((number, "malformed RTSP URL; verify it contains no private endpoint"))
                continue

            has_userinfo = parts.username is not None or parts.password is not None
            has_query_secret = any(key.lower() in SECRET_QUERY_KEYS and item for key, item in query)
            private_ip = False
            private_host = False
            if parts.hostname:
                hostname = parts.hostname.lower()
                private_host = hostname == "localhost" or hostname.endswith((".local", ".lan", ".home", ".internal"))
                try:
                    private_ip = ipaddress.ip_address(parts.hostname).is_private
                except ValueError:
                    pass
            if _fixture_only_rtsp(path, parts.hostname, parts.username, parts.password, query):
                continue
            if has_userinfo or has_query_secret:
                findings.append((number, "credential-bearing RTSP URL; matched value redacted"))
            elif private_ip or private_host:
                findings.append((number, "private-network RTSP endpoint; matched value redacted"))

        for match in SECRET_ASSIGNMENT_PATTERN.finditer(line):
            candidate = match.group(1).strip("'\".,;)")
            if candidate.lower() not in PLACEHOLDER_VALUES and len(candidate) >= 16:
                findings.append((number, "possible hard-coded secret; matched value redacted"))
                break
    return findings


def _artifact_findings(path: str, file_path: Path) -> list[str]:
    normalized = path.lower()
    parts = set(Path(normalized).parts)
    blocked_parts = {
        "dataset", "node_modules", "venv", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
        ".pnpm-store", ".vite", ".cache", "dist", "evidence", "recordings", "checkpoints",
    }
    if parts & blocked_parts or any(part.startswith("dataset_backup") for part in parts) or "backup_before_" in normalized or normalized.startswith("runs/firesight/"):
        return ["generated data, environment, backup, or runtime artifact path is not allowed"]
    if normalized.startswith("runs/") and path not in KNOWN_ARTIFACT_SHA256:
        return ["new runtime output under runs/ is not allowed"]
    if normalized == ".env" or (normalized.startswith(".env.") and normalized != ".env.example"):
        return ["environment secret file is not allowed"]
    if not file_path.is_file():
        return []
    if file_path.stat().st_size > MAX_FILE_BYTES:
        return ["file exceeds the 25 MiB change limit"]

    suffix = file_path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return ["video/media additions are not allowed in source changes"]
    if suffix in MODEL_SUFFIXES:
        expected = KNOWN_ARTIFACT_SHA256.get(path)
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if expected is None or actual != expected:
            return ["new or changed model artifact; review and maintainers must handle separately"]
    if suffix in {".jpg", ".jpeg", ".png", ".wav", ".mp3"} and any(part in parts for part in {"data", "dataset", "evidence", "recordings"}):
        return ["dataset, evidence, or recording asset is not allowed in source changes"]
    return []


def main() -> int:
    errors: list[tuple[str, int | None, str]] = []
    for path in sorted(changed_paths()):
        file_path = ROOT / path
        resolved_path = file_path.resolve()
        if file_path.is_symlink() or not resolved_path.is_relative_to(ROOT.resolve()):
            errors.append((path, None, "symbolic links and paths outside the repository are not allowed"))
            continue
        for message in _artifact_findings(path, file_path):
            errors.append((path, None, message))
        if not file_path.is_file() or file_path.suffix.lower() in MODEL_SUFFIXES | VIDEO_SUFFIXES:
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        errors.extend((path, line, message) for line, message in _text_findings(path, text))

    if errors:
        for path, line, message in errors:
            safe_path = path.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A").replace(":", "%3A").replace(",", "%2C")
            location = f" file={safe_path},line={line}" if line is not None else f" file={safe_path}"
            print(f"::error{location}::{message}")
        print(f"Repository safety check failed with {len(errors)} redacted finding(s).")
        return 1
    print("Repository safety check passed; no secrets or disallowed generated/artifact changes found.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Repository safety check could not complete: {exc}", file=sys.stderr)
        raise SystemExit(2)
