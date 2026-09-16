#!/usr/bin/env python3
"""Interactive end-to-end test against a *live* running instance of the
service -- not pytest, and not mocked. Exercises the real HTTP API exactly
as a client would.

Usage:
    python3 scripts/e2e_test.py               # interactive menu (default)
    python3 scripts/e2e_test.py --all          # run every step once, no menu
    python3 scripts/e2e_test.py --no-compose   # assume the API is already running

Each menu choice prints one short [PASS]/[FAIL] line per assertion it
makes. Picking a step that depends on an earlier one (e.g. "search" needs
an uploaded, completed file) runs whatever prerequisite steps haven't run
yet for you, silently, then does the step you asked for.

Uses httpx, already a project dependency (see requirements.txt) for the
TestClient in tests/ -- no extra install needed to run this.
"""

import subprocess
import sys
import time
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SAMPLE_FILE = SCRIPT_DIR / "sample.log"

API_URL = "http://localhost:8000/v1"
HEALTH_URL = "http://localhost:8000/health"
USER_A = "e2e-user-a"
USER_B = "e2e-user-b"
CHUNK_SIZE = 400  # deliberately small so the sample file needs several chunks


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def step(title: str) -> None:
    print(f"\n== {title} ==")


def info(msg: str) -> None:
    print(f"  {msg}")


class Session:
    """Live state for one run: the HTTP client and how far the sample
    upload has progressed. Each step method is idempotent -- calling it
    again after it already succeeded just re-confirms the same outcome --
    and lazily runs whatever earlier steps it depends on."""

    def __init__(self) -> None:
        self.client = httpx.Client(timeout=30)
        self.data = SAMPLE_FILE.read_bytes() if SAMPLE_FILE.exists() else b""
        self.file_size = len(self.data)
        self.file_id: str | None = None
        self.bytes_sent = 0
        self.completed = False
        self.indexed = False
        self.passed = 0
        self.failed = 0

    def record(self, success: bool, msg: str) -> None:
        if success:
            self.passed += 1
            ok(msg)
        else:
            self.failed += 1
            bad(msg)

    # ---- steps --------------------------------------------------------

    def register(self) -> None:
        if self.file_id is not None:
            return
        step("Register upload")
        resp = self.client.post(
            f"{API_URL}/files",
            headers={"X-User-Id": USER_A},
            json={"filename": "sample.log", "total_size": self.file_size},
        )
        if resp.status_code != 201:
            self.record(False, f"POST /files returned {resp.status_code}: {resp.text}")
            return
        self.file_id = resp.json()["file_id"]
        self.record(True, f"created upload, file_id={self.file_id}")

    def upload_with_interrupt(self) -> None:
        self.register()
        if self.file_id is None:
            return
        if self.bytes_sent >= self.file_size:
            return

        step("Chunked upload with a simulated interruption")
        offset = self.bytes_sent
        chunk_index = 0
        interrupted = False
        while offset < self.file_size:
            chunk_index += 1

            # Halfway through, stop without sending the rest -- simulates a
            # dropped connection -- then confirm /status reports the exact
            # resume point before continuing.
            if not interrupted and offset >= self.file_size // 2:
                interrupted = True
                status_resp = self.client.get(
                    f"{API_URL}/files/{self.file_id}/status", headers={"X-User-Id": USER_A}
                )
                reported_offset = status_resp.json()["bytes_received"]
                self.record(
                    reported_offset == offset,
                    f"interrupted after {offset} bytes; /status reports {reported_offset}"
                    if reported_offset == offset
                    else f"resume offset mismatch: sent {offset}, /status says {reported_offset}",
                )

            end = min(offset + CHUNK_SIZE, self.file_size)
            piece = self.data[offset:end]

            resp = self.client.put(
                f"{API_URL}/files/{self.file_id}/chunk",
                params={"offset": offset},
                headers={"X-User-Id": USER_A, "Content-Type": "application/octet-stream"},
                content=piece,
            )
            if resp.status_code != 200:
                self.record(
                    False,
                    f"chunk {chunk_index} (offset={offset}) rejected with HTTP {resp.status_code}: {resp.text}",
                )
                return
            offset = end
        self.bytes_sent = offset
        self.record(True, f"uploaded {self.file_size} bytes across {chunk_index} chunks")

    def complete(self) -> None:
        self.upload_with_interrupt()
        if self.file_id is None or self.completed:
            return

        step("Complete the upload")
        resp = self.client.post(f"{API_URL}/files/{self.file_id}/complete", headers={"X-User-Id": USER_A})
        upload_status = resp.json().get("upload_status")
        self.completed = upload_status == "completed"
        self.record(
            self.completed,
            "upload_status=completed"
            if self.completed
            else f"expected upload_status=completed, got '{upload_status}' (response: {resp.text})",
        )

    def poll_status(self) -> None:
        self.complete()
        if self.file_id is None or self.indexed:
            return

        step("Poll status until indexing catches up")
        processing_status = None
        for i in range(1, 61):
            resp = self.client.get(f"{API_URL}/files/{self.file_id}/status", headers={"X-User-Id": USER_A})
            body = resp.json()
            processing_status = body.get("processing_status")
            chunks_indexed = body.get("chunks_indexed", 0)
            if processing_status == "completed":
                self.indexed = True
                self.record(True, f"processing_status=completed ({chunks_indexed} passages indexed, ~{i}s)")
                return
            time.sleep(1)
        self.record(False, f"indexing did not finish within 60s (last processing_status={processing_status})")

    def search(self) -> None:
        self.poll_status()
        if self.file_id is None:
            return

        step("Semantic search (the assignment's own example)")
        resp = self.client.post(
            f"{API_URL}/files/{self.file_id}/search",
            headers={"X-User-Id": USER_A},
            json={"query": "database connectivity problems", "top_k": 3},
        )
        results = resp.json().get("results", [])
        top_text = results[0]["text"] if results else ""
        found = "Connection to database failed" in top_text
        self.record(
            found,
            f"query 'database connectivity problems' -> top hit is the DB error line (score={results[0]['score']})"
            if found
            else f"expected top search hit to mention the DB error, got: {top_text!r}",
        )

    def ownership_isolation(self) -> None:
        self.complete()
        if self.file_id is None:
            return

        step("Ownership isolation")
        resp = self.client.get(f"{API_URL}/files/{self.file_id}/status", headers={"X-User-Id": USER_B})
        self.record(
            resp.status_code == 404,
            "a different X-User-Id gets 404 for this file"
            if resp.status_code == 404
            else f"expected 404 for a non-owner, got HTTP {resp.status_code}",
        )

    def list_files(self) -> None:
        self.complete()
        if self.file_id is None:
            return

        step("List uploads for the owner")
        resp = self.client.get(f"{API_URL}/files", headers={"X-User-Id": USER_A})
        found = any(f["file_id"] == self.file_id for f in resp.json().get("files", []))
        self.record(found, "uploaded file appears in GET /files" if found else "uploaded file missing from GET /files")

    def delete(self) -> None:
        self.complete()
        if self.file_id is None:
            return

        step("Delete the file")
        resp = self.client.delete(f"{API_URL}/files/{self.file_id}", headers={"X-User-Id": USER_A})
        self.record(
            resp.status_code == 204,
            "DELETE returned 204" if resp.status_code == 204 else f"expected 204 from DELETE, got HTTP {resp.status_code}",
        )

        resp = self.client.get(f"{API_URL}/files/{self.file_id}/status", headers={"X-User-Id": USER_A})
        self.record(
            resp.status_code == 404,
            "file is gone after delete (404 on status)"
            if resp.status_code == 404
            else f"expected 404 after delete, got HTTP {resp.status_code}",
        )
        # A deleted file_id must not be reused by a later step this run.
        self.file_id = None
        self.bytes_sent = 0
        self.completed = False
        self.indexed = False

    def run_all(self) -> None:
        self.register()
        self.upload_with_interrupt()
        self.complete()
        self.poll_status()
        self.search()
        self.ownership_isolation()
        self.list_files()
        self.delete()

    def summary(self) -> None:
        print(f"\n== Summary: {self.passed} passed, {self.failed} failed ==")


MENU: list[tuple[str, str]] = [
    ("register", "Register a new upload"),
    ("upload_with_interrupt", "Upload in chunks, with a simulated mid-upload interruption"),
    ("complete", "Complete the upload"),
    ("poll_status", "Poll status until indexing catches up"),
    ("search", "Run the assignment's semantic search example"),
    ("ownership_isolation", "Confirm a different X-User-Id is refused the file"),
    ("list_files", "List uploads for the owner"),
    ("delete", "Delete the file and confirm it's gone"),
]


def start_compose() -> None:
    step("Starting docker compose")
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "--build"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        ok_health = False
    else:
        ok("stack started")
        ok_health = True
    return ok_health


def wait_for_health() -> bool:
    step("Waiting for the API to report healthy")
    for i in range(1, 61):
        try:
            resp = httpx.get(HEALTH_URL, timeout=5)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                ok(f"health check ok after ~{i * 2}s")
                return True
        except httpx.HTTPError:
            pass
        time.sleep(2)
    bad("service never became healthy within 120s")
    return False


def print_menu() -> None:
    print("\nWhat do you want to run?")
    for i, (_, label) in enumerate(MENU, start=1):
        print(f"  {i}. {label}")
    print("  a. Run all steps in order")
    print("  q. Quit")


def interactive_loop(session: Session) -> None:
    while True:
        print_menu()
        choice = input("> ").strip().lower()

        if choice in ("q", "quit", "exit"):
            break
        if choice in ("a", "all"):
            session.run_all()
            continue

        try:
            index = int(choice) - 1
            name, _ = MENU[index]
        except (ValueError, IndexError):
            print("  Not a valid choice, try again.")
            continue

        getattr(session, name)()

    session.summary()


def main() -> None:
    args = sys.argv[1:]

    if not SAMPLE_FILE.exists():
        bad(f"sample file missing at {SAMPLE_FILE}")
        sys.exit(1)

    if "--no-compose" not in args:
        if not start_compose():
            sys.exit(1)

    if not wait_for_health():
        sys.exit(1)

    info(f"sample file: {SAMPLE_FILE} ({SAMPLE_FILE.stat().st_size} bytes)")

    session = Session()

    if "--all" in args:
        session.run_all()
        session.summary()
        sys.exit(1 if session.failed else 0)

    interactive_loop(session)
    sys.exit(1 if session.failed else 0)


if __name__ == "__main__":
    main()
