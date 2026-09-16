#!/usr/bin/env python3
"""Interactive end-to-end test against a *live* running instance of the
service -- not pytest, and not mocked. Exercises the real HTTP API exactly
as a client would.

Usage:
    python3 scripts/e2e_test.py               # interactive menu (default)
    python3 scripts/e2e_test.py --all          # run every step once, no menu
    python3 scripts/e2e_test.py --no-compose   # assume the API is already running

Every menu choice always re-runs its step fresh -- picking the same number
twice in a row does the whole thing again rather than silently no-op'ing,
which also happens to exercise the API's own idempotency guarantees (e.g.
completing an already-completed upload, or checking status on a deleted
file) instead of hiding them behind a skip. A step that needs an upload in
flight creates one automatically if none exists yet.

Each step prints one [PASS]/[FAIL] line per assertion plus a short stats
line (byte/chunk/passage counts, elapsed time, HTTP status) so a human
watching the terminal gets more than just pass/fail. The search step also
prints the actual matched passages (rank, score, byte range, text) it got
back, not just whether the top one was the expected one.

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


def stats(**fields: object) -> None:
    """One compact line of extra numbers/state after a step's PASS/FAIL lines."""
    rendered = ", ".join(f"{k}={v}" for k, v in fields.items())
    print(f"  stats: {rendered}")


def preview_text(text: str, highlight: str | None = None, width: int = 220) -> str:
    """Shorten a passage for terminal display.

    A passage can span many log lines, so a plain head-truncation can cut
    off before the one line that actually explains why this hit matched
    (e.g. the DB error line, well past character 200 in a multi-line
    passage). When `highlight` appears in the text, center the window on
    it instead of always showing the start.
    """
    text = text.strip()
    if len(text) <= width:
        return text

    if highlight and highlight in text:
        idx = text.index(highlight)
        start = max(0, idx - width // 3)
        end = min(len(text), start + width)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{text[start:end]}{suffix}"

    return text[:width] + "..."


class Session:
    """Live state for one run: the HTTP client and the most recently
    created file_id. Every step always performs its real work again when
    invoked -- it does not skip based on prior state -- but will create an
    upload first if one doesn't exist yet, so any step can be picked on its
    own from a clean session."""

    def __init__(self) -> None:
        self.client = httpx.Client(timeout=30)
        self.data = SAMPLE_FILE.read_bytes() if SAMPLE_FILE.exists() else b""
        self.file_size = len(self.data)
        self.file_id: str | None = None
        self.bytes_sent = 0
        # Tracks whether the CURRENT file_id has been through complete()/
        # poll_status() at least once, purely so ensure_uploaded_and_completed()
        # (an implicit prerequisite check other steps make) doesn't redo work
        # a menu pick already did earlier in the same run_all()/session. A
        # step invoked directly from the menu always redoes its own work
        # regardless of these flags -- only the *helper* consults them.
        self.completed_once = False
        self.indexed_once = False
        self.passed = 0
        self.failed = 0
        self.run_count = 0  # how many times any step has been invoked, for the summary

    def record(self, success: bool, msg: str) -> None:
        if success:
            self.passed += 1
            ok(msg)
        else:
            self.failed += 1
            bad(msg)

    def ensure_uploaded_and_completed(self) -> None:
        """Prerequisite helper for steps that need a completed upload to act
        on. Only does the work this session hasn't already done for the
        current file_id -- so run_all() and a step's own implicit
        prerequisites don't reprint "Complete the upload" on every single
        later step. Picking "complete" directly from the menu bypasses this
        entirely and always redoes the real work (see complete() itself)."""
        if self.file_id is None:
            self.register()
        if self.bytes_sent < self.file_size:
            self.upload_with_interrupt()
        if not self.completed_once:
            self.complete()

    def ensure_indexed(self) -> None:
        """Prerequisite helper: only polls if this file_id hasn't already
        been confirmed fully indexed this session."""
        self.ensure_uploaded_and_completed()
        if not self.indexed_once:
            self.poll_status()

    # ---- steps --------------------------------------------------------

    def register(self) -> None:
        """Always creates a brand-new upload, replacing the session's
        current file_id -- picking this again is meant to register another
        file, not confirm the old one still exists."""
        self.run_count += 1
        step("Register upload")
        t0 = time.monotonic()
        resp = self.client.post(
            f"{API_URL}/files",
            headers={"X-User-Id": USER_A},
            json={"filename": "sample.log", "total_size": self.file_size},
        )
        elapsed = time.monotonic() - t0
        if resp.status_code != 201:
            self.record(False, f"POST /files returned {resp.status_code}: {resp.text}")
            return
        body = resp.json()
        self.file_id = body["file_id"]
        self.bytes_sent = 0
        self.completed_once = False
        self.indexed_once = False
        self.record(True, f"created upload, file_id={self.file_id}")
        stats(
            http_status=resp.status_code,
            chunk_size_limit=body.get("chunk_size"),
            total_size_declared=body.get("total_size"),
            elapsed_s=round(elapsed, 3),
        )

    def upload_with_interrupt(self) -> None:
        """Always (re-)uploads the sample file from byte 0 against the
        current file_id -- if it's already fully uploaded, this registers a
        fresh file first so there is always real work to do."""
        self.run_count += 1
        if self.file_id is None or self.bytes_sent >= self.file_size:
            self.register()
        if self.file_id is None:
            return

        step("Chunked upload with a simulated interruption")
        t0 = time.monotonic()
        offset = 0
        chunk_index = 0
        interrupted = False
        interrupt_checked_ok = False
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
                interrupt_checked_ok = reported_offset == offset
                self.record(
                    interrupt_checked_ok,
                    f"interrupted after {offset} bytes; /status reports {reported_offset}"
                    if interrupt_checked_ok
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

        elapsed = time.monotonic() - t0
        self.bytes_sent = offset
        chunks_enqueued = resp.json().get("chunks_enqueued")
        self.record(True, f"uploaded {self.file_size} bytes across {chunk_index} chunks")
        stats(
            chunks_sent=chunk_index,
            bytes_uploaded=offset,
            avg_chunk_bytes=round(offset / chunk_index) if chunk_index else 0,
            passages_enqueued_total=chunks_enqueued,
            interrupt_resume_verified=interrupt_checked_ok,
            elapsed_s=round(elapsed, 3),
        )

    def complete(self) -> None:
        """Always calls POST .../complete against the current file_id, even
        if it was already completed -- that's a real idempotency check
        (the API must return upload_status=completed again, not error)
        rather than something this script should hide by skipping."""
        self.run_count += 1
        if self.file_id is None:
            self.register()
        if self.file_id is not None and self.bytes_sent < self.file_size:
            self.upload_with_interrupt()
        if self.file_id is None:
            return

        step("Complete the upload")
        t0 = time.monotonic()
        resp = self.client.post(f"{API_URL}/files/{self.file_id}/complete", headers={"X-User-Id": USER_A})
        elapsed = time.monotonic() - t0
        body = resp.json()
        upload_status = body.get("upload_status")
        success = upload_status == "completed"
        self.record(
            success,
            "upload_status=completed"
            if success
            else f"expected upload_status=completed, got '{upload_status}' (response: {resp.text})",
        )

        # Call it again immediately: completing an already-completed upload
        # must be a safe no-op, not an error, per app/services/upload_service.py.
        resp2 = self.client.post(f"{API_URL}/files/{self.file_id}/complete", headers={"X-User-Id": USER_A})
        idempotent = resp2.status_code == 200 and resp2.json().get("upload_status") == "completed"
        self.record(
            idempotent,
            "calling complete a second time is idempotent (still upload_status=completed)"
            if idempotent
            else f"second complete() call was not idempotent: HTTP {resp2.status_code}, {resp2.text}",
        )
        stats(
            http_status=resp.status_code,
            bytes_received=body.get("bytes_received"),
            total_size_final=body.get("total_size"),
            chunks_total=body.get("chunks_total"),
            elapsed_s=round(elapsed, 3),
        )
        self.completed_once = success

    def poll_status(self) -> None:
        """Always polls fresh -- re-picking this after indexing already
        finished just confirms the terminal state again, printing the
        latest counts rather than skipping the HTTP calls entirely."""
        self.run_count += 1
        self.ensure_uploaded_and_completed()
        if self.file_id is None:
            return

        step("Poll status until indexing catches up")
        t0 = time.monotonic()
        processing_status = None
        body: dict = {}
        for i in range(1, 61):
            resp = self.client.get(f"{API_URL}/files/{self.file_id}/status", headers={"X-User-Id": USER_A})
            body = resp.json()
            processing_status = body.get("processing_status")
            if processing_status == "completed":
                break
            time.sleep(1)
        elapsed = time.monotonic() - t0

        success = processing_status == "completed"
        self.record(
            success,
            f"processing_status=completed ({body.get('chunks_indexed', 0)} passages indexed, ~{round(elapsed)}s)"
            if success
            else f"indexing did not finish within 60s (last processing_status={processing_status})",
        )
        stats(
            chunks_total=body.get("chunks_total"),
            chunks_indexed=body.get("chunks_indexed"),
            chunks_failed=body.get("chunks_failed"),
            processing_progress=body.get("processing_progress"),
            searchable=body.get("searchable"),
            poll_attempts=i,
            elapsed_s=round(elapsed, 3),
        )
        self.indexed_once = success

    def search(self) -> None:
        """Always re-issues the search query, even if already run this
        session -- confirms the same result is returned consistently."""
        self.run_count += 1
        self.ensure_indexed()
        if self.file_id is None:
            return

        step("Semantic search (the assignment's own example)")
        t0 = time.monotonic()
        resp = self.client.post(
            f"{API_URL}/files/{self.file_id}/search",
            headers={"X-User-Id": USER_A},
            json={"query": "database connectivity problems", "top_k": 3},
        )
        elapsed = time.monotonic() - t0
        body = resp.json()
        results = body.get("results", [])
        top_text = results[0]["text"] if results else ""
        found = "Connection to database failed" in top_text
        self.record(
            found,
            f"query 'database connectivity problems' -> top hit is the DB error line (score={results[0]['score']:.4f})"
            if found
            else f"expected top search hit to mention the DB error, got: {top_text!r}",
        )

        info(f"query: {body.get('query')!r}")
        for rank, hit in enumerate(results, start=1):
            text = preview_text(hit["text"], highlight="Connection to database failed")
            info(
                f"  #{rank} score={hit['score']:.4f} sequence={hit['sequence']} "
                f"bytes=[{hit['start_byte']}:{hit['end_byte']}]"
            )
            info(f"      text: {text!r}")

        scores = [f"{r['score']:.3f}" for r in results]
        stats(
            http_status=resp.status_code,
            total_hits=body.get("total_hits"),
            top_k_requested=3,
            scores=scores,
            elapsed_s=round(elapsed, 3),
        )

    def ownership_isolation(self) -> None:
        """Always re-checks that a second identity is refused the file."""
        self.run_count += 1
        self.ensure_uploaded_and_completed()
        if self.file_id is None:
            return

        step("Ownership isolation")
        t0 = time.monotonic()
        owner_resp = self.client.get(f"{API_URL}/files/{self.file_id}/status", headers={"X-User-Id": USER_A})
        other_resp = self.client.get(f"{API_URL}/files/{self.file_id}/status", headers={"X-User-Id": USER_B})
        elapsed = time.monotonic() - t0
        success = owner_resp.status_code == 200 and other_resp.status_code == 404
        self.record(
            success,
            "owner sees the file (200), a different X-User-Id gets 404"
            if success
            else f"expected owner=200/other=404, got owner={owner_resp.status_code}/other={other_resp.status_code}",
        )
        stats(
            owner_user=USER_A,
            other_user=USER_B,
            owner_http_status=owner_resp.status_code,
            other_http_status=other_resp.status_code,
            elapsed_s=round(elapsed, 3),
        )

    def list_files(self) -> None:
        """Always re-lists -- reports how many files this owner currently has,
        not just whether the one we care about is among them."""
        self.run_count += 1
        self.ensure_uploaded_and_completed()
        if self.file_id is None:
            return

        step("List uploads for the owner")
        t0 = time.monotonic()
        resp = self.client.get(f"{API_URL}/files", headers={"X-User-Id": USER_A})
        elapsed = time.monotonic() - t0
        files = resp.json().get("files", [])
        found = any(f["file_id"] == self.file_id for f in files)
        self.record(found, "uploaded file appears in GET /files" if found else "uploaded file missing from GET /files")
        stats(
            http_status=resp.status_code,
            total_files_for_owner=len(files),
            elapsed_s=round(elapsed, 3),
        )

    def delete(self) -> None:
        """Always deletes the current file_id, then always tries the delete
        a second time too -- deleting an already-deleted (i.e. unknown)
        file_id must 404, not 500 or succeed twice."""
        self.run_count += 1
        self.ensure_uploaded_and_completed()
        if self.file_id is None:
            return
        deleted_id = self.file_id

        step("Delete the file")
        t0 = time.monotonic()
        resp = self.client.delete(f"{API_URL}/files/{deleted_id}", headers={"X-User-Id": USER_A})
        first_delete_ok = resp.status_code == 204
        self.record(
            first_delete_ok,
            "DELETE returned 204" if first_delete_ok else f"expected 204 from DELETE, got HTTP {resp.status_code}",
        )

        status_resp = self.client.get(f"{API_URL}/files/{deleted_id}/status", headers={"X-User-Id": USER_A})
        gone = status_resp.status_code == 404
        self.record(
            gone,
            "file is gone after delete (404 on status)"
            if gone
            else f"expected 404 after delete, got HTTP {status_resp.status_code}",
        )

        # Deleting the same, now-unknown file_id again must 404, not error.
        redelete_resp = self.client.delete(f"{API_URL}/files/{deleted_id}", headers={"X-User-Id": USER_A})
        redelete_ok = redelete_resp.status_code == 404
        self.record(
            redelete_ok,
            "deleting an already-deleted file_id 404s (idempotent, not an error)"
            if redelete_ok
            else f"expected 404 re-deleting a gone file, got HTTP {redelete_resp.status_code}",
        )
        elapsed = time.monotonic() - t0

        stats(
            deleted_file_id=deleted_id,
            first_delete_status=resp.status_code,
            status_after_delete=status_resp.status_code,
            redelete_status=redelete_resp.status_code,
            elapsed_s=round(elapsed, 3),
        )

        # This file_id is gone; a later step needs a new one.
        self.file_id = None
        self.bytes_sent = 0

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
        print(f"\n== Summary: {self.passed} passed, {self.failed} failed ({self.run_count} steps run) ==")


MENU: list[tuple[str, str]] = [
    ("register", "Register a new upload (always creates a fresh file_id)"),
    ("upload_with_interrupt", "Upload in chunks, with a simulated mid-upload interruption"),
    ("complete", "Complete the upload (also verifies calling it twice is idempotent)"),
    ("poll_status", "Poll status until indexing catches up"),
    ("search", "Run the assignment's semantic search example"),
    ("ownership_isolation", "Confirm a different X-User-Id is refused the file"),
    ("list_files", "List uploads for the owner"),
    ("delete", "Delete the file (also verifies re-deleting 404s, not errors)"),
]


def start_compose() -> bool:
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
        bad("docker compose up failed")
        return False
    ok("stack started")
    return True


def wait_for_health() -> bool:
    step("Waiting for the API to report healthy")
    for i in range(1, 61):
        try:
            resp = httpx.get(HEALTH_URL, timeout=5)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                ok(f"health check ok after ~{i * 2}s")
                stats(**{k: v for k, v in resp.json().items()})
                return True
        except httpx.HTTPError:
            pass
        time.sleep(2)
    bad("service never became healthy within 120s")
    return False


def print_menu() -> None:
    print("\nWhat do you want to run? (always re-runs fresh, even if picked before)")
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
            if index < 0:
                raise ValueError
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
