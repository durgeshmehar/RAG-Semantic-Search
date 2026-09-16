"""Domain exceptions, decoupled from HTTP.

Services raise these; app/main.py's exception handlers are the single place
that translates them into HTTP status codes and response bodies. Nothing
below imports FastAPI -- a service can be called and its errors caught by
another service, a script, or a test with no HTTP framework involved, and the
HTTP mapping lives in exactly one place instead of being repeated at every
raise site across every router.
"""


class DomainError(Exception):
    """Base for every error a service can raise."""


class FileNotFound(DomainError):
    """No file with this id, or it exists but belongs to a different owner.

    Deliberately one exception for both cases (see sql_repository.files.get_owned):
    returning the same error for "doesn't exist" and "not yours" stops a
    client from telling the two apart by probing ids.
    """

    def __init__(self, file_id: str) -> None:
        self.file_id = file_id
        super().__init__(f"unknown file_id: {file_id}")


class UploadAlreadyDone(DomainError):
    """A chunk arrived for an upload that's already completed or finalizing."""

    def __init__(self, upload_status: str) -> None:
        self.upload_status = upload_status
        super().__init__(f"upload already {upload_status}")


class OffsetMismatch(DomainError):
    """The client's offset doesn't match how many bytes the server has.

    Carries the correct offset so the caller can tell the client exactly
    where to resume rather than just refusing.
    """

    def __init__(self, expected: int, got: int) -> None:
        self.expected = expected
        self.got = got
        super().__init__(
            f"offset mismatch: expected {expected}, got {got}. Resume from {expected}."
        )


class ChunkTooLarge(DomainError):
    def __init__(self, size: int, limit: int) -> None:
        self.size = size
        self.limit = limit
        super().__init__(f"chunk of {size} bytes exceeds limit of {limit}")


class FileTooLarge(DomainError):
    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"upload would exceed the maximum allowed size of {limit} bytes")


class NotTextFile(DomainError):
    def __init__(self) -> None:
        super().__init__(
            "this looks like a binary file, not text -- this service "
            "only supports UTF-8 text files"
        )


class NoBytesReceivedYet(DomainError):
    def __init__(self) -> None:
        super().__init__("no bytes received yet")


class TooManyConcurrentUploads(DomainError):
    """Raised by the upload limiter when no slot is free.

    Carries retry_after so the HTTP layer can set the header without needing
    to know the limiter's internal constant.
    """

    def __init__(self, limit: int, retry_after_seconds: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"server is handling {limit} uploads already; retry shortly"
        )


class NothingIndexedYet(DomainError):
    def __init__(self) -> None:
        super().__init__("no passages indexed yet; poll /status until searchable is true")
