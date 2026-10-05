"""errors: what the ingestion pipeline raises so the worker can tell a bad DOCUMENT from a bad MOMENT.

DocumentRejected means this document cannot be ingested and never will be as it is (the extraction was too
empty, a table was lost, nothing readable was left): the file goes to failed/ with the reason and the message is
acknowledged. Any other exception is treated as a problem of the moment (a model, the database, Tesseract) and the
message is given back for a retry."""


class DocumentRejected(Exception):
    """The document was read but cannot be ingested; the reason is for the review list."""

    def __init__(self, reason_code: str, reason: str, details: dict = None):
        super().__init__(reason)
        self.reason_code = reason_code
        self.reason = reason
        self.details = details or {}
