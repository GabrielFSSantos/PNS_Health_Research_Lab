from __future__ import annotations


class PNSHealthMLError(Exception):
    """Base error."""


class MissingDatasetError(PNSHealthMLError):
    """Required dataset file not found."""


class MissingRequiredColumnError(PNSHealthMLError):
    """Schema validation failed."""


class InvalidExperimentConfigError(PNSHealthMLError):
    """Invalid experiment configuration."""


class ArtifactPersistenceError(PNSHealthMLError):
    """Failed to persist artifacts."""
