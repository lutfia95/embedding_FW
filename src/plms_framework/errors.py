class FrameworkError(Exception):
    """Base exception with a stable machine-readable code."""

    code = "framework_error"

    def __init__(self, message: str, *, context: dict | None = None) -> None:
        super().__init__(message)
        self.context = context or {}

    def as_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "context": self.context}


class ValidationError(FrameworkError):
    code = "validation_error"


class CapabilityError(FrameworkError):
    code = "unsupported_capability"


class ModelUnavailableError(FrameworkError):
    code = "model_unavailable"


class DependencyUnavailableError(FrameworkError):
    code = "dependency_missing"


class ResourceLimitError(FrameworkError):
    code = "resource_limit"
