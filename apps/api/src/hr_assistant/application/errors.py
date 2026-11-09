class ApplicationError(Exception):
    status_code = 400


class NotFoundError(ApplicationError):
    status_code = 404


class ConflictError(ApplicationError):
    status_code = 409


class ServiceUnavailableError(ApplicationError):
    status_code = 503
