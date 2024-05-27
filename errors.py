class AdversarialAttackException(Exception):
    """Base Exception for AdversarialAttack"""

    def __init__(self, message):
        super().__init__(message)


class ImageException(AdversarialAttackException):
    """Exception raised for image exclusion errors"""

    def __init__(self, message):
        super().__init__(message)

