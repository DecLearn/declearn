# coding: utf-8

"""Communication flags used in declearn communication backends."""


# Registration flags.
REGISTRATION_UNSTARTED = "registration is not opened yet"
REGISTRATION_OPEN = "registration is open"
REGISTRATION_CLOSED = "registration is closed"
REGISTERED_WELCOME = "welcome, you are now registered"
REGISTERED_ALREADY = "you were already registered"

# Error flags.
CHECK_MESSAGE_TIMEOUT = "no available message at timeout"
INVALID_MESSAGE = "invalid message"
REJECT_UNREGISTERED = "rejected: not a registered user"
