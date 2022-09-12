class SymbolicException(object):

    @staticmethod
    def mismatch(x: str, y: str = "(not found)") -> Exception:
        return Exception("No match between variable name " + x + " and feature name " + y + ".")

    @staticmethod
    def not_supported(x: str) -> Exception:
        return Exception("Operation " + x + " is not supported")
