class KnowledgeException(object):
    @staticmethod
    def mismatch(x: str, y: str = "(not found)") -> Exception:
        return Exception(
            "No match between variable name " + x + " and feature name " + y + "."
        )

    @staticmethod
    def not_supported(x: str) -> Exception:
        return Exception("Operation " + x + " is not supported")

    @staticmethod
    def not_parsable(x: str, e: Exception) -> Exception:
        return Exception(
            "The logic program:\n\n" + x + "\n\nis not parsable.\n\n" + str(e)
        )
