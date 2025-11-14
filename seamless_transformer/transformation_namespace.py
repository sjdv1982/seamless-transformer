"""Namespace utilities for transformer execution (currently stubbed)."""


def build_transformation_namespace_sync(transformation, semantic_cache, codename):
    """Port placeholder for future implementation.

    The original Seamless implementation populates the namespace by resolving
    inputs, modules and code from the checksum-based transformation dict.
    For the transformer port this functionality still needs to be migrated,
    so the function raises NotImplementedError for now.
    """

    raise NotImplementedError("build_transformation_namespace_sync pending port")


__all__ = ["build_transformation_namespace_sync"]
