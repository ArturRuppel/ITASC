"""Quantifier plugins.

Dropping a ``*.py`` module here whose class subclasses
:class:`cellflow.contact_analysis.quantifier.Quantifier` (with a non-empty
``quantity_id``) registers a new per-position quantity. ``available_quantifiers``
imports every module here so they self-register.
"""
