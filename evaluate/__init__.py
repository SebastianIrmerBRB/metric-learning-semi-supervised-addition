"""Readers that turn epoch-tracking output into tables and pictures.

``training/epoch_tracking.py`` writes one directory per fold. A thesis chapter
needs forty of them in one table, so everything here reads many runs at once and
writes to a single output tree. Nothing in this package trains, and nothing it
reads is written back into a run.
"""
