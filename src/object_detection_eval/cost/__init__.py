"""Cost-per-frame study: aggregate per-call usage logs into USD and seconds per frame.

Torch-free. ``schemas`` declares every file format, ``compute`` turns usage logs and
run manifests into ``cost_per_frame.json``, ``report`` renders that JSON (and only that
JSON) to Markdown and a PNG, and ``frames`` / ``rerun`` hold the val-frame selection and
the rerun driver's engine.
"""
