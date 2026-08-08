# Third-party notices

The Python dependencies distributed by this project retain their own licenses. Exact
versions are recorded in `uv.lock`; their project metadata is the authoritative license
source.

FFmpeg is **not bundled** in the source, wheel, or current standalone candidate. FFmpeg
is an independent project available from [ffmpeg.org](https://ffmpeg.org/). Its license
depends on build configuration. Automated download remains disabled until each target
archive has an exact version, SHA-256, distributor record, build-license review, and
corresponding-source notice in `packaging/media-runtime-manifest.json`.
