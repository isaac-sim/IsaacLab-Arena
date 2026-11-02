# Visual assets

The small PNGs are the exact visual inputs used by the reference production.
They retain their original owners' branding. The repository's code license does
not grant rights to third-party trademarks or fonts.

| Asset | Source |
| --- | --- |
| Isaac Lab icon | IsaacLab `docs/source/_static/favicon.ico`, pinned revision and URL in `assets/isaaclab_icon_source.json`; converted to PNG |
| Codex icon | OpenAI's published VS Code extension icon; URL and checksum in `assets/branding_sources.json` |
| Physical Intelligence logo | Physical-Intelligence GitHub organization avatar, requested at 512 px (returned 392×392); source/checksum in the manifest |
| Omniverse icon | NVIDIA's installed Kit filepicker `icons/NvidiaLight/omniverse_logo.svg`, rasterized to 512×512; original provenance/checksum in the manifest |
| NVIDIA Sans regular/bold | Downloaded by `fetch_fonts.py` from the URLs in the manifest; fonts are not bundled |
| Eight scene thumbnails | Isolated Isaac Sim RTX renders of the actual USD assets; USD paths, scales, camera poses, bounding boxes, and alpha-mask method in `assets/rendered/sources.json` |

The Omniverse provenance names its installed extension source, not a runtime
dependency. The bundled PNG is used directly. To change branding,
replace the corresponding PNG and update its provenance. Keep a transparent
background where appropriate.

`render_asset_thumbnails.py` can recreate the scene thumbnails. It renders each
asset on its own, masks the background using finite foreground depth, checks that
the object is fully in frame, and writes a centered 512×512 RGBA image. It does
not crop assets from the composed kitchen screenshot. Kitchen surface references
are rendered while preserving their source transforms and materials.

For another environment, update the thumbnail script's USD specifications,
scales, prim references, and robot pose. Match the registered assets used by the
environment; do not substitute visually similar generated images. The existing
thumbnails are reference assets from the earlier recording, not evidence that
newly downloaded USDs have remained unchanged.
