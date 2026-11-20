# YAM workcell table layers

`industrial__yam_workcell_table.usda` adds the cable-routing placement surface
to Isaac Sim's Stainless Metal Table A03 asset. Its sibling closure layer keeps
the imported table static and references the public Isaac Sim 6.0 SimReady
asset on NVIDIA's content CDN.

The two authored USDA layers are bundled with Arena. The referenced table
geometry is downloaded by Isaac Sim at runtime, so first use requires network
access or an already populated asset cache. Use of the referenced NVIDIA asset
is governed by the [NVIDIA Software License Agreement and NVIDIA AI Product
Specific Terms](https://docs.omniverse.nvidia.com/services/latest/common/NVIDIA_Omniverse_License_Agreement.html).
