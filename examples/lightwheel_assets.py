from collections import defaultdict

from lightwheel_sdk.loader import (
    floorplan_loader,
    login_manager,
    object_loader,
    scene_loader,
)

# List all available assets in the Lightwheel registry
registry_data = object_loader.list_registry()

# Group by registryType
by_type = defaultdict(list)
for item in registry_data:
    by_type[item["registryType"]].append(item["name"])

print(f"=== Lightwheel Registry: {len(registry_data)} total assets ===\n")
for rtype, names in sorted(by_type.items()):
    print(f"--- {rtype} ({len(names)} assets) ---")
    for name in sorted(names):
        print(f"  {name}")
    print()

# Load an object
file_path, object_name, metadata = object_loader.acquire_by_registry(
    registry_type="objects",
    registry_name=["alphabet_soup"],
    file_type="USD"
)