# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script sets up the PyCharm settings for the Isaac Lab project.

This script creates/updates the PyCharm configuration files to include the necessary Python paths
from Isaac Sim and Isaac Lab extensions.

This is necessary because Isaac Sim 2022.2.1 onwards requires explicit path configuration
for proper code completion and navigation in PyCharm.
"""

import re
import sys
import os
import pathlib
import xml.etree.ElementTree as ET
from xml.dom import minidom


ISAACLAB_ARENA_DIR = pathlib.Path(__file__).parents[2]
ISAACLAB_DIR = ISAACLAB_ARENA_DIR / "submodules" / "IsaacLab"
"""Path to the Isaac Lab directory."""

try:
    import isaacsim  # noqa: F401

    isaacsim_dir = os.environ.get("ISAAC_PATH", "")
except ModuleNotFoundError or ImportError:
    isaacsim_dir = os.path.join(ISAACLAB_DIR, "_isaac_sim")
except EOFError:
    print("Unable to trigger EULA acceptance. This is likely due to the script being run in a non-interactive shell.")
    print("Please run the script in an interactive shell to accept the EULA.")
    print("Skipping the setup of the PyCharm settings...")
    sys.exit(0)

# check if the isaac-sim directory exists
if not os.path.exists(isaacsim_dir):
    raise FileNotFoundError(
        f"Could not find the isaac-sim directory: {isaacsim_dir}. There are two possible reasons for this:"
        f"\n\t1. The Isaac Sim directory does not exist as a symlink at: {os.path.join(ISAACLAB_DIR, '_isaac_sim')}"
        "\n\t2. The script could not import the 'isaacsim' package. This could be due to the 'isaacsim' package not "
        "being installed in the Python environment.\n"
        "\nPlease make sure that the Isaac Sim directory exists or that the 'isaacsim' package is installed."
    )

ISAACSIM_DIR = isaacsim_dir
"""Path to the isaac-sim directory."""


def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def get_python_paths_from_isaacsim():
    """Extract python paths from Isaac Sim's VSCode settings.

    Returns:
        List of absolute paths to add to Python path.
    """
    # isaac-sim settings
    isaacsim_vscode_filename = os.path.join(ISAACSIM_DIR, ".vscode", "settings.json")

    # we use the isaac-sim settings file to get the python.analysis.extraPaths for kit extensions
    # if this file does not exist, we will not add any extra paths
    if os.path.exists(isaacsim_vscode_filename):
        # read the path names from the isaac-sim settings file
        with open(isaacsim_vscode_filename) as f:
            vscode_settings = f.read()
        # extract the path names
        # search for the python.analysis.extraPaths section and extract the contents
        settings = re.search(
            r"\"python.analysis.extraPaths\": \[.*?\]", vscode_settings, flags=re.MULTILINE | re.DOTALL
        )
        if settings:
            settings = settings.group(0)
            settings = settings.split('"python.analysis.extraPaths": [')[-1]
            settings = settings.split("]")[0]

            # read the path names from the isaac-sim settings file
            path_names = settings.split(",")
            path_names = [path_name.strip().strip('"') for path_name in path_names]
            path_names = [path_name for path_name in path_names if len(path_name) > 0]

            # convert to absolute paths
            path_names = [os.path.abspath(os.path.join(ISAACSIM_DIR, path_name)) for path_name in path_names]
        else:
            path_names = []
    else:
        path_names = []
        print(
            f"[WARN] Could not find Isaac Sim VSCode settings: {isaacsim_vscode_filename}."
            "\n\tThis will result in missing Python paths in the PyCharm configuration,"
            "\n\twhich limits the functionality of code completion and navigation."
            "\n\tHowever, it does not affect the functionality of the Isaac Lab project."
            "\n\tWe are working on a fix for this issue with the Isaac Sim team."
        )

    # add the path names that are in the Isaac Lab extensions directory
    isaaclab_extensions = os.listdir(os.path.join(ISAACLAB_DIR, "source"))
    path_names.extend([os.path.abspath(os.path.join(ISAACLAB_DIR, "source", ext)) for ext in isaaclab_extensions])

    return path_names


def create_or_update_misc_xml():
    """Create or update .idea/misc.xml with the Python interpreter path."""
    misc_xml_filename = os.path.join(ISAACLAB_ARENA_DIR, ".idea", "misc.xml")
    misc_xml_template_filename = os.path.join(ISAACLAB_ARENA_DIR, ".idea", "tools", "misc.template.xml")

    # Ensure .idea directory exists
    os.makedirs(os.path.dirname(misc_xml_filename), exist_ok=True)

    # Read executable name
    python_exe = sys.executable

    # Check if template exists, if so use it as base
    if os.path.exists(misc_xml_template_filename):
        tree = ET.parse(misc_xml_template_filename)
        root = tree.getroot()
    elif os.path.exists(misc_xml_filename):
        tree = ET.parse(misc_xml_filename)
        root = tree.getroot()
    else:
        root = ET.Element("project", version="4")

    # Find or create ProjectRootManager component
    project_root_manager = root.find(".//component[@name='ProjectRootManager']")
    if project_root_manager is None:
        project_root_manager = ET.SubElement(root, "component", name="ProjectRootManager")

    # We make an exception for replacing the default interpreter if the
    # path (/kit/python/bin/python3) indicates that we are using a local/container
    # installation of IsaacSim.
    if "kit/python/bin/python3" not in python_exe:
        project_root_manager.set("version", "2")
        project_root_manager.set("project-jdk-name", "Python")
        project_root_manager.set("project-jdk-type", "Python SDK")

    # Add header comment
    header_message = (
        "<!--\n"
        "  This file is automatically generated by the setup_pycharm.py script.\n"
        "  Do not edit this file directly.\n"
        f"  Generated from: {misc_xml_template_filename}\n"
        "-->\n"
    )

    # Write the file
    xml_string = prettify_xml(root)
    # Insert header after XML declaration
    xml_lines = xml_string.split('\n')
    if xml_lines[0].startswith('<?xml'):
        xml_string = xml_lines[0] + '\n' + header_message + '\n'.join(xml_lines[1:])
    else:
        xml_string = header_message + xml_string

    with open(misc_xml_filename, "w") as f:
        f.write(xml_string)

    print(f"[INFO] Updated PyCharm interpreter configuration: {misc_xml_filename}")


def create_pycharm_libraries_xml(python_paths):
    """Create libraries configuration for additional Python paths.

    Args:
        python_paths: List of absolute paths to add as library roots.
    """
    libraries_dir = os.path.join(ISAACLAB_ARENA_DIR, ".idea", "libraries")
    os.makedirs(libraries_dir, exist_ok=True)

    # Create a library XML file
    library_filename = os.path.join(libraries_dir, "Isaac_Sim_Paths.xml")

    root = ET.Element("component", name="libraryTable")
    library = ET.SubElement(root, "library", name="Isaac Sim Paths")
    classes = ET.SubElement(library, "CLASSES")

    for path in python_paths:
        if os.path.exists(path):
            # Convert to file URL format
            file_url = f"file://{path}"
            ET.SubElement(classes, "root", url=file_url)

    ET.SubElement(library, "JAVADOC")
    ET.SubElement(library, "SOURCES")

    # Write the file
    xml_string = prettify_xml(root)
    with open(library_filename, "w") as f:
        f.write(xml_string)

    print(f"[INFO] Created PyCharm library configuration: {library_filename}")


def copy_run_configurations():
    """Copy run configuration templates from tools directory."""
    run_configs_dir = os.path.join(ISAACLAB_ARENA_DIR, ".idea", "runConfigurations")
    run_configs_template_dir = os.path.join(ISAACLAB_ARENA_DIR, ".idea", "tools", "runConfigurations")
    
    os.makedirs(run_configs_dir, exist_ok=True)

    # Check if template directory exists
    if not os.path.exists(run_configs_template_dir):
        print(f"[WARN] Run configuration templates directory not found: {run_configs_template_dir}")
        return

    # Copy all template run configurations
    template_files = [f for f in os.listdir(run_configs_template_dir) if f.endswith('.xml')]
    
    for template_file in template_files:
        template_path = os.path.join(run_configs_template_dir, template_file)
        target_path = os.path.join(run_configs_dir, template_file)
        
        # Don't overwrite if it already exists
        if os.path.exists(target_path):
            print(f"[INFO] Run configuration already exists: {template_file}")
            continue
        
        # Read template and add header
        with open(template_path) as f:
            content = f.read()
        
        # Add header comment
        header_message = (
            "<!--\n"
            "  This file is automatically generated by the setup_pycharm.py script.\n"
            "  Do not edit this file directly.\n"
            f"  Generated from: {template_path}\n"
            "-->\n"
        )
        
        # Insert header after XML declaration
        lines = content.split('\n')
        if lines[0].startswith('<?xml'):
            content = lines[0] + '\n' + header_message + '\n'.join(lines[1:])
        else:
            content = header_message + content
        
        # Write to target
        with open(target_path, "w") as f:
            f.write(content)
        
        print(f"[INFO] Created run configuration: {template_file}")


def copy_additional_configurations():
    """Copy additional configuration files from templates."""
    # Copy workspace.xml if it doesn't exist
    workspace_xml_filename = os.path.join(ISAACLAB_ARENA_DIR, ".idea", "workspace.xml")
    workspace_xml_template_filename = os.path.join(ISAACLAB_ARENA_DIR, ".idea", "tools", "workspace.template.xml")
    
    if not os.path.exists(workspace_xml_filename) and os.path.exists(workspace_xml_template_filename):
        with open(workspace_xml_template_filename) as f:
            content = f.read()
        
        header_message = (
            "<!--\n"
            "  This file is automatically generated by the setup_pycharm.py script.\n"
            "  Do not edit this file directly.\n"
            f"  Generated from: {workspace_xml_template_filename}\n"
            "-->\n"
        )
        
        lines = content.split('\n')
        if lines[0].startswith('<?xml'):
            content = lines[0] + '\n' + header_message + '\n'.join(lines[1:])
        else:
            content = header_message + content
        
        with open(workspace_xml_filename, "w") as f:
            f.write(content)
        
        print(f"[INFO] Created workspace configuration: {workspace_xml_filename}")
    
    # Copy code style configurations
    code_styles_dir = os.path.join(ISAACLAB_ARENA_DIR, ".idea", "codeStyles")
    code_styles_template_dir = os.path.join(ISAACLAB_ARENA_DIR, ".idea", "tools", "codeStyles")
    
    if os.path.exists(code_styles_template_dir):
        os.makedirs(code_styles_dir, exist_ok=True)
        
        for template_file in os.listdir(code_styles_template_dir):
            if template_file.endswith('.xml'):
                template_path = os.path.join(code_styles_template_dir, template_file)
                target_path = os.path.join(code_styles_dir, template_file)
                
                if not os.path.exists(target_path):
                    with open(template_path) as f:
                        content = f.read()
                    
                    with open(target_path, "w") as f:
                        f.write(content)
                    
                    print(f"[INFO] Created code style configuration: {template_file}")
    
    # Copy inspection profiles
    inspection_profiles_dir = os.path.join(ISAACLAB_ARENA_DIR, ".idea", "inspectionProfiles")
    inspection_profiles_template_dir = os.path.join(ISAACLAB_ARENA_DIR, ".idea", "tools", "inspectionProfiles")
    
    if os.path.exists(inspection_profiles_template_dir):
        os.makedirs(inspection_profiles_dir, exist_ok=True)
        
        for template_file in os.listdir(inspection_profiles_template_dir):
            if template_file.endswith('.xml'):
                template_path = os.path.join(inspection_profiles_template_dir, template_file)
                target_path = os.path.join(inspection_profiles_dir, template_file)
                
                if not os.path.exists(target_path):
                    with open(template_path) as f:
                        content = f.read()
                    
                    with open(target_path, "w") as f:
                        f.write(content)
                    
                    print(f"[INFO] Created inspection profile: {template_file}")


def main():
    """Main function to setup PyCharm configuration."""
    print("[INFO] Setting up PyCharm configuration for Isaac Lab Arena...")

    # Get Python paths from Isaac Sim
    python_paths = get_python_paths_from_isaacsim()
    print(f"[INFO] Found {len(python_paths)} Python paths to configure")

    # Create/update PyCharm configuration files
    create_or_update_misc_xml()
    create_pycharm_libraries_xml(python_paths)
    copy_run_configurations()
    copy_additional_configurations()

    print("\n[SUCCESS] PyCharm configuration completed!")


if __name__ == "__main__":
    main()
