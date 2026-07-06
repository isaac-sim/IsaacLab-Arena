#!/usr/bin/env python3
"""Update PyCharm run configurations with Isaac Lab environment variables.

This script automatically extracts environment variables from Isaac Sim setup scripts
and injects them into PyCharm run configurations so debugging works correctly.
"""

import os
import pathlib
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom


ISAACLAB_ARENA_DIR = pathlib.Path(__file__).parents[2]
ISAACLAB_DIR = ISAACLAB_ARENA_DIR / "submodules" / "IsaacLab"
ISAAC_SIM_PATH = ISAACLAB_DIR / "_isaac_sim"
RUN_CONFIGS_DIR = ISAACLAB_ARENA_DIR / ".idea" / "runConfigurations"


def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def get_isaac_sim_env_vars():
    """Extract environment variables from Isaac Sim setup scripts."""
    env_vars = {}
    
    # Set Isaac Lab variables
    env_vars["ISAACLAB_PATH"] = str(ISAACLAB_DIR)
    env_vars["RESOURCE_NAME"] = "IsaacSim"
    
    # Check if Isaac Sim directory exists
    if not ISAAC_SIM_PATH.exists():
        print(f"[WARNING] Isaac Sim directory not found at: {ISAAC_SIM_PATH}")
        print("[WARNING] Only ISAACLAB_PATH and RESOURCE_NAME will be set.")
        return env_vars
    
    # Set Isaac Sim variables
    env_vars["ISAAC_PATH"] = str(ISAAC_SIM_PATH)
    env_vars["CARB_APP_PATH"] = str(ISAAC_SIM_PATH / "kit")
    env_vars["EXP_PATH"] = str(ISAAC_SIM_PATH / "apps")
    
    # Source the setup scripts to get PYTHONPATH and LD_LIBRARY_PATH
    script_path = ISAAC_SIM_PATH / "setup_python_env.sh"
    if script_path.exists():
        try:
            # Create a script that sources setup and prints env vars
            # Note: We also source setup_conda_env.sh to get the filtered PYTHONPATH
            # that removes Kit Python stdlib (which conflicts with conda Python)
            cmd = f"""
            cd {ISAAC_SIM_PATH}
            source ./setup_conda_env.sh > /dev/null 2>&1
            echo "PYTHONPATH=$PYTHONPATH"
            echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
            """
            
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        if key in ["PYTHONPATH", "LD_LIBRARY_PATH"]:
                            # Convert absolute paths to PyCharm variables where possible
                            value = value.replace(str(ISAAC_SIM_PATH), "$PROJECT_DIR$/submodules/IsaacLab/_isaac_sim")
                            value = value.replace(str(ISAACLAB_DIR), "$PROJECT_DIR$/submodules/IsaacLab")
                            env_vars[key] = value
            else:
                print(f"[WARNING] Failed to extract env vars from setup scripts")
        except Exception as e:
            print(f"[WARNING] Error extracting env vars: {e}")
    
    return env_vars


def update_run_configuration(config_file, env_vars):
    """Update a run configuration XML file with environment variables."""
    try:
        tree = ET.parse(config_file)
        root = tree.getroot()
        
        # Find the configuration element
        config = root.find(".//configuration")
        if config is None:
            print(f"[SKIP] No configuration element found in {config_file.name}")
            return False
        
        # Check if it's a Python configuration
        if config.get("type") != "PythonConfigurationType":
            print(f"[SKIP] Not a Python configuration: {config_file.name}")
            return False
        
        # Find or create envs element
        envs = config.find("envs")
        if envs is None:
            # Find the right place to insert (after PARENT_ENVS option)
            parent_envs_option = None
            for i, child in enumerate(config):
                if child.tag == "option" and child.get("name") == "PARENT_ENVS":
                    parent_envs_option = i
                    break
            
            if parent_envs_option is not None:
                envs = ET.Element("envs")
                config.insert(parent_envs_option + 1, envs)
            else:
                # If no PARENT_ENVS found, add at the beginning
                envs = ET.Element("envs")
                config.insert(0, envs)
        
        # Remove existing Isaac Lab related env vars to avoid duplicates
        isaac_env_keys = {"ISAACLAB_PATH", "RESOURCE_NAME", "ISAAC_PATH", "CARB_APP_PATH", 
                         "EXP_PATH", "PYTHONPATH", "LD_LIBRARY_PATH"}
        for env_elem in list(envs.findall("env")):
            if env_elem.get("name") in isaac_env_keys:
                envs.remove(env_elem)
        
        # Keep PYTHONUNBUFFERED if it exists
        pythonunbuffered_exists = any(
            env.get("name") == "PYTHONUNBUFFERED" for env in envs.findall("env")
        )
        if not pythonunbuffered_exists:
            ET.SubElement(envs, "env", name="PYTHONUNBUFFERED", value="1")
        
        # Add Isaac Lab environment variables
        for key, value in env_vars.items():
            ET.SubElement(envs, "env", name=key, value=value)
        
        # Remove "Setup IsaacLab Environment" before-launch task (no longer needed)
        method_elem = config.find("method")
        if method_elem is not None:
            for option in list(method_elem.findall("option")):
                if (option.get("name") == "RunConfigurationTask" and 
                    option.get("run_configuration_name") == "Setup IsaacLab Environment"):
                    method_elem.remove(option)
                    print(f"[INFO] Removed obsolete before-launch task from {config_file.name}")
        
        # Write the file
        xml_string = prettify_xml(root)
        
        # Check if it has a header comment
        has_header = False
        if config_file.exists():
            with open(config_file, 'r') as f:
                first_lines = f.read(200)
                if "automatically generated" in first_lines:
                    has_header = True
        
        # Preserve header if it exists
        if has_header:
            with open(config_file, 'r') as f:
                lines = f.readlines()
                header_lines = []
                for line in lines:
                    if line.strip().startswith('<!--'):
                        header_lines.append(line)
                    elif '-->' in line:
                        header_lines.append(line)
                        break
                    elif header_lines:
                        header_lines.append(line)
                
                if header_lines:
                    # Insert header after XML declaration
                    xml_lines = xml_string.split('\n')
                    if xml_lines[0].startswith('<?xml'):
                        header_text = ''.join(header_lines)
                        xml_string = xml_lines[0] + '\n' + header_text + '\n'.join(xml_lines[1:])
        
        with open(config_file, 'w') as f:
            f.write(xml_string)
        
        print(f"[UPDATED] {config_file.name}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to update {config_file.name}: {e}")
        return False


def main():
    """Main function."""
    print("=" * 70)
    print("Isaac Lab PyCharm Run Configuration Updater")
    print("=" * 70)
    print()
    
    # Get environment variables
    print("[INFO] Extracting environment variables from Isaac Sim...")
    env_vars = get_isaac_sim_env_vars()
    print(f"[INFO] Found {len(env_vars)} environment variables")
    print()
    
    # Ensure run configurations directory exists
    if not RUN_CONFIGS_DIR.exists():
        print(f"[ERROR] Run configurations directory not found: {RUN_CONFIGS_DIR}")
        return 1
    
    # Update all Python run configurations
    print("[INFO] Updating run configurations...")
    config_files = list(RUN_CONFIGS_DIR.glob("*.xml"))
    
    if not config_files:
        print("[WARNING] No run configuration files found")
        return 1
    
    updated = 0
    for config_file in sorted(config_files):
        if update_run_configuration(config_file, env_vars):
            updated += 1
    
    print()
    print("=" * 70)
    print(f"[SUCCESS] Updated {updated} out of {len(config_files)} configurations")
    print("=" * 70)
    print()
    print("Environment variables set:")
    for key in sorted(env_vars.keys()):
        print(f"  - {key}")
    print()
    print("[INFO] You may need to reload (File -> Reload All from Disk) for changes to take effect.")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())

