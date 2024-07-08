import os
import subprocess
import sys

def main():
    # Ensure JupyterLab is installed
    try:
        import jupyterlab
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'jupyterlab'])

    # Ensure ipykernel is installed
    try:
        import ipykernel
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ipykernel'])

    # Register the kernel
    kernel_name = "castro_env"
    display_name = "Python (castro_env)"
    subprocess.check_call([sys.executable, '-m', 'ipykernel', 'install', '--user', '--name', kernel_name, '--display-name', display_name])

    # Launch JupyterLab
    subprocess.check_call([sys.executable, '-m', 'jupyter', 'lab'])

if __name__ == "__main__":
    main()
