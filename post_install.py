import subprocess
import sys

def main():
    # Register the kernel
    kernel_name = "castro_env"
    display_name = "Python (castro_env)"
    subprocess.check_call([sys.executable, '-m', 'ipykernel', 'install', '--user', '--name', kernel_name, '--display-name', display_name])

if __name__ == "__main__":
    main()
