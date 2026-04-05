import os
import subprocess
import sys


REQUIRED_PACKAGES = [
    "openai",
    "openenv-core",
    "huggingface_hub",
    "fastapi",
    "uvicorn",
    "pydantic",
]


def check_python_version() -> bool:
    current = sys.version_info
    version_str = f"{current.major}.{current.minor}.{current.micro}"
    print(f"[INFO] Python version: {version_str}")
    if current >= (3, 10):
        print("[OK] Python version valid")
        return True
    print("[WARN] Python version is below 3.10")
    return False


def run_tool_check(command, tool_name, install_hint) -> bool:
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError:
        print(f"[ERROR] {tool_name} not found")
        print(f"[WARN] {install_hint}")
        return False

    if result.returncode == 0:
        output = (result.stdout or result.stderr).strip()
        print(f"[OK] {tool_name} detected: {output}")
        return True

    print(f"[ERROR] {tool_name} check failed")
    print(f"[WARN] {install_hint}")
    return False


def can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def install_package(package: str) -> bool:
    print(f"[INSTALLING] {package}...")
    result = subprocess.run(["pip", "install", package])
    if result.returncode == 0:
        print(f"[OK] Installed {package}")
        return True
    print(f"[ERROR] Failed to install {package}")
    return False


def check_and_install_packages() -> bool:
    all_good = True

    for package in REQUIRED_PACKAGES:
        import_name = package.replace("-", "_")
        if can_import(import_name):
            print(f"[OK] Library available: {package}")
            continue

        print(f"[WARN] Library missing: {package}")
        if not install_package(package):
            all_good = False

    return all_good


def check_hf_login() -> bool:
    token = os.getenv("HF_TOKEN")
    if token:
        print("[OK] HF_TOKEN found")
        return True

    print("[WARN] HF_TOKEN missing")
    print("Run: huggingface-cli login")
    return False


def check_env_vars() -> bool:
    ok = True
    api_base = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")

    if api_base:
        print("[OK] API_BASE_URL set")
    else:
        print("[WARN] API_BASE_URL missing")
        print("[WARN] Suggested default: https://router.huggingface.co/v1")
        ok = False

    if model_name:
        print("[OK] MODEL_NAME set")
    else:
        print("[WARN] MODEL_NAME missing")
        print("[WARN] Suggested default: Qwen/Qwen2.5-72B-Instruct")
        ok = False

    if hf_token:
        print("[OK] HF_TOKEN set")
    else:
        print("[WARN] HF_TOKEN missing")
        print("[WARN] Suggested action: Run: huggingface-cli login")
        ok = False

    return ok


def main() -> None:
    print("[INFO] Running setup checks...")

    python_ok = check_python_version()
    git_ok = run_tool_check(["git", "--version"], "Git", "Install Git and ensure it is in PATH")
    docker_ok = run_tool_check(["docker", "--version"], "Docker", "Install Docker Desktop and ensure it is in PATH")
    packages_ok = check_and_install_packages()
    hf_ok = check_hf_login()
    env_ok = check_env_vars()

    if python_ok and git_ok and docker_ok and packages_ok and hf_ok and env_ok:
        print("Setup Complete")
    else:
        print("Fix required")


if __name__ == "__main__":
    main()
