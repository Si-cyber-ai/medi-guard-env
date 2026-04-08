import subprocess
import re
import sys

def run_inference():
    try:
        result = subprocess.run(
            ["python", "inference.py"],
            capture_output=True,
            text=True,
            timeout=1200
        )
        return result.stdout
    except Exception as e:
        print("❌ Failed to run inference:", e)
        sys.exit(1)


def validate_logs(output):
    lines = output.strip().split("\n")

    start_count = 0
    end_count = 0

    valid = True

    for line in lines:
        if line.startswith("[START]"):
            start_count += 1

        elif line.startswith("[STEP]"):
            # Validate STEP format
            pattern = r"\[STEP\] step=\d+ action=\w+ reward=-?\d+\.\d+ done=(true|false) error=.*"
            if not re.match(pattern, line):
                print("❌ Invalid STEP format:", line)
                valid = False

        elif line.startswith("[END]"):
            end_count += 1

            # Validate score
            match = re.search(r"score=([0-9.]+)", line)
            if match:
                score = float(match.group(1))
                if not (0 < score < 1):
                    print("❌ Score out of range:", score)
                    valid = False
            else:
                print("❌ Score missing in END:", line)
                valid = False

        else:
            print("❌ Invalid log line (extra output):", line)
            valid = False

    # Check counts
    if start_count != 3:
        print(f"❌ Expected 3 START logs, got {start_count}")
        valid = False

    if end_count != 3:
        print(f"❌ Expected 3 END logs, got {end_count}")
        valid = False

    return valid


def main():
    print("🚀 Running inference...")
    output = run_inference()

    print("🔍 Validating logs...")
    is_valid = validate_logs(output)

    if is_valid:
        print("\n✅ ALL CHECKS PASSED — READY TO SUBMIT 🚀")
    else:
        print("\n❌ VALIDATION FAILED — FIX ISSUES")


if __name__ == "__main__":
    main()