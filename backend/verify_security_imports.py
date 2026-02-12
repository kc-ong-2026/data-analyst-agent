#!/usr/bin/env python3
"""Quick verification script to ensure security modules can be imported."""

import sys
import traceback


def verify_imports():
    """Verify that all security modules can be imported without errors."""
    print("=" * 60)
    print("Security Module Import Verification")
    print("=" * 60)

    modules_to_test = [
        ("Security exceptions", "app.services.security.exceptions"),
        ("Code validator", "app.services.security.code_validator"),
        ("Sandbox executor", "app.services.security.sandbox_executor"),
        ("Audit logger", "app.services.security.audit_logger"),
        ("Security package", "app.services.security"),
    ]

    all_passed = True

    for name, module_path in modules_to_test:
        try:
            print(f"\n[TEST] {name}...", end=" ")
            __import__(module_path)
            print("✅ PASS")
        except Exception as e:
            print("❌ FAIL")
            print(f"  Error: {e}")
            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All imports successful!")
        print("=" * 60)
        return 0
    else:
        print("❌ Some imports failed. Please fix the errors above.")
        print("=" * 60)
        return 1


def verify_syntax():
    """Verify Python syntax of security module files."""
    import ast
    from pathlib import Path

    print("\n" + "=" * 60)
    print("Syntax Verification")
    print("=" * 60)

    security_dir = Path(__file__).parent / "app" / "services" / "security"
    if not security_dir.exists():
        print(f"❌ Security directory not found: {security_dir}")
        return 1

    all_passed = True
    for py_file in security_dir.glob("*.py"):
        if py_file.name.startswith("__"):
            continue

        try:
            print(f"\n[CHECK] {py_file.name}...", end=" ")
            with open(py_file) as f:
                code = f.read()
                ast.parse(code)
            print("✅ Valid syntax")
        except SyntaxError as e:
            print("❌ Syntax error")
            print(f"  Line {e.lineno}: {e.msg}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All files have valid syntax!")
        print("=" * 60)
        return 0
    else:
        print("❌ Some files have syntax errors.")
        print("=" * 60)
        return 1


def main():
    """Run all verification checks."""
    print("\n🔒 Security Implementation Verification\n")

    # Check syntax first (faster and doesn't require dependencies)
    syntax_result = verify_syntax()

    if syntax_result != 0:
        print("\n⚠️  Syntax check failed. Fix syntax errors before testing imports.")
        return syntax_result

    # Then try imports (requires dependencies)
    import_result = verify_imports()

    print("\n" + "=" * 60)
    if syntax_result == 0 and import_result == 0:
        print("🎉 All verification checks passed!")
        print("=" * 60)
        return 0
    else:
        print("⚠️  Some verification checks failed.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
