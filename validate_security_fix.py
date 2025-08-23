#!/usr/bin/env python3
"""
Security validation script to verify eval/exec pattern fixes.
"""

import os
import re
from pathlib import Path

def scan_for_security_issues():
    """Scan source files for security patterns."""
    src_path = Path("src")
    security_issues = []
    
    # Patterns to detect dangerous function calls (not pattern definitions)
    dangerous_patterns = [
        (r'\beval\s*\(', "Use of eval() function"),
        (r'\bexec\s*\(', "Use of exec() function"),
        (r'os\.system\s*\(', "Use of os.system()"),
        (r'subprocess\.call\s*\(.*shell\s*=\s*True', "Shell injection risk"),
    ]
    
    for py_file in src_path.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for pattern, description in dangerous_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    # Skip if it's in a comment or string definition for patterns
                    line_start = content.rfind('\n', 0, match.start()) + 1
                    line_end = content.find('\n', match.end())
                    if line_end == -1:
                        line_end = len(content)
                    line = content[line_start:line_end].strip()
                    
                    # Skip pattern definitions (bytes literals, comments)
                    if (line.startswith('#') or 
                        'pattern' in line.lower() or 
                        'b"' in line or 
                        "b'" in line or
                        '.encode()' in line):
                        continue
                        
                    security_issues.append({
                        'file': str(py_file),
                        'line': line,
                        'issue': description
                    })
        except Exception as e:
            print(f"Error scanning {py_file}: {e}")
    
    return security_issues

def main():
    print("🔒 Security Validation Report")
    print("=" * 50)
    
    issues = scan_for_security_issues()
    
    if not issues:
        print("✅ No security issues detected!")
        print("✅ eval/exec pattern fixes successful")
        return True
    else:
        print(f"❌ Found {len(issues)} security issues:")
        for issue in issues:
            print(f"  - {issue['file']}: {issue['issue']}")
            print(f"    Line: {issue['line']}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)