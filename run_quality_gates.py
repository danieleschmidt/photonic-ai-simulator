#!/usr/bin/env python3
"""
Quality Gates Execution Script.

Runs comprehensive quality gates and validation for the photonic AI system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import time
import logging
from pathlib import Path

# Import quality gate framework
from quality_gates import run_quality_gates, QualityGateConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Main execution function for quality gates."""
    print("🚀 TERRAGON LABS - PHOTONIC AI QUALITY GATES")
    print("=" * 60)
    print("Running comprehensive quality validation...")
    
    try:
        # Configure quality gates
        config = QualityGateConfig(
            min_test_coverage=85.0,
            max_acceptable_errors=0,
            security_scan_required=True,
            documentation_coverage=80.0,
            code_quality_threshold=8.0
        )
        
        # Execute quality gates
        start_time = time.time()
        
        print("\n📊 EXECUTING QUALITY GATES...")
        print("-" * 40)
        
        # Run comprehensive quality validation
        from quality_gates import QualityGateRunner
        runner = QualityGateRunner(config)
        quality_report = runner.run_all_tests()
        
        execution_time = time.time() - start_time
        
        # Display results
        print(f"\n🏆 QUALITY GATES RESULTS")
        print("=" * 60)
        print(f"Overall Status: {quality_report.overall_status.upper()}")
        print(f"Test Coverage: {quality_report.test_coverage_percentage:.1f}%")
        print(f"Tests Passed: {quality_report.passed_tests}/{quality_report.total_tests}")
        print(f"Failed Tests: {quality_report.failed_tests}")
        print(f"Execution Time: {quality_report.execution_time_s:.2f}s")
        print(f"Code Quality Score: {quality_report.code_quality_score}/10")
        
        # Performance benchmarks
        print(f"\n⚡ PERFORMANCE BENCHMARKS")
        print("-" * 30)
        for benchmark, passed in quality_report.performance_benchmarks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{benchmark}: {status}")
        
        # Security scan results
        print(f"\n🔒 SECURITY SCAN RESULTS")
        print("-" * 30)
        for check, result in quality_report.security_scan_results.items():
            status = "✅ PASS" if result == "passed" else "❌ FAIL"
            print(f"{check}: {status}")
        
        # Detailed test results
        if quality_report.failed_tests > 0:
            print(f"\n❌ FAILED TESTS")
            print("-" * 30)
            for result in quality_report.detailed_results:
                if result.status == "failed":
                    print(f"• {result.test_name}")
                    if result.error_message:
                        print(f"  Error: {result.error_message[:100]}...")
        
        # Recommendations
        if quality_report.recommendations:
            print(f"\n📋 RECOMMENDATIONS")
            print("-" * 30)
            for i, rec in enumerate(quality_report.recommendations, 1):
                print(f"{i}. {rec}")
        
        # Summary assessment
        print(f"\n📈 QUALITY ASSESSMENT SUMMARY")
        print("=" * 60)
        
        if quality_report.overall_status == "passed":
            print("🎉 ALL QUALITY GATES PASSED!")
            print("✅ System meets production-grade quality standards")
            print("✅ Ready for deployment and production use")
            
            # Calculate quality score
            quality_score = (
                (quality_report.test_coverage_percentage / 100) * 0.4 +
                (quality_report.passed_tests / max(quality_report.total_tests, 1)) * 0.3 +
                (sum(quality_report.performance_benchmarks.values()) / max(len(quality_report.performance_benchmarks), 1)) * 0.2 +
                (quality_report.code_quality_score / 10) * 0.1
            )
            
            print(f"📊 Overall Quality Score: {quality_score * 100:.1f}%")
            
            if quality_score >= 0.9:
                print("🏆 EXCELLENCE: Outstanding quality metrics achieved!")
            elif quality_score >= 0.8:
                print("🥈 HIGH QUALITY: Very good quality standards met")
            else:
                print("🥉 GOOD: Quality standards met with room for improvement")
        
        elif quality_report.overall_status == "warning":
            print("⚠️  QUALITY GATES PASSED WITH WARNINGS")
            print("✅ Core functionality validated")
            print("⚠️  Some performance or quality improvements recommended")
            print("📝 Address recommendations before production deployment")
        
        else:
            print("❌ QUALITY GATES FAILED")
            print("❌ System does not meet minimum quality standards")
            print("🔧 Fix identified issues before proceeding")
            
            # Provide specific guidance
            if quality_report.test_coverage_percentage < config.min_test_coverage:
                print(f"❗ Test coverage ({quality_report.test_coverage_percentage:.1f}%) below minimum ({config.min_test_coverage}%)")
            
            if quality_report.failed_tests > config.max_acceptable_errors:
                print(f"❗ {quality_report.failed_tests} tests failing (maximum allowed: {config.max_acceptable_errors})")
        
        # Generate comprehensive report
        report_path = runner.generate_quality_report(quality_report, "quality_gates_report.json")
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        # Return appropriate exit code
        if quality_report.overall_status == "passed":
            return 0
        elif quality_report.overall_status == "warning":
            return 0  # Warnings are acceptable
        else:
            return 1
    
    except Exception as e:
        print(f"\n❌ QUALITY GATES EXECUTION FAILED")
        print(f"Error: {e}")
        logging.error(f"Quality gates failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)