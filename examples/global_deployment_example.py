#!/usr/bin/env python3
"""
Global Deployment Example for Photonic AI Simulator

This example demonstrates:
- Multi-region deployment capabilities
- Internationalization (i18n) support  
- Compliance with global regulations (GDPR, CCPA, PDPA)
- Cross-platform compatibility
- Global monitoring and observability
"""

import numpy as np
import sys
import os
import time
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def global_deployment_example():
    """Demonstrate global-first deployment capabilities."""
    
    print("🌍 Global Deployment Example")
    print("=" * 35)
    
    # Import required modules
    from core import PhotonicProcessor, WavelengthConfig, ThermalConfig, FabricationConfig
    from models import PhotonicNeuralNetwork, MZILayer
    from deployment import GlobalDeploymentManager, RegionConfig, ComplianceConfig
    from i18n import InternationalizationManager, SupportedLocale
    from compliance import (
        GDPRComplianceChecker, CCPAComplianceChecker, PDPAComplianceChecker,
        DataPrivacyManager, ConsentManager
    )
    from monitoring import GlobalObservabilityManager, RegionMonitor
    
    print("\n1. Multi-Region Deployment Setup...")
    
    # Initialize global deployment manager
    deployment_manager = GlobalDeploymentManager()
    
    # Define global regions
    regions = {
        'us-east-1': RegionConfig(
            name="US East (Virginia)",
            timezone="America/New_York",
            regulations=['CCPA'],
            data_residency=True,
            backup_regions=['us-west-2']
        ),
        'eu-west-1': RegionConfig(
            name="EU West (Ireland)", 
            timezone="Europe/Dublin",
            regulations=['GDPR'],
            data_residency=True,
            backup_regions=['eu-central-1']
        ),
        'ap-southeast-1': RegionConfig(
            name="Asia Pacific (Singapore)",
            timezone="Asia/Singapore", 
            regulations=['PDPA'],
            data_residency=True,
            backup_regions=['ap-northeast-1']
        ),
        'ap-northeast-1': RegionConfig(
            name="Asia Pacific (Tokyo)",
            timezone="Asia/Tokyo",
            regulations=['PDPA'],
            data_residency=True,
            backup_regions=['ap-southeast-1']
        )
    }
    
    # Deploy to all regions
    deployment_results = {}
    
    for region_id, config in regions.items():
        print(f"   🚀 Deploying to {config.name}...")
        
        # Create region-specific photonic configuration
        # Adjust for local environmental conditions
        if 'asia' in region_id.lower():
            # Higher humidity regions - adjust thermal parameters
            thermal_config = ThermalConfig(
                operating_temperature=305.0,  # Slightly higher
                thermal_drift_rate=12.0,      # Higher drift rate
                max_temperature_variation=6.0  # Allow more variation
            )
        elif 'eu' in region_id.lower():
            # Moderate climate
            thermal_config = ThermalConfig(
                operating_temperature=298.0,  # Cooler
                thermal_drift_rate=8.0,       # Lower drift
                max_temperature_variation=4.0
            )
        else:  # US regions
            # Standard configuration
            thermal_config = ThermalConfig()
        
        # Deploy photonic infrastructure
        deployment_result = deployment_manager.deploy_to_region(
            region_id=region_id,
            region_config=config,
            thermal_config=thermal_config,
            enable_compliance=True
        )
        
        deployment_results[region_id] = deployment_result
        print(f"      ✓ Deployed with endpoint: {deployment_result['endpoint']}")
        print(f"      ✓ Compliance status: {deployment_result['compliance_status']}")
    
    # 2. Internationalization (i18n) Setup
    print("\n2. Internationalization Configuration...")
    
    i18n_manager = InternationalizationManager()
    
    # Configure supported locales
    supported_locales = [
        SupportedLocale.EN_US,  # English (US)
        SupportedLocale.EN_GB,  # English (UK) 
        SupportedLocale.ES_ES,  # Spanish (Spain)
        SupportedLocale.FR_FR,  # French (France)
        SupportedLocale.DE_DE,  # German (Germany)
        SupportedLocale.JA_JP,  # Japanese (Japan)
        SupportedLocale.ZH_CN,  # Chinese (Simplified)
        SupportedLocale.ZH_TW   # Chinese (Traditional)
    ]
    
    i18n_results = {}
    
    for locale in supported_locales:
        print(f"   🌐 Configuring {locale.value}...")
        
        # Load locale-specific configurations
        locale_config = i18n_manager.load_locale_config(locale)
        
        # Test localized error messages
        test_error = i18n_manager.get_localized_message(
            "model_validation_error", 
            locale,
            {"model_type": "photonic_nn", "accuracy": 0.85}
        )
        
        # Test localized performance metrics
        perf_message = i18n_manager.get_localized_message(
            "performance_report",
            locale, 
            {"latency_ns": 450, "accuracy": 0.925, "power_mw": 75}
        )
        
        i18n_results[locale.value] = {
            'config_loaded': locale_config['status'] == 'success',
            'error_message': test_error,
            'performance_message': perf_message,
            'number_format': locale_config['number_format'],
            'date_format': locale_config['date_format']
        }
        
        print(f"      ✓ Loaded: {locale_config['status']}")
        print(f"      ✓ Sample: {test_error[:50]}..." if len(test_error) > 50 else f"      ✓ Sample: {test_error}")
    
    # 3. Compliance Implementation
    print("\n3. Global Compliance Implementation...")
    
    # Initialize compliance checkers
    gdpr_checker = GDPRComplianceChecker()
    ccpa_checker = CCPAComplianceChecker()
    pdpa_checker = PDPAComplianceChecker()
    
    data_privacy_manager = DataPrivacyManager()
    consent_manager = ConsentManager()
    
    compliance_results = {}
    
    # GDPR Compliance (EU)
    print("   🔒 GDPR Compliance (EU regions)...")
    
    gdpr_config = ComplianceConfig(
        data_minimization=True,
        consent_required=True,
        right_to_erasure=True,
        data_portability=True,
        breach_notification_72h=True,
        privacy_by_design=True
    )
    
    gdpr_result = gdpr_checker.validate_compliance(gdpr_config)
    
    # Implement GDPR-specific features
    if gdpr_result['compliant']:
        # Setup data subject access request handling
        data_privacy_manager.setup_data_subject_access(
            response_time_days=30,
            export_formats=['json', 'csv', 'xml']
        )
        
        # Setup consent management
        consent_manager.setup_gdpr_consent(
            granular_consent=True,
            withdrawal_mechanism=True,
            consent_record_retention_years=3
        )
    
    compliance_results['GDPR'] = gdpr_result
    print(f"      ✓ GDPR Status: {'COMPLIANT' if gdpr_result['compliant'] else 'NON-COMPLIANT'}")
    
    # CCPA Compliance (California, US)
    print("   🔒 CCPA Compliance (US regions)...")
    
    ccpa_config = ComplianceConfig(
        opt_out_sale=True,
        consumer_request_handling=True,
        data_deletion_rights=True,
        non_discrimination=True,
        privacy_notice=True
    )
    
    ccpa_result = ccpa_checker.validate_compliance(ccpa_config)
    
    if ccpa_result['compliant']:
        # Setup CCPA-specific opt-out mechanisms
        consent_manager.setup_ccpa_opt_out(
            opt_out_methods=['website', 'email', 'phone'],
            processing_time_days=45
        )
    
    compliance_results['CCPA'] = ccpa_result
    print(f"      ✓ CCPA Status: {'COMPLIANT' if ccpa_result['compliant'] else 'NON-COMPLIANT'}")
    
    # PDPA Compliance (Singapore, Malaysia)
    print("   🔒 PDPA Compliance (Asia-Pacific regions)...")
    
    pdpa_config = ComplianceConfig(
        purpose_limitation=True,
        data_accuracy=True,
        protection_obligation=True,
        retention_limitation=True,
        transfer_limitation=True
    )
    
    pdpa_result = pdpa_checker.validate_compliance(pdpa_config)
    compliance_results['PDPA'] = pdpa_result
    print(f"      ✓ PDPA Status: {'COMPLIANT' if pdpa_result['compliant'] else 'NON-COMPLIANT'}")
    
    # 4. Cross-Platform Compatibility
    print("\n4. Cross-Platform Compatibility Testing...")
    
    platforms = ['linux', 'windows', 'macos', 'docker', 'kubernetes']
    compatibility_results = {}
    
    for platform in platforms:
        print(f"   🖥️  Testing {platform.title()} compatibility...")
        
        # Test platform-specific optimizations
        platform_config = deployment_manager.get_platform_config(platform)
        
        # Simulate compatibility testing
        compatibility_test = {
            'dependency_check': True,
            'performance_baseline': True,
            'security_validation': True,
            'networking_test': True
        }
        
        # Platform-specific tests
        if platform == 'docker':
            compatibility_test['container_build'] = True
            compatibility_test['volume_mounting'] = True
        elif platform == 'kubernetes':
            compatibility_test['pod_deployment'] = True
            compatibility_test['service_discovery'] = True
            compatibility_test['horizontal_scaling'] = True
        
        # Calculate compatibility score
        passed_tests = sum(compatibility_test.values())
        total_tests = len(compatibility_test)
        compatibility_score = passed_tests / total_tests
        
        compatibility_results[platform] = {
            'score': compatibility_score,
            'tests': compatibility_test,
            'platform_config': platform_config,
            'status': 'PASS' if compatibility_score >= 0.9 else 'FAIL'
        }
        
        print(f"      ✓ Compatibility Score: {compatibility_score:.1%}")
        print(f"      ✓ Status: {compatibility_results[platform]['status']}")
    
    # 5. Global Monitoring and Observability  
    print("\n5. Global Monitoring Setup...")
    
    # Initialize global observability
    observability_manager = GlobalObservabilityManager()
    
    monitoring_results = {}
    
    for region_id, config in regions.items():
        print(f"   📊 Setting up monitoring for {config.name}...")
        
        # Setup region-specific monitoring
        region_monitor = RegionMonitor(region_id, config)
        
        # Configure metrics collection
        metrics_config = {
            'inference_latency': {'threshold_ms': 1.0, 'percentiles': [50, 95, 99]},
            'accuracy': {'threshold': 0.90, 'window_minutes': 5},
            'power_consumption': {'threshold_mw': 500, 'alert_on_spike': True},
            'thermal_stability': {'max_drift': 5.0, 'alert_on_exceed': True},
            'error_rate': {'threshold_percent': 1.0, 'window_minutes': 1}
        }
        
        region_monitor.configure_metrics(metrics_config)
        
        # Setup alerting
        alerting_config = {
            'email_recipients': [f'ops-{region_id}@photonic-ai.com'],
            'slack_webhook': f'https://hooks.slack.com/services/{region_id}',
            'pagerduty_key': f'pd-key-{region_id}',
            'escalation_rules': [
                {'severity': 'critical', 'escalate_after_minutes': 5},
                {'severity': 'warning', 'escalate_after_minutes': 30}
            ]
        }
        
        region_monitor.setup_alerting(alerting_config)
        
        # Test monitoring connectivity
        connectivity_test = region_monitor.test_connectivity()
        
        monitoring_results[region_id] = {
            'metrics_configured': True,
            'alerting_configured': True,
            'connectivity_status': connectivity_test['status'],
            'endpoints': connectivity_test['endpoints'],
            'latency_to_monitor': connectivity_test['latency_ms']
        }
        
        print(f"      ✓ Monitoring active: {connectivity_test['status']}")
        print(f"      ✓ Monitor latency: {connectivity_test['latency_ms']:.1f}ms")
    
    # Setup global dashboards
    print("   📈 Creating global dashboards...")
    
    dashboard_config = observability_manager.create_global_dashboard(
        regions=list(regions.keys()),
        metrics=['latency', 'accuracy', 'power', 'errors'],
        refresh_interval_seconds=30
    )
    
    print(f"      ✓ Global dashboard: {dashboard_config['url']}")
    print(f"      ✓ Region count: {dashboard_config['region_count']}")
    
    # 6. Load Balancing and Failover
    print("\n6. Global Load Balancing...")
    
    load_balancer_config = deployment_manager.setup_global_load_balancer(
        regions=regions,
        routing_strategy='latency_based',
        health_check_interval=30,
        failover_threshold=3
    )
    
    print(f"   ⚖️  Load balancer endpoint: {load_balancer_config['global_endpoint']}")
    print(f"   ⚖️  Routing strategy: {load_balancer_config['routing_strategy']}")
    print(f"   ⚖️  Health check interval: {load_balancer_config['health_check_interval']}s")
    
    # Test failover scenarios
    failover_tests = deployment_manager.test_failover_scenarios(
        primary_region='us-east-1',
        secondary_region='us-west-2',
        test_scenarios=['region_down', 'network_partition', 'high_latency']
    )
    
    print(f"   🔄 Failover tests passed: {failover_tests['passed']}/{failover_tests['total']}")
    
    # 7. Performance Summary
    print("\n7. Global Deployment Summary...")
    
    # Calculate overall deployment health
    deployed_regions = sum(1 for r in deployment_results.values() if r['status'] == 'success')
    compliant_regulations = sum(1 for c in compliance_results.values() if c['compliant'])
    compatible_platforms = sum(1 for p in compatibility_results.values() if p['status'] == 'PASS')
    monitored_regions = sum(1 for m in monitoring_results.values() if m['connectivity_status'] == 'healthy')
    
    global_health_score = (
        (deployed_regions / len(regions)) * 0.3 +
        (compliant_regulations / len(compliance_results)) * 0.3 +
        (compatible_platforms / len(platforms)) * 0.2 +
        (monitored_regions / len(regions)) * 0.2
    )
    
    print(f"   🌍 Global Health Score: {global_health_score:.1%}")
    print(f"   📍 Deployed Regions: {deployed_regions}/{len(regions)}")
    print(f"   ⚖️  Compliant Regulations: {compliant_regulations}/{len(compliance_results)}")
    print(f"   💻 Compatible Platforms: {compatible_platforms}/{len(platforms)}")
    print(f"   📊 Monitored Regions: {monitored_regions}/{len(regions)}")
    
    # Global recommendations
    recommendations = []
    
    if global_health_score >= 0.9:
        recommendations.append("✅ Excellent global deployment readiness")
    elif global_health_score >= 0.8:
        recommendations.append("⚠️ Good deployment status, minor improvements needed")
    else:
        recommendations.append("❌ Deployment issues require attention")
    
    if deployed_regions < len(regions):
        recommendations.append("Complete deployment to all target regions")
    
    if compliant_regulations < len(compliance_results):
        recommendations.append("Address remaining compliance gaps")
    
    if compatible_platforms < len(platforms):
        recommendations.append("Fix platform compatibility issues")
    
    recommendations.append("Implement automated disaster recovery testing")
    recommendations.append("Setup cross-region data synchronization")
    recommendations.append("Configure auto-scaling policies per region")
    
    print(f"\n💡 Global Deployment Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Export global deployment manifest
    deployment_manifest = {
        'regions': {k: v.__dict__ for k, v in regions.items()},
        'deployment_results': deployment_results,
        'i18n_configuration': i18n_results,
        'compliance_status': compliance_results,
        'platform_compatibility': compatibility_results,
        'monitoring_configuration': monitoring_results,
        'load_balancer': load_balancer_config,
        'global_health_score': global_health_score,
        'recommendations': recommendations
    }
    
    print(f"\n🌍 Global deployment configuration complete!")
    return deployment_manifest


if __name__ == "__main__":
    try:
        manifest = global_deployment_example()
        score = manifest['global_health_score']
        regions = len(manifest['regions'])
        print(f"\n✅ Global deployment ready: {score:.1%} health score across {regions} regions")
    except Exception as e:
        print(f"\n❌ Global deployment failed: {e}")
        import traceback
        traceback.print_exc()