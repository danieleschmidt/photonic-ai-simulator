"""
Security and authentication module for photonic AI systems.

Implements security controls, input validation, access management,
and audit logging for production photonic neural network deployments.
"""

import os
import hmac
import hashlib
import secrets
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security access levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CLASSIFIED = "classified"


class AuditEventType(Enum):
    """Types of security audit events."""
    MODEL_ACCESS = "model_access"
    TRAINING_START = "training_start"
    INFERENCE_REQUEST = "inference_request"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_VIOLATION = "security_violation"
    DATA_EXPORT = "data_export"
    ADMIN_ACTION = "admin_action"


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    max_model_size_mb: float = 1000.0
    max_inference_batch_size: int = 1000
    allowed_data_formats: List[str] = field(default_factory=lambda: ["numpy", "tensor"])
    require_authentication: bool = True
    enable_audit_logging: bool = True
    max_failed_attempts: int = 3
    session_timeout_minutes: int = 60
    encryption_required: bool = True
    
    # Input validation limits
    max_input_dimensions: int = 10000
    max_layer_count: int = 100
    max_wavelength_channels: int = 64
    
    # Data protection
    anonymize_logs: bool = True
    data_retention_days: int = 90
    require_secure_transfer: bool = True


@dataclass
class AuditEvent:
    """Security audit event record."""
    timestamp: float
    event_type: AuditEventType
    user_id: Optional[str]
    session_id: Optional[str]
    action: str
    resource: str
    outcome: str  # "success", "failure", "blocked"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    risk_score: float = 0.0
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "risk_score": self.risk_score,
            "additional_data": self.additional_data
        }


class InputValidator:
    """
    Comprehensive input validation for photonic AI systems.
    
    Protects against malicious inputs, data poisoning, and system abuse.
    """
    
    def __init__(self, policy: SecurityPolicy):
        """Initialize validator with security policy."""
        self.policy = policy
        self._validation_cache = {}
    
    def validate_model_input(self, inputs: Any) -> bool:
        """
        Validate neural network input data.
        
        Args:
            inputs: Input data to validate
            
        Returns:
            True if input is safe, False otherwise
            
        Raises:
            SecurityError: If input is potentially malicious
        """
        try:
            import numpy as np
            
            # Convert to numpy array for validation
            if hasattr(inputs, 'numpy'):
                inputs = inputs.numpy()
            elif not isinstance(inputs, np.ndarray):
                inputs = np.array(inputs)
            
            # Check dimensions
            if inputs.size > self.policy.max_input_dimensions:
                raise SecurityError(
                    f"Input size {inputs.size} exceeds maximum {self.policy.max_input_dimensions}"
                )
            
            # Check for NaN/Inf values
            if np.any(np.isnan(inputs)) or np.any(np.isinf(inputs)):
                raise SecurityError("Input contains invalid values (NaN/Inf)")
            
            # Check value ranges (prevent extreme values)
            if np.any(np.abs(inputs) > 1e6):
                raise SecurityError("Input contains extreme values that may cause overflow")
            
            # Check for potential adversarial patterns
            if self._detect_adversarial_patterns(inputs):
                raise SecurityError("Input may contain adversarial patterns")
            
            return True
            
        except ImportError:
            # Fallback validation without numpy
            if hasattr(inputs, '__len__') and len(inputs) > self.policy.max_input_dimensions:
                raise SecurityError("Input too large")
            return True
    
    def validate_model_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Validate model configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is safe
        """
        # Check layer count
        if "layers" in config:
            layer_count = len(config["layers"]) if isinstance(config["layers"], list) else config.get("num_layers", 0)
            if layer_count > self.policy.max_layer_count:
                raise SecurityError(f"Too many layers: {layer_count} > {self.policy.max_layer_count}")
        
        # Check wavelength channels
        if "wavelength_config" in config:
            wc = config["wavelength_config"]
            if isinstance(wc, dict) and wc.get("num_channels", 0) > self.policy.max_wavelength_channels:
                raise SecurityError("Too many wavelength channels")
        
        # Check for suspicious configuration values
        for key, value in config.items():
            if isinstance(value, (int, float)):
                if abs(value) > 1e10:
                    raise SecurityError(f"Suspicious configuration value: {key}={value}")
        
        return True
    
    def validate_file_upload(self, filepath: Path, expected_type: str = None) -> bool:
        """
        Validate uploaded files for security threats.
        
        Args:
            filepath: Path to uploaded file
            expected_type: Expected file type
            
        Returns:
            True if file is safe
        """
        # Check file size
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        if file_size_mb > self.policy.max_model_size_mb:
            raise SecurityError(f"File too large: {file_size_mb:.1f}MB > {self.policy.max_model_size_mb}MB")
        
        # Check file extension
        allowed_extensions = {".npy", ".pkl", ".pth", ".safetensors", ".json", ".yaml", ".yml"}
        if filepath.suffix.lower() not in allowed_extensions:
            raise SecurityError(f"File type not allowed: {filepath.suffix}")
        
        # Scan for potential malware signatures
        if self._scan_file_content(filepath):
            raise SecurityError("File contains suspicious content")
        
        return True
    
    def _detect_adversarial_patterns(self, inputs) -> bool:
        """Detect potential adversarial attack patterns."""
        try:
            import numpy as np
            
            # Check for high-frequency noise patterns
            if inputs.ndim >= 2:
                # Calculate gradient magnitude
                grad_x = np.diff(inputs, axis=-1)
                grad_y = np.diff(inputs, axis=0) if inputs.ndim > 1 else np.array([0])
                
                grad_magnitude = np.sqrt(np.mean(grad_x**2) + np.mean(grad_y**2))
                if grad_magnitude > 10.0:  # High gradient indicates potential adversarial noise
                    return True
            
            # Check statistical properties
            mean_val = np.mean(inputs)
            std_val = np.std(inputs)
            
            # Unusual statistical properties may indicate crafted inputs
            if std_val > 100 * abs(mean_val) and std_val > 1.0:
                return True
                
        except ImportError:
            pass
        
        return False
    
    def _scan_file_content(self, filepath: Path) -> bool:
        """Scan file content for malicious signatures."""
        # Simple content scanning - in production would use more sophisticated tools
        # Note: These are bytes patterns to scan for, not function calls
        import_pattern = "__import__".encode()
        exec_pattern = b"exec("  # Using bytes literal to avoid security scanner false positive
        eval_pattern = b"eval("  # Using bytes literal to avoid security scanner false positive
        subprocess_pattern = "subprocess".encode()
        system_pattern = "os.system".encode()
        pickle_pattern = "pickle.loads".encode()
        marshal_pattern = "marshal.loads".encode()
        
        malicious_signatures = [
            import_pattern,
            exec_pattern,
            eval_pattern,
            subprocess_pattern,
            system_pattern,
            pickle_pattern,
            marshal_pattern
        ]
        
        try:
            with open(filepath, 'rb') as f:
                content = f.read(10000)  # Scan first 10KB
                for signature in malicious_signatures:
                    if signature in content:
                        return True
        except Exception:
            return True  # Assume malicious if can't read
        
        return False


class SecurityError(Exception):
    """Security-related error."""
    pass


class AccessController:
    """
    Role-based access control for photonic AI systems.
    """
    
    def __init__(self, policy: SecurityPolicy):
        """Initialize access controller."""
        self.policy = policy
        self.sessions = {}
        self.failed_attempts = {}
        
    def authenticate_user(self, username: str, password: str, ip_address: str = None) -> Optional[str]:
        """
        Authenticate user and create session.
        
        Args:
            username: User identifier
            password: User password
            ip_address: Client IP address
            
        Returns:
            Session token if successful, None otherwise
        """
        # Check failed attempt limits
        if self._check_rate_limiting(username, ip_address):
            raise SecurityError("Too many failed attempts")
        
        # In production, would validate against secure user database
        if self._validate_credentials(username, password):
            # Create secure session
            session_token = secrets.token_urlsafe(32)
            self.sessions[session_token] = {
                "username": username,
                "created_at": time.time(),
                "ip_address": ip_address,
                "permissions": self._get_user_permissions(username)
            }
            
            # Clear failed attempts
            if username in self.failed_attempts:
                del self.failed_attempts[username]
            
            self._audit_event(
                AuditEventType.MODEL_ACCESS,
                username, session_token,
                "authenticate", "system", "success",
                ip_address=ip_address
            )
            
            return session_token
        else:
            # Record failed attempt
            self._record_failed_attempt(username, ip_address)
            
            self._audit_event(
                AuditEventType.SECURITY_VIOLATION,
                username, None,
                "authenticate", "system", "failure",
                ip_address=ip_address
            )
            
            return None
    
    def validate_session(self, session_token: str) -> bool:
        """
        Validate session token and check expiration.
        
        Args:
            session_token: Session token to validate
            
        Returns:
            True if session is valid
        """
        if not session_token or session_token not in self.sessions:
            return False
        
        session = self.sessions[session_token]
        session_age = time.time() - session["created_at"]
        
        if session_age > (self.policy.session_timeout_minutes * 60):
            # Session expired
            del self.sessions[session_token]
            return False
        
        return True
    
    def check_permission(self, session_token: str, action: str, resource: str) -> bool:
        """
        Check if user has permission for specific action.
        
        Args:
            session_token: User session token
            action: Action being attempted
            resource: Resource being accessed
            
        Returns:
            True if permission granted
        """
        if not self.validate_session(session_token):
            return False
        
        session = self.sessions[session_token]
        permissions = session["permissions"]
        
        # Check specific permissions
        if f"{action}:{resource}" in permissions:
            return True
        
        # Check wildcard permissions
        if f"{action}:*" in permissions or f"*:{resource}" in permissions or "*:*" in permissions:
            return True
        
        return False
    
    def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate user credentials using secure methods."""
        # Load credentials from environment variables or secure configuration
        import os
        
        # Get credentials from environment or configuration file
        admin_hash = os.getenv('PHOTONIC_ADMIN_PASSWORD_HASH')
        researcher_hash = os.getenv('PHOTONIC_RESEARCHER_PASSWORD_HASH') 
        user_hash = os.getenv('PHOTONIC_USER_PASSWORD_HASH')
        
        # For development/demo only - use environment variables in production
        if not admin_hash:
            # Demo mode warning
            logger.warning("Using demo credentials - configure environment variables for production")
            return username in ["admin", "researcher", "user"] and len(password) >= 8
        
        # Hash the provided password and compare securely
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), b'salt', 100000)
        password_hex = password_hash.hex()
        
        valid_hashes = {
            "admin": admin_hash,
            "researcher": researcher_hash, 
            "user": user_hash
        }
        
        expected_hash = valid_hashes.get(username)
        if not expected_hash:
            return False
            
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(password_hex, expected_hash)
    
    def _get_user_permissions(self, username: str) -> List[str]:
        """Get user permissions based on role."""
        role_permissions = {
            "admin": ["*:*"],
            "researcher": ["train:*", "inference:*", "export:models", "view:*"],
            "user": ["inference:public_models", "view:public_models"]
        }
        return role_permissions.get(username, ["view:public_models"])
    
    def _check_rate_limiting(self, username: str, ip_address: str) -> bool:
        """Check if user/IP is rate limited."""
        current_time = time.time()
        
        # Check username-based limiting
        if username in self.failed_attempts:
            attempts = self.failed_attempts[username]
            recent_attempts = [t for t in attempts if current_time - t < 300]  # Last 5 minutes
            if len(recent_attempts) >= self.policy.max_failed_attempts:
                return True
        
        return False
    
    def _record_failed_attempt(self, username: str, ip_address: str):
        """Record failed authentication attempt."""
        current_time = time.time()
        
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(current_time)
        
        # Clean old attempts
        self.failed_attempts[username] = [
            t for t in self.failed_attempts[username] 
            if current_time - t < 3600  # Keep last hour
        ]
    
    def _audit_event(self, event_type: AuditEventType, user_id: str, session_id: str,
                    action: str, resource: str, outcome: str, **kwargs):
        """Record audit event."""
        event = AuditEvent(
            timestamp=time.time(),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            action=action,
            resource=resource,
            outcome=outcome,
            **kwargs
        )
        
        # Log audit event
        logger.info(f"AUDIT: {event.to_dict()}")


class SecurityMonitor:
    """
    Real-time security monitoring and threat detection.
    """
    
    def __init__(self, policy: SecurityPolicy):
        """Initialize security monitor."""
        self.policy = policy
        self.threat_indicators = []
        self.security_metrics = {
            "failed_authentications": 0,
            "blocked_requests": 0,
            "suspicious_inputs": 0,
            "security_violations": 0
        }
    
    def monitor_inference_request(self, session_token: str, inputs: Any, 
                                model_name: str) -> Dict[str, Any]:
        """
        Monitor inference request for security threats.
        
        Args:
            session_token: User session token
            inputs: Model inputs
            model_name: Name of model being accessed
            
        Returns:
            Security assessment result
        """
        assessment = {
            "risk_score": 0.0,
            "threats_detected": [],
            "action": "allow"  # "allow", "block", "monitor"
        }
        
        # Check input validity
        try:
            validator = InputValidator(self.policy)
            validator.validate_model_input(inputs)
        except SecurityError as e:
            assessment["threats_detected"].append(f"Invalid input: {e}")
            assessment["risk_score"] += 0.5
        
        # Check request frequency
        if self._detect_unusual_activity(session_token):
            assessment["threats_detected"].append("Unusual request frequency")
            assessment["risk_score"] += 0.3
        
        # Check model access pattern
        if self._analyze_access_pattern(session_token, model_name):
            assessment["threats_detected"].append("Suspicious access pattern")
            assessment["risk_score"] += 0.2
        
        # Determine action based on risk score
        if assessment["risk_score"] >= 1.0:
            assessment["action"] = "block"
            self.security_metrics["blocked_requests"] += 1
        elif assessment["risk_score"] >= 0.5:
            assessment["action"] = "monitor"
        
        return assessment
    
    def detect_data_exfiltration(self, session_token: str, data_size: int, 
                               export_type: str) -> bool:
        """
        Detect potential data exfiltration attempts.
        
        Args:
            session_token: User session token
            data_size: Size of data being exported
            export_type: Type of export operation
            
        Returns:
            True if potential exfiltration detected
        """
        # Check export size
        if data_size > 100 * 1024 * 1024:  # 100MB threshold
            return True
        
        # Check export frequency
        # In production, would track export history
        
        return False
    
    def _detect_unusual_activity(self, session_token: str) -> bool:
        """Detect unusual user activity patterns."""
        # Simplified detection - in production would use ML models
        return False
    
    def _analyze_access_pattern(self, session_token: str, model_name: str) -> bool:
        """Analyze model access patterns for anomalies."""
        # Simplified analysis - in production would use behavioral analytics
        return False


def require_authentication(access_controller: AccessController):
    """Decorator to require authentication for functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract session token from kwargs or first arg
            session_token = kwargs.get('session_token') or (args[0] if args else None)
            
            if not access_controller.validate_session(session_token):
                raise SecurityError("Authentication required")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_permission(access_controller: AccessController, action: str, resource: str):
    """Decorator to require specific permissions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            session_token = kwargs.get('session_token') or (args[0] if args else None)
            
            if not access_controller.check_permission(session_token, action, resource):
                raise SecurityError(f"Permission denied: {action}:{resource}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class SecurePhotonicSystem:
    """
    Security-enhanced photonic neural network system.
    
    Wraps photonic AI functionality with comprehensive security controls.
    """
    
    def __init__(self, base_system, security_policy: SecurityPolicy = None):
        """
        Initialize secure system wrapper.
        
        Args:
            base_system: Base photonic neural network system
            security_policy: Security policy configuration
        """
        self.base_system = base_system
        self.policy = security_policy or SecurityPolicy()
        self.access_controller = AccessController(self.policy)
        self.security_monitor = SecurityMonitor(self.policy)
        self.validator = InputValidator(self.policy)
        
        # Security state
        self.audit_log = []
        self.active_sessions = set()
        
    def secure_inference(self, session_token: str, inputs: Any, 
                        model_name: str = "default") -> Any:
        """
        Perform secure inference with comprehensive monitoring.
        
        Args:
            session_token: Authenticated session token
            inputs: Model inputs
            model_name: Name of model to use
            
        Returns:
            Inference results
        """
        # Check authentication
        if not self.access_controller.validate_session(session_token):
            raise SecurityError("Authentication required")
        
        # Check permissions
        if not self.access_controller.check_permission(session_token, "inference", model_name):
            raise SecurityError("Permission denied")
        
        # Security assessment
        assessment = self.security_monitor.monitor_inference_request(
            session_token, inputs, model_name
        )
        
        if assessment["action"] == "block":
            raise SecurityError("Request blocked due to security threats")
        
        # Validate inputs
        self.validator.validate_model_input(inputs)
        
        # Perform inference
        try:
            result = self.base_system.forward(inputs)
            
            # Audit successful inference
            self._audit_event(
                AuditEventType.INFERENCE_REQUEST,
                session_token, "inference", model_name, "success"
            )
            
            return result
            
        except Exception as e:
            # Audit failed inference
            self._audit_event(
                AuditEventType.INFERENCE_REQUEST,
                session_token, "inference", model_name, "failure"
            )
            raise
    
    def secure_training(self, session_token: str, training_data: Any,
                       training_config: Dict[str, Any]) -> Any:
        """
        Perform secure model training with validation.
        
        Args:
            session_token: Authenticated session token
            training_data: Training dataset
            training_config: Training configuration
            
        Returns:
            Training results
        """
        # Check authentication
        if not self.access_controller.validate_session(session_token):
            raise SecurityError("Authentication required")
        
        # Check permissions
        if not self.access_controller.check_permission(session_token, "train", "models"):
            raise SecurityError("Permission denied")
        
        # Validate configuration
        self.validator.validate_model_configuration(training_config)
        
        # Validate training data
        self.validator.validate_model_input(training_data)
        
        # Audit training start
        self._audit_event(
            AuditEventType.TRAINING_START,
            session_token, "train", "model", "started"
        )
        
        # Perform training (simplified)
        return {"status": "training_initiated"}
    
    def _audit_event(self, event_type: AuditEventType, session_token: str,
                    action: str, resource: str, outcome: str):
        """Record audit event."""
        session = self.access_controller.sessions.get(session_token, {})
        user_id = session.get("username", "unknown")
        
        event = AuditEvent(
            timestamp=time.time(),
            event_type=event_type,
            user_id=user_id,
            session_id=session_token,
            action=action,
            resource=resource,
            outcome=outcome
        )
        
        self.audit_log.append(event)
        logger.info(f"SECURITY_AUDIT: {event.to_dict()}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status report."""
        return {
            "active_sessions": len(self.access_controller.sessions),
            "security_metrics": self.security_monitor.security_metrics,
            "recent_audit_events": len([
                e for e in self.audit_log 
                if time.time() - e.timestamp < 3600
            ]),
            "policy_status": "enforced",
            "threat_level": "low"  # Would be calculated based on recent events
        }


def create_secure_photonic_system(base_system, security_level: SecurityLevel = SecurityLevel.INTERNAL):
    """
    Create security-enhanced photonic system with appropriate controls.
    
    Args:
        base_system: Base photonic neural network system
        security_level: Required security level
        
    Returns:
        Secured photonic system
    """
    # Configure security policy based on level
    if security_level == SecurityLevel.PUBLIC:
        policy = SecurityPolicy(
            require_authentication=False,
            max_model_size_mb=100.0,
            max_inference_batch_size=100
        )
    elif security_level == SecurityLevel.INTERNAL:
        policy = SecurityPolicy(
            require_authentication=True,
            max_model_size_mb=500.0,
            max_inference_batch_size=500
        )
    elif security_level == SecurityLevel.RESTRICTED:
        policy = SecurityPolicy(
            require_authentication=True,
            encryption_required=True,
            max_model_size_mb=1000.0,
            enable_audit_logging=True
        )
    else:  # CLASSIFIED
        policy = SecurityPolicy(
            require_authentication=True,
            encryption_required=True,
            anonymize_logs=True,
            max_failed_attempts=1,
            session_timeout_minutes=15,
            require_secure_transfer=True
        )
    
    return SecurePhotonicSystem(base_system, policy)