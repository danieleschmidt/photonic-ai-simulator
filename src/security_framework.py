"""
Security Framework for Photonic AI Systems.

Provides comprehensive security measures including access control,
data protection, secure communications, and threat detection for
photonic neural networks and federated learning systems.
"""

import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import base64
import json
from datetime import datetime, timedelta
import threading
import numpy as np


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityCredentials:
    """Security credentials for photonic AI access."""
    user_id: str
    api_key: str
    security_level: SecurityLevel
    permissions: List[str]
    issued_at: datetime
    expires_at: datetime
    ip_whitelist: List[str] = field(default_factory=list)
    rate_limit: int = 1000  # requests per hour


@dataclass
class ThreatEvent:
    """Security threat event record."""
    threat_id: str
    threat_type: str
    severity: ThreatLevel
    source_ip: str
    target_component: str
    timestamp: datetime
    details: Dict[str, Any]
    mitigated: bool = False


class PhotonicSecurityManager:
    """
    Comprehensive security manager for photonic AI systems.
    
    Handles authentication, authorization, encryption, and threat detection
    for secure operation of photonic neural networks.
    """
    
    def __init__(self, security_config: Dict[str, Any] = None):
        self.security_config = security_config or {}
        self.credentials_store = {}
        self.active_sessions = {}
        self.threat_log = []
        self.rate_limits = {}
        
        # Security keys
        self.master_key = self._generate_master_key()
        self.encryption_key = self._derive_encryption_key()
        
        # Threat detection
        self.threat_detector = ThreatDetector()
        
        # Audit logging
        self.security_logger = logging.getLogger("photonic_security")
        self.security_logger.setLevel(logging.INFO)
        
        # Session management
        self.session_lock = threading.Lock()
        
        self.security_logger.info("Photonic security manager initialized")
    
    def _generate_master_key(self) -> bytes:
        """Generate cryptographically secure master key."""
        return secrets.token_bytes(32)  # 256-bit key
    
    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from master key."""
        return hashlib.pbkdf2_hmac(
            'sha256',
            self.master_key,
            b'photonic_ai_salt',
            100000,  # iterations
            32  # key length
        )
    
    def create_credentials(self, 
                          user_id: str,
                          security_level: SecurityLevel,
                          permissions: List[str],
                          validity_hours: int = 24) -> SecurityCredentials:
        """
        Create new security credentials for a user.
        
        Args:
            user_id: Unique user identifier
            security_level: Security clearance level
            permissions: List of granted permissions
            validity_hours: Credential validity period
            
        Returns:
            SecurityCredentials object
        """
        # Generate secure API key
        api_key = self._generate_api_key(user_id, security_level)
        
        # Set expiration
        issued_at = datetime.utcnow()
        expires_at = issued_at + timedelta(hours=validity_hours)
        
        credentials = SecurityCredentials(
            user_id=user_id,
            api_key=api_key,
            security_level=security_level,
            permissions=permissions,
            issued_at=issued_at,
            expires_at=expires_at
        )
        
        # Store credentials
        self.credentials_store[api_key] = credentials
        
        self.security_logger.info(f"Created credentials for user {user_id} with level {security_level.value}")
        
        return credentials
    
    def _generate_api_key(self, user_id: str, security_level: SecurityLevel) -> str:
        """Generate secure API key."""
        # Create key payload
        payload = {
            "user_id": user_id,
            "security_level": security_level.value,
            "timestamp": time.time(),
            "nonce": secrets.token_hex(8)
        }
        
        # Sign payload
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        signature = hmac.new(self.master_key, payload_bytes, hashlib.sha256).hexdigest()
        
        # Encode as API key
        api_key = base64.b64encode(payload_bytes + b'.' + signature.encode()).decode()
        
        return f"pk_{api_key}"  # photonic key prefix
    
    def authenticate(self, api_key: str, source_ip: str = None) -> Optional[SecurityCredentials]:
        """
        Authenticate user with API key.
        
        Args:
            api_key: User's API key
            source_ip: Source IP address
            
        Returns:
            SecurityCredentials if valid, None otherwise
        """
        try:
            # Check rate limiting
            if not self._check_rate_limit(api_key, source_ip):
                self._log_threat(
                    "rate_limit_exceeded",
                    ThreatLevel.MEDIUM,
                    source_ip or "unknown",
                    "authentication",
                    {"api_key": api_key[:10] + "..."}
                )
                return None
            
            # Validate API key format
            if not api_key.startswith("pk_"):
                self._log_threat(
                    "invalid_api_key_format",
                    ThreatLevel.LOW,
                    source_ip or "unknown",
                    "authentication",
                    {"api_key": api_key[:10] + "..."}
                )
                return None
            
            # Get credentials
            credentials = self.credentials_store.get(api_key)
            if not credentials:
                self._log_threat(
                    "unknown_api_key",
                    ThreatLevel.MEDIUM,
                    source_ip or "unknown",
                    "authentication",
                    {"api_key": api_key[:10] + "..."}
                )
                return None
            
            # Check expiration
            if datetime.utcnow() > credentials.expires_at:
                self._log_threat(
                    "expired_credentials",
                    ThreatLevel.LOW,
                    source_ip or "unknown",
                    "authentication",
                    {"user_id": credentials.user_id}
                )
                return None
            
            # Check IP whitelist
            if credentials.ip_whitelist and source_ip not in credentials.ip_whitelist:
                self._log_threat(
                    "unauthorized_ip",
                    ThreatLevel.HIGH,
                    source_ip or "unknown",
                    "authentication",
                    {"user_id": credentials.user_id, "ip": source_ip}
                )
                return None
            
            # Create session
            session_id = self._create_session(credentials, source_ip)
            
            self.security_logger.info(f"Successful authentication for user {credentials.user_id}")
            
            return credentials
            
        except Exception as e:
            self.security_logger.error(f"Authentication error: {e}")
            return None
    
    def authorize(self, credentials: SecurityCredentials, 
                 required_permission: str,
                 resource_security_level: SecurityLevel = SecurityLevel.PUBLIC) -> bool:
        """
        Authorize user action based on permissions and security level.
        
        Args:
            credentials: User credentials
            required_permission: Required permission for action
            resource_security_level: Security level of target resource
            
        Returns:
            True if authorized, False otherwise
        """
        # Check security level clearance
        level_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.SECRET: 3,
            SecurityLevel.TOP_SECRET: 4
        }
        
        user_level = level_hierarchy[credentials.security_level]
        required_level = level_hierarchy[resource_security_level]
        
        if user_level < required_level:
            self._log_threat(
                "insufficient_security_clearance",
                ThreatLevel.MEDIUM,
                "authenticated_user",
                "authorization",
                {
                    "user_id": credentials.user_id,
                    "user_level": credentials.security_level.value,
                    "required_level": resource_security_level.value
                }
            )
            return False
        
        # Check specific permission
        if required_permission not in credentials.permissions and "admin" not in credentials.permissions:
            self._log_threat(
                "insufficient_permissions",
                ThreatLevel.MEDIUM,
                "authenticated_user",
                "authorization",
                {
                    "user_id": credentials.user_id,
                    "required_permission": required_permission,
                    "user_permissions": credentials.permissions
                }
            )
            return False
        
        return True
    
    def _check_rate_limit(self, api_key: str, source_ip: str) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - 3600  # 1 hour
        
        for key in list(self.rate_limits.keys()):
            self.rate_limits[key] = [t for t in self.rate_limits[key] if t > cutoff_time]
            if not self.rate_limits[key]:
                del self.rate_limits[key]
        
        # Check API key rate limit
        api_requests = self.rate_limits.get(f"api:{api_key}", [])
        if len(api_requests) >= 1000:  # 1000 requests per hour
            return False
        
        # Check IP rate limit
        if source_ip:
            ip_requests = self.rate_limits.get(f"ip:{source_ip}", [])
            if len(ip_requests) >= 5000:  # 5000 requests per hour per IP
                return False
        
        # Record request
        api_requests.append(current_time)
        self.rate_limits[f"api:{api_key}"] = api_requests
        
        if source_ip:
            ip_requests = self.rate_limits.get(f"ip:{source_ip}", [])
            ip_requests.append(current_time)
            self.rate_limits[f"ip:{source_ip}"] = ip_requests
        
        return True
    
    def _create_session(self, credentials: SecurityCredentials, source_ip: str) -> str:
        """Create authenticated session."""
        with self.session_lock:
            session_id = secrets.token_urlsafe(32)
            
            self.active_sessions[session_id] = {
                "credentials": credentials,
                "source_ip": source_ip,
                "created_at": datetime.utcnow(),
                "last_activity": datetime.utcnow()
            }
            
            return session_id
    
    def _log_threat(self, threat_type: str, severity: ThreatLevel, 
                   source_ip: str, target_component: str, details: Dict[str, Any]):
        """Log security threat event."""
        threat_event = ThreatEvent(
            threat_id=secrets.token_hex(8),
            threat_type=threat_type,
            severity=severity,
            source_ip=source_ip,
            target_component=target_component,
            timestamp=datetime.utcnow(),
            details=details
        )
        
        self.threat_log.append(threat_event)
        
        # Log at appropriate level
        log_level = {
            ThreatLevel.LOW: logging.INFO,
            ThreatLevel.MEDIUM: logging.WARNING,
            ThreatLevel.HIGH: logging.ERROR,
            ThreatLevel.CRITICAL: logging.CRITICAL
        }[severity]
        
        self.security_logger.log(
            log_level,
            f"Security threat detected: {threat_type} from {source_ip} targeting {target_component}"
        )
        
        # Trigger automatic mitigation for high/critical threats
        if severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self._auto_mitigate_threat(threat_event)
    
    def _auto_mitigate_threat(self, threat_event: ThreatEvent):
        """Automatically mitigate high-severity threats."""
        if threat_event.threat_type in ["brute_force_attack", "unauthorized_access_attempt"]:
            # Temporarily block source IP
            self._block_ip(threat_event.source_ip, duration_minutes=60)
            threat_event.mitigated = True
            
        elif threat_event.threat_type == "suspicious_data_access":
            # Revoke potentially compromised sessions
            self._revoke_sessions_from_ip(threat_event.source_ip)
            threat_event.mitigated = True
    
    def _block_ip(self, ip_address: str, duration_minutes: int):
        """Block IP address for specified duration."""
        # In a real implementation, this would update firewall rules
        self.security_logger.warning(f"Blocked IP {ip_address} for {duration_minutes} minutes")
    
    def _revoke_sessions_from_ip(self, ip_address: str):
        """Revoke all active sessions from specific IP."""
        with self.session_lock:
            sessions_to_revoke = [
                session_id for session_id, session in self.active_sessions.items()
                if session["source_ip"] == ip_address
            ]
            
            for session_id in sessions_to_revoke:
                del self.active_sessions[session_id]
            
            self.security_logger.warning(f"Revoked {len(sessions_to_revoke)} sessions from IP {ip_address}")
    
    def encrypt_data(self, data: Union[str, bytes, np.ndarray]) -> bytes:
        """
        Encrypt sensitive data using AES-256.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data with IV prepended
        """
        from cryptography.fernet import Fernet
        
        # Convert data to bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        else:
            data_bytes = data
        
        # Generate Fernet key from encryption key
        fernet_key = base64.urlsafe_b64encode(self.encryption_key)
        fernet = Fernet(fernet_key)
        
        # Encrypt data
        encrypted_data = fernet.encrypt(data_bytes)
        
        return encrypted_data
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using AES-256.
        
        Args:
            encrypted_data: Encrypted data with IV prepended
            
        Returns:
            Decrypted data
        """
        from cryptography.fernet import Fernet
        
        # Generate Fernet key from encryption key
        fernet_key = base64.urlsafe_b64encode(self.encryption_key)
        fernet = Fernet(fernet_key)
        
        # Decrypt data
        decrypted_data = fernet.decrypt(encrypted_data)
        
        return decrypted_data
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        current_time = datetime.utcnow()
        
        # Count recent threats
        recent_threats = [
            t for t in self.threat_log
            if (current_time - t.timestamp).total_seconds() < 3600
        ]
        
        threat_counts = {}
        for threat in recent_threats:
            threat_counts[threat.severity.value] = threat_counts.get(threat.severity.value, 0) + 1
        
        return {
            "active_sessions": len(self.active_sessions),
            "total_credentials": len(self.credentials_store),
            "recent_threats": len(recent_threats),
            "threat_breakdown": threat_counts,
            "security_level": "operational",
            "last_threat": self.threat_log[-1].timestamp.isoformat() if self.threat_log else None
        }


class ThreatDetector:
    """
    Advanced threat detection system for photonic AI.
    
    Monitors system behavior and detects potential security threats
    using pattern analysis and anomaly detection.
    """
    
    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_thresholds = {
            "request_rate": 2.0,  # Standard deviations
            "error_rate": 3.0,
            "response_time": 2.5,
            "data_access_pattern": 2.0
        }
        
        self.logger = logging.getLogger("threat_detector")
    
    def analyze_request_pattern(self, request_data: Dict[str, Any]) -> Optional[ThreatEvent]:
        """Analyze request patterns for anomalies."""
        # Simplified anomaly detection
        request_rate = request_data.get("requests_per_minute", 0)
        error_rate = request_data.get("error_rate", 0)
        
        # Check for potential DDoS
        if request_rate > 1000:  # Very high request rate
            return ThreatEvent(
                threat_id=secrets.token_hex(8),
                threat_type="potential_ddos",
                severity=ThreatLevel.HIGH,
                source_ip=request_data.get("source_ip", "unknown"),
                target_component="api_gateway",
                timestamp=datetime.utcnow(),
                details={"request_rate": request_rate}
            )
        
        # Check for potential brute force
        if error_rate > 0.5:  # High error rate
            return ThreatEvent(
                threat_id=secrets.token_hex(8),
                threat_type="potential_brute_force",
                severity=ThreatLevel.MEDIUM,
                source_ip=request_data.get("source_ip", "unknown"),
                target_component="authentication",
                timestamp=datetime.utcnow(),
                details={"error_rate": error_rate}
            )
        
        return None
    
    def analyze_data_access(self, access_data: Dict[str, Any]) -> Optional[ThreatEvent]:
        """Analyze data access patterns for suspicious behavior."""
        # Check for unusual data volume access
        data_volume = access_data.get("data_volume_mb", 0)
        access_time = access_data.get("access_time_seconds", 0)
        
        if data_volume > 1000:  # Large data access
            return ThreatEvent(
                threat_id=secrets.token_hex(8),
                threat_type="suspicious_data_access",
                severity=ThreatLevel.MEDIUM,
                source_ip=access_data.get("source_ip", "unknown"),
                target_component="data_layer",
                timestamp=datetime.utcnow(),
                details={"data_volume_mb": data_volume, "access_time_s": access_time}
            )
        
        return None


class SecureDataHandler:
    """
    Secure data handling for photonic AI systems.
    
    Provides secure storage, transmission, and processing of sensitive
    data including model weights, training data, and user information.
    """
    
    def __init__(self, security_manager: PhotonicSecurityManager):
        self.security_manager = security_manager
        self.logger = logging.getLogger("secure_data_handler")
    
    def secure_model_weights(self, weights: np.ndarray, 
                           security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL) -> bytes:
        """
        Securely serialize and encrypt model weights.
        
        Args:
            weights: Neural network weights
            security_level: Security classification
            
        Returns:
            Encrypted weight data
        """
        # Add metadata
        weight_data = {
            "weights": weights,
            "shape": weights.shape,
            "dtype": str(weights.dtype),
            "security_level": security_level.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checksum": hashlib.sha256(weights.tobytes()).hexdigest()
        }
        
        # Serialize
        import pickle
        serialized_data = pickle.dumps(weight_data)
        
        # Encrypt
        encrypted_data = self.security_manager.encrypt_data(serialized_data)
        
        self.logger.info(f"Secured model weights with {security_level.value} classification")
        
        return encrypted_data
    
    def load_secure_weights(self, encrypted_data: bytes, 
                          expected_checksum: str = None) -> np.ndarray:
        """
        Load and verify secure model weights.
        
        Args:
            encrypted_data: Encrypted weight data
            expected_checksum: Expected data checksum for verification
            
        Returns:
            Decrypted model weights
        """
        try:
            # Decrypt
            decrypted_data = self.security_manager.decrypt_data(encrypted_data)
            
            # Deserialize
            import pickle
            weight_data = pickle.loads(decrypted_data)
            
            # Verify checksum
            weights = weight_data["weights"]
            actual_checksum = hashlib.sha256(weights.tobytes()).hexdigest()
            
            if expected_checksum and actual_checksum != expected_checksum:
                raise ValueError("Weight data integrity check failed")
            
            stored_checksum = weight_data.get("checksum")
            if stored_checksum and actual_checksum != stored_checksum:
                raise ValueError("Stored weight data integrity check failed")
            
            self.logger.info(f"Successfully loaded secure weights with shape {weights.shape}")
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Failed to load secure weights: {e}")
            raise
    
    def sanitize_input_data(self, data: np.ndarray) -> np.ndarray:
        """
        Sanitize input data to prevent attacks.
        
        Args:
            data: Raw input data
            
        Returns:
            Sanitized data
        """
        # Check for malicious patterns
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            self.logger.warning("Input data contains NaN or infinite values")
            # Replace with safe values
            data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Check for extremely large values (potential overflow attack)
        if np.any(np.abs(data) > 1e10):
            self.logger.warning("Input data contains extremely large values")
            # Clip to safe range
            data = np.clip(data, -1e6, 1e6)
        
        # Check for suspicious patterns (all zeros, all ones, etc.)
        if np.all(data == 0) or np.all(data == 1):
            self.logger.warning("Input data has suspicious uniform pattern")
        
        return data
    
    def secure_federated_communication(self, message: Dict[str, Any], 
                                     recipient_public_key: str = None) -> bytes:
        """
        Secure communication for federated learning.
        
        Args:
            message: Message to send
            recipient_public_key: Public key of recipient
            
        Returns:
            Encrypted message
        """
        # Add message metadata
        secure_message = {
            "payload": message,
            "sender": "photonic_ai_system",
            "timestamp": datetime.utcnow().isoformat(),
            "message_id": secrets.token_hex(16)
        }
        
        # Sign message
        message_bytes = json.dumps(secure_message, sort_keys=True).encode()
        signature = hmac.new(
            self.security_manager.master_key,
            message_bytes,
            hashlib.sha256
        ).hexdigest()
        
        secure_message["signature"] = signature
        
        # Encrypt
        encrypted_message = self.security_manager.encrypt_data(
            json.dumps(secure_message).encode()
        )
        
        return encrypted_message
    
    def verify_federated_message(self, encrypted_message: bytes) -> Dict[str, Any]:
        """
        Verify and decrypt federated learning message.
        
        Args:
            encrypted_message: Encrypted message
            
        Returns:
            Verified message payload
        """
        try:
            # Decrypt
            decrypted_data = self.security_manager.decrypt_data(encrypted_message)
            message = json.loads(decrypted_data.decode())
            
            # Verify signature
            payload = message.copy()
            signature = payload.pop("signature")
            
            message_bytes = json.dumps(payload, sort_keys=True).encode()
            expected_signature = hmac.new(
                self.security_manager.master_key,
                message_bytes,
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                raise ValueError("Message signature verification failed")
            
            # Check message age
            timestamp = datetime.fromisoformat(message["timestamp"])
            age = datetime.utcnow() - timestamp
            
            if age.total_seconds() > 300:  # 5 minute expiry
                raise ValueError("Message has expired")
            
            return message["payload"]
            
        except Exception as e:
            self.logger.error(f"Failed to verify federated message: {e}")
            raise


def secure_photonic_operation(security_level: SecurityLevel = SecurityLevel.INTERNAL,
                             required_permission: str = "photonic.read"):
    """
    Decorator for securing photonic AI operations.
    
    Args:
        security_level: Required security clearance
        required_permission: Required permission
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract credentials from kwargs or context
            credentials = kwargs.pop('credentials', None)
            
            if not credentials:
                raise PermissionError("Authentication required for photonic operations")
            
            # Create security manager (in practice, this would be injected)
            security_manager = PhotonicSecurityManager()
            
            # Authorize operation
            if not security_manager.authorize(credentials, required_permission, security_level):
                raise PermissionError(f"Insufficient permissions for {func.__name__}")
            
            # Execute with security context
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Initialize security system
    security_manager = PhotonicSecurityManager()
    
    # Create test credentials
    test_credentials = security_manager.create_credentials(
        user_id="test_user",
        security_level=SecurityLevel.CONFIDENTIAL,
        permissions=["photonic.read", "photonic.write", "quantum.access"]
    )
    
    print(f"Created credentials: {test_credentials.api_key}")
    
    # Test authentication
    auth_result = security_manager.authenticate(test_credentials.api_key, "192.168.1.100")
    print(f"Authentication result: {auth_result is not None}")
    
    # Test authorization
    auth_success = security_manager.authorize(
        test_credentials,
        "photonic.read",
        SecurityLevel.INTERNAL
    )
    print(f"Authorization result: {auth_success}")
    
    # Test data encryption
    test_data = np.random.randn(100, 100)
    encrypted = security_manager.encrypt_data(test_data)
    decrypted = security_manager.decrypt_data(encrypted)
    print(f"Encryption test: {np.array_equal(test_data.tobytes(), decrypted)}")
    
    # Security status
    status = security_manager.get_security_status()
    print(f"Security status: {status}")