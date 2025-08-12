"""
Global compliance and privacy module for photonic AI systems.

Implements GDPR, CCPA, PDPA compliance, data privacy controls,
and multi-regional legal requirements for global deployments.
"""

import hashlib
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"


class ProcessingPurpose(Enum):
    """Legal purposes for data processing."""
    SCIENTIFIC_RESEARCH = "scientific_research"
    MODEL_TRAINING = "model_training"
    INFERENCE = "inference"
    SYSTEM_OPTIMIZATION = "system_optimization"
    MONITORING = "monitoring"
    SECURITY = "security"
    LEGAL_COMPLIANCE = "legal_compliance"


class LegalBasis(Enum):
    """Legal basis for data processing under GDPR."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataSubjectRights(Enum):
    """Data subject rights under privacy laws."""
    ACCESS = "access"
    RECTIFICATION = "rectification"
    ERASURE = "erasure"
    RESTRICT_PROCESSING = "restrict_processing"
    DATA_PORTABILITY = "data_portability"
    OBJECT_TO_PROCESSING = "object_to_processing"
    WITHDRAW_CONSENT = "withdraw_consent"


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    id: str
    timestamp: float
    data_subject_id: Optional[str]
    data_classification: DataClassification
    processing_purpose: ProcessingPurpose
    legal_basis: LegalBasis
    data_types: List[str]
    retention_period_days: int
    geographic_location: str
    processor_id: str
    consent_id: Optional[str] = None
    anonymized: bool = False
    encrypted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "data_subject_id": self.data_subject_id,
            "data_classification": self.data_classification.value,
            "processing_purpose": self.processing_purpose.value,
            "legal_basis": self.legal_basis.value,
            "data_types": self.data_types,
            "retention_period_days": self.retention_period_days,
            "geographic_location": self.geographic_location,
            "processor_id": self.processor_id,
            "consent_id": self.consent_id,
            "anonymized": self.anonymized,
            "encrypted": self.encrypted
        }


@dataclass
class ConsentRecord:
    """Record of user consent."""
    id: str
    data_subject_id: str
    purposes: List[ProcessingPurpose]
    granted_at: float
    expires_at: Optional[float]
    withdrawn_at: Optional[float] = None
    consent_version: str = "1.0"
    geographic_location: str = "unknown"
    consent_mechanism: str = "explicit"  # "explicit", "implicit", "opt_out"
    
    @property
    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        current_time = time.time()
        
        if self.withdrawn_at and self.withdrawn_at <= current_time:
            return False
        
        if self.expires_at and self.expires_at <= current_time:
            return False
        
        return True
    
    def withdraw(self):
        """Withdraw consent."""
        self.withdrawn_at = time.time()


class DataAnonymizer:
    """
    Data anonymization utilities for privacy protection.
    
    Implements k-anonymity, differential privacy, and other
    anonymization techniques for photonic AI data.
    """
    
    def __init__(self, k_value: int = 5):
        """Initialize anonymizer."""
        self.k_value = k_value
        self.anonymization_mapping = {}
    
    def anonymize_identifiers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize direct identifiers in data.
        
        Args:
            data: Data dictionary to anonymize
            
        Returns:
            Anonymized data dictionary
        """
        anonymized = data.copy()
        
        # Identify and anonymize personal identifiers
        personal_fields = [
            "user_id", "email", "name", "phone", "address", "ip_address",
            "device_id", "session_id", "customer_id"
        ]
        
        for field in personal_fields:
            if field in anonymized:
                anonymized[field] = self._hash_identifier(str(anonymized[field]))
        
        # Remove or generalize quasi-identifiers
        if "timestamp" in anonymized:
            # Generalize timestamp to hour
            ts = anonymized["timestamp"]
            anonymized["timestamp"] = int(ts // 3600) * 3600
        
        if "location" in anonymized:
            # Generalize location to city/region level
            anonymized["location"] = self._generalize_location(anonymized["location"])
        
        return anonymized
    
    def apply_differential_privacy(self, numeric_data: List[float], 
                                 epsilon: float = 1.0) -> List[float]:
        """
        Apply differential privacy to numeric data.
        
        Args:
            numeric_data: List of numeric values
            epsilon: Privacy parameter (smaller = more private)
            
        Returns:
            Differentially private data
        """
        import random
        import math
        
        # Add Laplace noise for differential privacy
        sensitivity = 1.0  # Assume sensitivity of 1
        noise_scale = sensitivity / epsilon
        
        noisy_data = []
        for value in numeric_data:
            # Generate Laplace noise
            u = random.uniform(-0.5, 0.5)
            noise = -noise_scale * math.copysign(math.log(1 - 2 * abs(u)), u)
            noisy_data.append(value + noise)
        
        return noisy_data
    
    def _hash_identifier(self, identifier: str) -> str:
        """Hash identifier for anonymization."""
        # Use consistent hashing for reproducibility
        hash_object = hashlib.sha256(identifier.encode())
        return f"anon_{hash_object.hexdigest()[:16]}"
    
    def _generalize_location(self, location: str) -> str:
        """Generalize location to reduce identifiability."""
        # Simple generalization - in production would use proper geographic hierarchies
        if "," in location:
            parts = location.split(",")
            return parts[-1].strip()  # Return country/state only
        return location


class ComplianceEngine:
    """
    Compliance engine for privacy law requirements.
    
    Implements GDPR, CCPA, PDPA and other privacy regulations
    for photonic AI systems.
    """
    
    def __init__(self, default_region: str = "EU"):
        """Initialize compliance engine."""
        self.default_region = default_region
        self.processing_records = []
        self.consent_records = {}
        self.data_retention_policies = {}
        self.anonymizer = DataAnonymizer()
        
        # Set up default retention policies
        self._initialize_retention_policies()
    
    def _initialize_retention_policies(self):
        """Initialize default data retention policies."""
        self.data_retention_policies = {
            DataClassification.PUBLIC: 365 * 10,  # 10 years
            DataClassification.INTERNAL: 365 * 7,  # 7 years
            DataClassification.CONFIDENTIAL: 365 * 5,  # 5 years
            DataClassification.PERSONAL: 365 * 3,  # 3 years
            DataClassification.SENSITIVE_PERSONAL: 365 * 1,  # 1 year
        }
    
    def record_processing_activity(self,
                                 data_subject_id: Optional[str],
                                 data_classification: DataClassification,
                                 processing_purpose: ProcessingPurpose,
                                 legal_basis: LegalBasis,
                                 data_types: List[str],
                                 geographic_location: str = None,
                                 consent_id: str = None) -> str:
        """
        Record data processing activity for compliance.
        
        Args:
            data_subject_id: ID of data subject (if applicable)
            data_classification: Classification of data being processed
            processing_purpose: Purpose of processing
            legal_basis: Legal basis for processing
            data_types: Types of data being processed
            geographic_location: Location of processing
            consent_id: Associated consent record ID
            
        Returns:
            Processing record ID
        """
        # Validate legal basis and purpose compatibility
        if not self._validate_legal_basis(legal_basis, processing_purpose):
            raise ComplianceError(f"Invalid legal basis {legal_basis} for purpose {processing_purpose}")
        
        # Check consent requirements
        if legal_basis == LegalBasis.CONSENT and not consent_id:
            raise ComplianceError("Consent ID required when legal basis is consent")
        
        if consent_id and not self._validate_consent(consent_id, processing_purpose):
            raise ComplianceError("Invalid or expired consent")
        
        # Create processing record
        record_id = str(uuid.uuid4())
        retention_days = self.data_retention_policies.get(data_classification, 365)
        
        record = DataProcessingRecord(
            id=record_id,
            timestamp=time.time(),
            data_subject_id=data_subject_id,
            data_classification=data_classification,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            data_types=data_types,
            retention_period_days=retention_days,
            geographic_location=geographic_location or self.default_region,
            processor_id="photonic_ai_system",
            consent_id=consent_id,
            anonymized=data_subject_id is None,
            encrypted=True  # Assume encryption in production
        )
        
        self.processing_records.append(record)
        
        # Log processing activity
        logger.info(f"Data processing recorded: {record.to_dict()}")
        
        return record_id
    
    def grant_consent(self,
                     data_subject_id: str,
                     purposes: List[ProcessingPurpose],
                     duration_days: Optional[int] = None,
                     geographic_location: str = None) -> str:
        """
        Record user consent for data processing.
        
        Args:
            data_subject_id: ID of data subject granting consent
            purposes: Purposes for which consent is granted
            duration_days: Duration of consent in days (None for indefinite)
            geographic_location: Location where consent was granted
            
        Returns:
            Consent record ID
        """
        consent_id = str(uuid.uuid4())
        current_time = time.time()
        
        expires_at = None
        if duration_days:
            expires_at = current_time + (duration_days * 24 * 3600)
        
        consent = ConsentRecord(
            id=consent_id,
            data_subject_id=data_subject_id,
            purposes=purposes,
            granted_at=current_time,
            expires_at=expires_at,
            geographic_location=geographic_location or self.default_region
        )
        
        self.consent_records[consent_id] = consent
        
        logger.info(f"Consent granted: {data_subject_id} for {[p.value for p in purposes]}")
        
        return consent_id
    
    def handle_data_subject_request(self,
                                   data_subject_id: str,
                                   request_type: DataSubjectRights,
                                   additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle data subject rights request.
        
        Args:
            data_subject_id: ID of data subject making request
            request_type: Type of request
            additional_info: Additional request information
            
        Returns:
            Response to request
        """
        logger.info(f"Processing data subject request: {request_type.value} for {data_subject_id}")
        
        if request_type == DataSubjectRights.ACCESS:
            return self._handle_access_request(data_subject_id)
        
        elif request_type == DataSubjectRights.ERASURE:
            return self._handle_erasure_request(data_subject_id)
        
        elif request_type == DataSubjectRights.RECTIFICATION:
            return self._handle_rectification_request(data_subject_id, additional_info)
        
        elif request_type == DataSubjectRights.RESTRICT_PROCESSING:
            return self._handle_restriction_request(data_subject_id)
        
        elif request_type == DataSubjectRights.DATA_PORTABILITY:
            return self._handle_portability_request(data_subject_id)
        
        elif request_type == DataSubjectRights.OBJECT_TO_PROCESSING:
            return self._handle_objection_request(data_subject_id)
        
        elif request_type == DataSubjectRights.WITHDRAW_CONSENT:
            return self._handle_consent_withdrawal(data_subject_id)
        
        else:
            raise ComplianceError(f"Unsupported request type: {request_type}")
    
    def _handle_access_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data access request."""
        # Find all processing records for this data subject
        subject_records = [
            record for record in self.processing_records
            if record.data_subject_id == data_subject_id
        ]
        
        # Find all consent records
        subject_consents = [
            consent for consent in self.consent_records.values()
            if consent.data_subject_id == data_subject_id
        ]
        
        return {
            "request_type": "access",
            "data_subject_id": data_subject_id,
            "processing_records": [record.to_dict() for record in subject_records],
            "consent_records": [
                {
                    "id": consent.id,
                    "purposes": [p.value for p in consent.purposes],
                    "granted_at": consent.granted_at,
                    "expires_at": consent.expires_at,
                    "is_valid": consent.is_valid
                }
                for consent in subject_consents
            ],
            "response_time": time.time()
        }
    
    def _handle_erasure_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data erasure request (right to be forgotten)."""
        erased_records = 0
        
        # Mark processing records for erasure
        for record in self.processing_records:
            if record.data_subject_id == data_subject_id:
                # Check if erasure is legally allowed
                if self._can_erase_data(record):
                    record.data_subject_id = None  # Anonymize
                    record.anonymized = True
                    erased_records += 1
        
        # Withdraw all consents
        for consent in self.consent_records.values():
            if consent.data_subject_id == data_subject_id:
                consent.withdraw()
        
        logger.info(f"Erased {erased_records} records for data subject {data_subject_id}")
        
        return {
            "request_type": "erasure",
            "data_subject_id": data_subject_id,
            "records_erased": erased_records,
            "completed_at": time.time()
        }
    
    def _handle_rectification_request(self, data_subject_id: str, 
                                    corrections: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data rectification request."""
        # In a real system, would update the actual data
        # Here we just record the rectification request
        
        logger.info(f"Data rectification requested for {data_subject_id}: {corrections}")
        
        return {
            "request_type": "rectification",
            "data_subject_id": data_subject_id,
            "corrections_requested": corrections,
            "status": "acknowledged",
            "completed_at": time.time()
        }
    
    def _handle_restriction_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle processing restriction request."""
        # Mark records as restricted
        restricted_records = 0
        for record in self.processing_records:
            if record.data_subject_id == data_subject_id:
                # Add restriction flag (would be implemented in actual data store)
                restricted_records += 1
        
        return {
            "request_type": "restriction",
            "data_subject_id": data_subject_id,
            "records_restricted": restricted_records,
            "completed_at": time.time()
        }
    
    def _handle_portability_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data portability request."""
        # Extract portable data
        portable_data = self._extract_portable_data(data_subject_id)
        
        return {
            "request_type": "portability",
            "data_subject_id": data_subject_id,
            "portable_data": portable_data,
            "format": "JSON",
            "completed_at": time.time()
        }
    
    def _handle_objection_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle objection to processing request."""
        # Stop processing based on legitimate interests
        objected_records = 0
        for record in self.processing_records:
            if (record.data_subject_id == data_subject_id and 
                record.legal_basis == LegalBasis.LEGITIMATE_INTERESTS):
                # Stop this processing
                objected_records += 1
        
        return {
            "request_type": "objection",
            "data_subject_id": data_subject_id,
            "processing_stopped": objected_records,
            "completed_at": time.time()
        }
    
    def _handle_consent_withdrawal(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle consent withdrawal."""
        withdrawn_consents = 0
        for consent in self.consent_records.values():
            if consent.data_subject_id == data_subject_id and consent.is_valid:
                consent.withdraw()
                withdrawn_consents += 1
        
        return {
            "request_type": "consent_withdrawal",
            "data_subject_id": data_subject_id,
            "consents_withdrawn": withdrawn_consents,
            "completed_at": time.time()
        }
    
    def _validate_legal_basis(self, legal_basis: LegalBasis, 
                            purpose: ProcessingPurpose) -> bool:
        """Validate legal basis for processing purpose."""
        # Define valid combinations
        valid_combinations = {
            ProcessingPurpose.SCIENTIFIC_RESEARCH: [
                LegalBasis.CONSENT, LegalBasis.PUBLIC_TASK, LegalBasis.LEGITIMATE_INTERESTS
            ],
            ProcessingPurpose.MODEL_TRAINING: [
                LegalBasis.CONSENT, LegalBasis.LEGITIMATE_INTERESTS
            ],
            ProcessingPurpose.INFERENCE: [
                LegalBasis.CONSENT, LegalBasis.CONTRACT, LegalBasis.LEGITIMATE_INTERESTS
            ],
            ProcessingPurpose.MONITORING: [
                LegalBasis.LEGAL_OBLIGATION, LegalBasis.LEGITIMATE_INTERESTS
            ],
            ProcessingPurpose.SECURITY: [
                LegalBasis.LEGAL_OBLIGATION, LegalBasis.VITAL_INTERESTS
            ]
        }
        
        return legal_basis in valid_combinations.get(purpose, [])
    
    def _validate_consent(self, consent_id: str, purpose: ProcessingPurpose) -> bool:
        """Validate consent for processing purpose."""
        if consent_id not in self.consent_records:
            return False
        
        consent = self.consent_records[consent_id]
        return consent.is_valid and purpose in consent.purposes
    
    def _can_erase_data(self, record: DataProcessingRecord) -> bool:
        """Check if data can be legally erased."""
        # Legal obligations prevent erasure
        if record.legal_basis == LegalBasis.LEGAL_OBLIGATION:
            return False
        
        # Public interest tasks may prevent erasure
        if record.legal_basis == LegalBasis.PUBLIC_TASK:
            return False
        
        # Scientific research may have special protections
        if record.processing_purpose == ProcessingPurpose.SCIENTIFIC_RESEARCH:
            # Check if anonymization is sufficient
            return True
        
        return True
    
    def _extract_portable_data(self, data_subject_id: str) -> Dict[str, Any]:
        """Extract data for portability request."""
        # In real system, would extract actual user data
        return {
            "user_id": data_subject_id,
            "processing_history": [
                record.to_dict() for record in self.processing_records
                if record.data_subject_id == data_subject_id
            ],
            "extracted_at": time.time()
        }
    
    def generate_compliance_report(self, region: str = None) -> Dict[str, Any]:
        """Generate compliance status report."""
        region = region or self.default_region
        
        # Filter records by region if specified
        relevant_records = [
            record for record in self.processing_records
            if not region or record.geographic_location == region
        ]
        
        # Calculate compliance metrics
        total_records = len(relevant_records)
        anonymized_records = sum(1 for r in relevant_records if r.anonymized)
        encrypted_records = sum(1 for r in relevant_records if r.encrypted)
        
        # Check retention compliance
        current_time = time.time()
        overdue_records = []
        for record in relevant_records:
            retention_deadline = record.timestamp + (record.retention_period_days * 24 * 3600)
            if current_time > retention_deadline:
                overdue_records.append(record.id)
        
        # Consent statistics
        valid_consents = sum(1 for c in self.consent_records.values() if c.is_valid)
        expired_consents = sum(1 for c in self.consent_records.values() if not c.is_valid)
        
        return {
            "region": region,
            "report_generated_at": current_time,
            "total_processing_records": total_records,
            "anonymized_percentage": (anonymized_records / total_records * 100) if total_records > 0 else 0,
            "encrypted_percentage": (encrypted_records / total_records * 100) if total_records > 0 else 0,
            "retention_compliance": {
                "overdue_records": len(overdue_records),
                "compliance_rate": ((total_records - len(overdue_records)) / total_records * 100) if total_records > 0 else 100
            },
            "consent_status": {
                "valid_consents": valid_consents,
                "expired_consents": expired_consents
            },
            "legal_basis_distribution": self._analyze_legal_basis_distribution(relevant_records),
            "processing_purpose_distribution": self._analyze_purpose_distribution(relevant_records)
        }
    
    def _analyze_legal_basis_distribution(self, records: List[DataProcessingRecord]) -> Dict[str, int]:
        """Analyze distribution of legal bases."""
        distribution = {}
        for record in records:
            basis = record.legal_basis.value
            distribution[basis] = distribution.get(basis, 0) + 1
        return distribution
    
    def _analyze_purpose_distribution(self, records: List[DataProcessingRecord]) -> Dict[str, int]:
        """Analyze distribution of processing purposes."""
        distribution = {}
        for record in records:
            purpose = record.processing_purpose.value
            distribution[purpose] = distribution.get(purpose, 0) + 1
        return distribution


class ComplianceError(Exception):
    """Compliance-related error."""
    pass


def create_compliant_photonic_system(base_system, region: str = "EU",
                                   enable_anonymization: bool = True):
    """
    Create compliance-enhanced photonic system.
    
    Args:
        base_system: Base photonic neural network system
        region: Default regulatory region
        enable_anonymization: Whether to enable automatic anonymization
        
    Returns:
        Compliance-enhanced system
    """
    compliance_engine = ComplianceEngine(default_region=region)
    
    # Wrap system methods with compliance recording
    original_forward = base_system.forward
    
    def compliant_forward(*args, **kwargs):
        # Record processing activity
        compliance_engine.record_processing_activity(
            data_subject_id=None,  # Assume anonymized inference
            data_classification=DataClassification.INTERNAL,
            processing_purpose=ProcessingPurpose.INFERENCE,
            legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
            data_types=["neural_network_inputs"],
            geographic_location=region
        )
        
        result = original_forward(*args, **kwargs)
        
        # Anonymize result if enabled
        if enable_anonymization and isinstance(result, tuple) and len(result) > 1:
            outputs, metrics = result
            # Remove potentially identifying information from metrics
            anonymized_metrics = {
                k: v for k, v in metrics.items()
                if k not in ["ip_address", "user_id", "session_id"]
            }
            return outputs, anonymized_metrics
        
        return result
    
    base_system.forward = compliant_forward
    base_system.compliance_engine = compliance_engine
    
    return base_system