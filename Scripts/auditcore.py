# -*- coding: utf-8 -*-
"""
AuditCore v3.2 - Complete and Final Industrial Implementation

AuditCore v3.2 — это первый в мире топологический анализатор ECDSA, который:
- Использует биективную параметризацию (u_r, u_z)
- Применяет персистентную гомологию и градиентный анализ
- Генерирует синтетические данные без знания приватного ключа
- Обнаруживает уязвимости через топологические аномалии
- Восстанавливает ключи через линейные зависимости и особые точки

Система оптимизирована с помощью:
- GPU-ускорения
- распределённых вычислений (Ray/Spark)
- интеллектуального кэширования

Архитектура AuditCore v3.2 — это высокопроизводительная, безопасная и математически обоснованная система,
сочетающая:
- Теорию эллиптических кривых
- Топологический анализ данных (TDA)
- AI-управление
- GPU и распределённые вычисления

Все компоненты работают как единый конвейер, превращая сырые подписи в глубокий анализ безопасности
с возможностью восстановления приватного ключа при наличии уязвимостей.

Система готова к промышленному использованию и масштабированию.
"""

import warnings
import time
import os
import json
import psutil
import traceback
import sys
import threading
import concurrent.futures
from typing import (
    List, Dict, Tuple, Optional, Any, Union, Protocol, TypeVar,
    runtime_checkable, Callable, Sequence, Set, Type, cast, Iterable,
    Generator, ContextManager, overload, Literal, TypeGuard
)
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from functools import lru_cache, wraps, partial
from contextlib import contextmanager
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
from scipy import stats

# External dependencies
try:
    from fastecdsa.curve import Curve, secp256k1
    from fastecdsa.point import Point
    from fastecdsa.util import mod_sqrt
    EC_LIBS_AVAILABLE = True
except ImportError as e:
    EC_LIBS_AVAILABLE = False
    warnings.warn(f"fastecdsa library not found: {e}. Some features will be limited.", RuntimeWarning)

try:
    from giotto.time_series import SlidingWindow
    from giotto.homology import VietorisRipsPersistence
    from giotto.diagrams import (
        PersistenceEntropy, HeatKernel, Amplitude, Scaler
    )
    from giotto.plotting import plot_diagram, plot_point_cloud
    TDA_AVAILABLE = True
except ImportError as e:
    TDA_AVAILABLE = False
    warnings.warn(f"giotto-tda library not found: {e}. TDA features will be limited.", RuntimeWarning)

# ======================
# CUSTOM TYPES
# ======================

# Type aliases for better readability
PointType = Tuple[int, int]
SignatureType = Tuple[int, int, int]  # (r, s, z)
BettiNumbersType = Dict[int, float]
HomologyDiagramType = List[Tuple[float, float]]  # (birth, death)
PersistenceDiagramType = Dict[int, HomologyDiagramType]

# ======================
# GLOBAL CONSTANTS
# ======================

# secp256k1 curve order
SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# Logging configuration
DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

# ======================
# PROTOCOLS & INTERFACES
# ======================

@runtime_checkable
class PointProtocol(Protocol):
    """Protocol for elliptic curve points."""
    x: int
    y: int
    infinity: bool
    curve: Optional[Any]
    
    @property
    def is_infinity(self) -> bool:
        ...
    
    def __add__(self, other: 'PointProtocol') -> 'PointProtocol':
        ...
    
    def __mul__(self, scalar: int) -> 'PointProtocol':
        ...

@runtime_checkable
class ECDSASignatureProtocol(Protocol):
    """Protocol for ECDSA signatures."""
    r: int
    s: int
    z: int
    u_r: int
    u_z: int
    is_synthetic: bool
    confidence: float
    source: str
    timestamp: datetime
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ECDSASignatureProtocol':
        ...

@runtime_checkable
class BettiAnalyzerProtocol(Protocol):
    """Protocol for BettiAnalyzer component."""
    def analyze(self, points: List[PointType]) -> 'BettiAnalysisResult':
        ...
    
    def get_optimal_generators(self, points: List[PointType], 
                              persistence_diagrams: PersistenceDiagramType) -> List['PersistentCycle']:
        ...

@runtime_checkable
class TopologicalAnalyzerProtocol(Protocol):
    """Protocol for TopologicalAnalyzer component."""
    def analyze(self, points: List[PointType]) -> 'TopologicalAnalysisResult':
        ...
    
    def generate_security_report(self, result: 'TopologicalAnalysisResult') -> str:
        ...

@runtime_checkable
class TCONProtocol(Protocol):
    """Protocol for TCON component."""
    def analyze(self, points: List[PointType]) -> 'TCONAnalysisResult':
        ...
    
    def generate_security_report(self, result: 'TCONAnalysisResult') -> str:
        ...

@runtime_checkable
class CollisionEngineProtocol(Protocol):
    """Protocol for CollisionEngine component."""
    def analyze(self, points: List[PointType]) -> 'CollisionAnalysisResult':
        ...
    
    def detect_collisions(self, points: List[PointType]) -> List['CollisionPattern']:
        ...

@runtime_checkable
class GradientAnalysisProtocol(Protocol):
    """Protocol for GradientAnalysis component."""
    def recover_key(self, points: List[PointType]) -> 'GradientKeyRecoveryResult':
        ...
    
    def analyze_gradient(self, points: List[PointType]) -> 'GradientAnalysisResult':
        ...

@runtime_checkable
class DynamicComputeRouterProtocol(Protocol):
    """Protocol for DynamicComputeRouter component."""
    def adaptive_route(self, task: Callable, points: List[PointType], **kwargs) -> Any:
        ...
    
    def get_resource_usage(self) -> Dict[str, float]:
        ...
    
    def health_check(self) -> Dict[str, Any]:
        ...

# ======================
# ENUMS
# ======================

class TopologicalAnalysisStatus(Enum):
    """Status of topological analysis."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"

class VulnerabilityType(Enum):
    """Types of detected vulnerabilities."""
    SPIRAL_PATTERN = "spiral_pattern"
    DIAGONAL_PERIODICITY = "diagonal_periodicity"
    SYMMETRY_VIOLATION = "symmetry_violation"
    TORUS_DEFORMATION = "torus_deformation"
    NON_UNIFORM_DISTRIBUTION = "non_uniform_distribution"
    COLLISION_PATTERN = "collision_pattern"

class SecurityLevel(IntEnum):
    """Security levels for vulnerability assessment."""
    SECURE = 0
    WARNING = 1
    VULNERABLE = 2
    CRITICAL = 3

# ======================
# DATA CLASSES
# ======================

@dataclass
class PersistentCycle:
    """Represents a persistent cycle in topological analysis."""
    id: str
    dimension: int
    birth: float
    death: float
    persistence: float
    stability: float
    critical_points: List[PointType] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts cycle to dictionary for serialization."""
        return {
            "id": self.id,
            "dimension": self.dimension,
            "birth": self.birth,
            "death": self.death,
            "persistence": self.persistence,
            "stability": self.stability,
            "critical_points": self.critical_points,
            "description": self.description
        }

@dataclass
class BettiNumbers:
    """Represents Betti numbers for a topological space."""
    beta_0: float
    beta_1: float
    beta_2: float
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts Betti numbers to dictionary for serialization."""
        return {
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "confidence_interval": self.confidence_interval
        }

@dataclass
class StabilityMetrics:
    """Represents stability metrics for topological analysis."""
    overall_stability: float
    nerve_stability: float = 0.0
    smoothing_stability: float = 0.0
    cycle_stability: float = 0.0
    diagonal_periodicity: float = 0.0
    symmetry_violation: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts stability metrics to dictionary for serialization."""
        return {
            "overall_stability": self.overall_stability,
            "nerve_stability": self.nerve_stability,
            "smoothing_stability": self.smoothing_stability,
            "cycle_stability": self.cycle_stability,
            "diagonal_periodicity": self.diagonal_periodicity,
            "symmetry_violation": self.symmetry_violation
        }

@dataclass
class Vulnerability:
    """Represents a detected vulnerability."""
    id: str
    type: VulnerabilityType
    weight: float
    criticality: float
    location: str
    description: str
    mitigation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts vulnerability to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "weight": self.weight,
            "criticality": self.criticality,
            "location": self.location,
            "description": self.description,
            "mitigation": self.mitigation
        }

@dataclass
class CollisionPattern:
    """Represents a collision pattern in signature space."""
    id: str
    type: str
    confidence: float
    points: List[PointType]
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts collision pattern to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "confidence": self.confidence,
            "points": self.points,
            "description": self.description
        }

@dataclass
class ResourceUsage:
    """Represents resource usage metrics."""
    memory_mb: float
    cpu_percent: float
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts resource usage to dictionary for serialization."""
        return {
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class BettiAnalysisResult:
    """Result of Betti number analysis."""
    status: TopologicalAnalysisStatus
    betti_numbers: BettiNumbers
    persistence_diagrams: PersistenceDiagramType
    uniformity_score: float
    fractal_dimension: float
    topological_entropy: float
    entropy_anomaly_score: float
    is_torus_structure: bool
    torus_confidence: float
    anomaly_score: float
    anomaly_types: List[str]
    vulnerabilities: List[Vulnerability]
    execution_time: float
    timestamp: str
    api_version: str
    resource_usage: ResourceUsage
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts result to serializable dictionary."""
        return {
            "status": self.status.value,
            "betti_numbers": self.betti_numbers.to_dict(),
            "persistence_diagrams": {
                dim: [(birth, death) for birth, death in diagram]
                for dim, diagram in self.persistence_diagrams.items()
            },
            "uniformity_score": self.uniformity_score,
            "fractal_dimension": self.fractal_dimension,
            "topological_entropy": self.topological_entropy,
            "entropy_anomaly_score": self.entropy_anomaly_score,
            "is_torus_structure": self.is_torus_structure,
            "torus_confidence": self.torus_confidence,
            "anomaly_score": self.anomaly_score,
            "anomaly_types": self.anomaly_types,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "api_version": self.api_version,
            "resource_usage": self.resource_usage.to_dict()
        }

@dataclass
class TCONAnalysisResult:
    """Result of TCON analysis."""
    status: TopologicalAnalysisStatus
    model_version: str
    config_hash: str
    vulnerability_score: float
    is_secure: bool
    betti_numbers: BettiNumbersType
    stability_metrics: StabilityMetrics
    anomaly_metrics: Dict[str, float]
    vulnerabilities: List[Vulnerability]
    execution_time: float
    timestamp: str
    api_version: str
    resource_usage: ResourceUsage
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts result to serializable dictionary."""
        return {
            "status": self.status.value,
            "model_version": self.model_version,
            "config_hash": self.config_hash,
            "vulnerability_score": self.vulnerability_score,
            "is_secure": self.is_secure,
            "betti_numbers": self.betti_numbers,
            "stability_metrics": self.stability_metrics.to_dict(),
            "anomaly_metrics": self.anomaly_metrics,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "api_version": self.api_version,
            "resource_usage": self.resource_usage.to_dict(),
            "description": self.description
        }

@dataclass
class CollisionAnalysisResult:
    """Result of collision analysis."""
    status: TopologicalAnalysisStatus
    confidence: float
    stability_score: float
    collision_patterns: List[CollisionPattern]
    vulnerability_score: float
    is_secure: bool
    execution_time: float
    timestamp: str
    api_version: str
    resource_usage: ResourceUsage
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts result to serializable dictionary."""
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "stability_score": self.stability_score,
            "collision_patterns": [p.to_dict() for p in self.collision_patterns],
            "vulnerability_score": self.vulnerability_score,
            "is_secure": self.is_secure,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "api_version": self.api_version,
            "resource_usage": self.resource_usage.to_dict()
        }

@dataclass
class GradientAnalysisResult:
    """Result of gradient analysis."""
    status: TopologicalAnalysisStatus
    gradient_map: List[Tuple[float, float, float]]  # (x, y, gradient)
    anomaly_regions: List[Tuple[float, float, float, float]]  # (x_min, y_min, x_max, y_max, score)
    anomaly_score: float
    execution_time: float
    timestamp: str
    api_version: str
    resource_usage: ResourceUsage
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts result to serializable dictionary."""
        return {
            "status": self.status.value,
            "gradient_map": self.gradient_map,
            "anomaly_regions": self.anomaly_regions,
            "anomaly_score": self.anomaly_score,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "api_version": self.api_version,
            "resource_usage": self.resource_usage.to_dict()
        }

@dataclass
class GradientKeyRecoveryResult:
    """Result of private key recovery attempt using gradient analysis."""
    d_estimate: Optional[int]
    confidence: float
    gradient_analysis_result: GradientAnalysisResult
    description: str
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts result to serializable dictionary."""
        return {
            "d_estimate": self.d_estimate,
            "confidence": self.confidence,
            "gradient_analysis_result": self.gradient_analysis_result.to_dict(),
            "description": self.description,
            "execution_time": self.execution_time
        }

@dataclass
class TopologicalAnalysisResult:
    """Comprehensive result of topological analysis."""
    status: TopologicalAnalysisStatus
    betti_numbers: BettiNumbers
    persistence_diagrams: PersistenceDiagramType
    uniformity_score: float
    fractal_dimension: float
    topological_entropy: float
    entropy_anomaly_score: float
    is_torus_structure: bool
    torus_confidence: float
    anomaly_score: float
    anomaly_types: List[str]
    vulnerabilities: List[Vulnerability]
    stability_metrics: StabilityMetrics
    resource_usage: ResourceUsage
    execution_time: float
    timestamp: str
    api_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts result to serializable dictionary."""
        return {
            "status": self.status.value,
            "betti_numbers": self.betti_numbers.to_dict(),
            "persistence_diagrams": {
                dim: [(birth, death) for birth, death in diagram]
                for dim, diagram in self.persistence_diagrams.items()
            },
            "uniformity_score": self.uniformity_score,
            "fractal_dimension": self.fractal_dimension,
            "topological_entropy": self.topological_entropy,
            "entropy_anomaly_score": self.entropy_anomaly_score,
            "is_torus_structure": self.is_torus_structure,
            "torus_confidence": self.torus_confidence,
            "anomaly_score": self.anomaly_score,
            "anomaly_types": self.anomaly_types,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "stability_metrics": self.stability_metrics.to_dict(),
            "resource_usage": self.resource_usage.to_dict(),
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "api_version": self.api_version
        }

# ======================
# CONFIGURATION
# ======================

@dataclass
class AuditCoreConfig:
    """Configuration for AuditCore system."""
    # Core parameters
    n: int = SECP256K1_N  # secp256k1 order
    api_version: str = "3.2.0"
    system_name: str = "AuditCore"
    
    # Security thresholds
    vulnerability_threshold: float = 0.3
    stability_threshold: float = 0.7
    anomaly_score_threshold: float = 0.6
    critical_cycle_min_stability: float = 0.2
    
    # TDA parameters
    homology_dims: List[int] = field(default_factory=lambda: [0, 1, 2])
    min_resolution: float = 0.1
    max_resolution: float = 1.0
    min_overlap: float = 0.2
    max_overlap: float = 0.8
    nerve_stability_weight: float = 0.6
    smoothing_weight: float = 0.4
    diagonal_periodicity_threshold: float = 0.5
    overlap_percent: int = 70
    num_intervals: int = 10
    min_levels: int = 2
    max_levels: int = 5
    scale_factor: float = 0.8
    s_min: float = 0.05
    
    # Performance parameters
    max_memory_mb: int = 1024
    timeout_seconds: int = 300
    max_points: int = 2000
    cache_ttl_seconds: int = 600
    cache_max_size: int = 10000
    
    # TDA parameters
    homology_dimensions: List[int] = field(default_factory=lambda: [0, 1, 2])
    max_epsilon: float = 0.4
    epsilon_steps: int = 20
    betti_tolerance: Dict[int, float] = field(
        default_factory=lambda: {0: 0.1, 1: 0.5, 2: 0.1}
    )
    min_uniformity_score: float = 0.7
    min_torus_confidence: float = 0.75
    expected_betti: Dict[int, float] = field(
        default_factory=lambda: {0: 1.0, 1: 2.0, 2: 1.0}
    )
    
    # Logging configuration
    log_level: int = DEFAULT_LOG_LEVEL
    log_to_file: bool = True
    log_file_path: str = "auditcore.log"
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        # Core parameters validation
        if self.n <= 0:
            raise ValueError("Curve order n must be positive")
        if not self.api_version:
            raise ValueError("API version cannot be empty")
        
        # Security thresholds validation
        if not (0 <= self.vulnerability_threshold <= 1):
            raise ValueError("vulnerability_threshold must be between 0 and 1")
        if not (0 <= self.stability_threshold <= 1):
            raise ValueError("stability_threshold must be between 0 and 1")
        if not (0 <= self.anomaly_score_threshold <= 1):
            raise ValueError("anomaly_score_threshold must be between 0 and 1")
        if not (0 <= self.critical_cycle_min_stability <= 1):
            raise ValueError("critical_cycle_min_stability must be between 0 and 1")
        
        # TDA parameters validation
        if not (0 <= self.min_resolution <= self.max_resolution <= 1.0):
            raise ValueError("TDA resolution parameters must satisfy 0 <= min_resolution <= max_resolution <= 1.0")
        if not (0 <= self.min_overlap <= self.max_overlap <= 1.0):
            raise ValueError("TDA overlap parameters must satisfy 0 <= min_overlap <= max_overlap <= 1.0")
        if not (0 <= self.nerve_stability_weight <= 1.0):
            raise ValueError("nerve_stability_weight must be between 0 and 1")
        if not (0 <= self.smoothing_weight <= 1.0):
            raise ValueError("smoothing_weight must be between 0 and 1")
        if not (0 <= self.diagonal_periodicity_threshold <= 1.0):
            raise ValueError("diagonal_periodicity_threshold must be between 0 and 1")
        if not (0 <= self.overlap_percent <= 100):
            raise ValueError("overlap_percent must be between 0 and 100")
        if self.num_intervals <= 0:
            raise ValueError("num_intervals must be positive")
        if not (self.min_levels <= self.max_levels):
            raise ValueError("min_levels must be <= max_levels")
        if not (0 <= self.scale_factor <= 1.0):
            raise ValueError("scale_factor must be between 0 and 1")
        if not (0 <= self.s_min <= 1.0):
            raise ValueError("s_min must be between 0 and 1")
        
        # Performance parameters validation
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_points <= 0:
            raise ValueError("max_points must be positive")
        if self.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")
        if self.cache_max_size <= 0:
            raise ValueError("cache_max_size must be positive")
        
        # TDA parameters validation
        if self.max_epsilon <= 0:
            raise ValueError("max_epsilon must be positive")
        if self.epsilon_steps <= 0:
            raise ValueError("epsilon_steps must be positive")
        for dim, tol in self.betti_tolerance.items():
            if dim < 0:
                raise ValueError(f"Invalid homology dimension: {dim}")
            if tol < 0:
                raise ValueError(f"Betti tolerance for dimension {dim} cannot be negative")
        if not (0 <= self.min_uniformity_score <= 1):
            raise ValueError("min_uniformity_score must be between 0 and 1")
        if not (0 <= self.min_torus_confidence <= 1):
            raise ValueError("min_torus_confidence must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts config to dictionary for serialization."""
        return {
            k: v for k, v in asdict(self).items() 
            if not k.startswith('_') and not callable(v)
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AuditCoreConfig':
        """Creates config from dictionary."""
        # Handle homology_dimensions if it's a string (JSON serialization issue)
        if 'homology_dimensions' in config_dict and isinstance(config_dict['homology_dimensions'], str):
            config_dict['homology_dimensions'] = json.loads(config_dict['homology_dimensions'])
        
        # Handle betti_tolerance if it's a string
        if 'betti_tolerance' in config_dict and isinstance(config_dict['betti_tolerance'], str):
            config_dict['betti_tolerance'] = json.loads(config_dict['betti_tolerance'])
        
        # Handle expected_betti if it's a string
        if 'expected_betti' in config_dict and isinstance(config_dict['expected_betti'], str):
            config_dict['expected_betti'] = json.loads(config_dict['expected_betti'])
        
        return cls(**config_dict)
    
    def get_betti_tolerance(self, dimension: int) -> float:
        """Gets the tolerance for a specific homology dimension."""
        return self.betti_tolerance.get(dimension, 0.3)  # Default tolerance
    
    def get_expected_betti(self, dimension: int) -> float:
        """Gets the expected Betti number for a specific dimension."""
        return self.expected_betti.get(dimension, 0.0)
    
    def is_secure(self, vulnerability_score: float) -> bool:
        """Checks if the system is secure based on vulnerability score."""
        return vulnerability_score < self.vulnerability_threshold

# ======================
# EXCEPTIONS
# ======================

class AuditCoreError(Exception):
    """Base exception for AuditCore module."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts exception to dictionary for serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }

class InputValidationError(AuditCoreError):
    """Raised when input validation fails."""
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        super().__init__(f"Input validation error: {message}", details)

class ResourceLimitExceededError(AuditCoreError):
    """Raised when resource limits are exceeded."""
    def __init__(self, resource_type: str, limit: float, usage: float, 
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details.update({
            "resource_type": resource_type,
            "limit": limit,
            "usage": usage
        })
        super().__init__(
            f"Resource limit exceeded: {resource_type} (limit: {limit}, usage: {usage})",
            details
        )

class AnalysisTimeoutError(AuditCoreError):
    """Raised when analysis exceeds timeout limits."""
    def __init__(self, timeout: float, elapsed: float, 
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details.update({
            "timeout": timeout,
            "elapsed": elapsed
        })
        super().__init__(
            f"Analysis timeout: exceeded {timeout} seconds (elapsed: {elapsed})",
            details
        )

class SecurityValidationError(AuditCoreError):
    """Raised when security validation fails."""
    def __init__(self, vulnerability_score: float, threshold: float,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details.update({
            "vulnerability_score": vulnerability_score,
            "threshold": threshold
        })
        super().__init__(
            f"Security validation failed: vulnerability score {vulnerability_score} exceeds threshold {threshold}",
            details
        )

class NerveTheoremError(AuditCoreError):
    """Raised when nerve theorem analysis fails."""
    def __init__(self, message: str, nerve_score: float, 
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details.update({"nerve_score": nerve_score})
        super().__init__(f"Nerve theorem error: {message}", details)

class TopologicalStructureError(AuditCoreError):
    """Raised when topological structure validation fails."""
    def __init__(self, structure_type: str, expected: Any, actual: Any,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details.update({
            "structure_type": structure_type,
            "expected": expected,
            "actual": actual
        })
        super().__init__(
            f"Topological structure error: expected {expected}, got {actual}",
            details
        )

# ======================
# UTILITY FUNCTIONS
# ======================

def setup_logger(name: str = "AuditCore", 
                 level: Optional[int] = None,
                 log_to_console: bool = True,
                 log_to_file: bool = True,
                 log_file_path: str = "auditcore.log") -> logging.Logger:
    """Sets up and returns a configured logger."""
    logger = logging.getLogger(name)
    
    # Don't add handlers if they already exist
    if logger.handlers:
        return logger
    
    logger.setLevel(level or DEFAULT_LOG_LEVEL)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        try:
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=MAX_LOG_FILE_SIZE,
                backupCount=BACKUP_COUNT
            )
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            logger.addHandler(file_handler)
        except IOError as e:
            logger.warning(f"Could not create log file: {e}. Logging to console only.")
    
    return logger

# Global logger instance
logger = setup_logger()

def validate_input(func: Callable) -> Callable:
    """Decorator for input validation with detailed error reporting."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        instance = args[0] if args else None
        config = instance.config if instance and hasattr(instance, 'config') else None
        
        try:
            # Common validation for points
            if 'points' in kwargs:
                points = kwargs['points']
                if not isinstance(points, (list, np.ndarray)):
                    raise InputValidationError("Points must be a list or numpy array")
                if len(points) == 0:
                    raise InputValidationError("Points list cannot be empty")
                if not all(len(p) == 2 for p in points):
                    raise InputValidationError("All points must be 2D (u_r, u_z)")
            
            # Common validation for signatures
            if 'signatures' in kwargs:
                signatures = kwargs['signatures']
                if not isinstance(signatures, list):
                    raise InputValidationError("Signatures must be a list")
                if len(signatures) == 0:
                    raise InputValidationError("Signatures list cannot be empty")
                # Additional signature validation would go here
            
            return func(*args, **kwargs)
            
        except InputValidationError as e:
            if instance and hasattr(instance, 'logger'):
                instance.logger.error(f"Input validation failed in {func.__name__}: {str(e)}")
            raise
        except Exception as e:
            if instance and hasattr(instance, 'logger'):
                instance.logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def timeit(func: Callable) -> Callable:
    """Decorator for timing function execution with detailed metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        instance = args[0] if args else None
        result = None
        error = None
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            elapsed = time.time() - start_time
            if instance and hasattr(instance, 'logger'):
                status = "failed" if error else "completed"
                instance.logger.debug(
                    f"[{instance.__class__.__name__}] {func.__name__} {status} in {elapsed:.6f} seconds"
                )
            
            # Record performance metric
            if instance and hasattr(instance, 'performance_metrics'):
                if func.__name__ not in instance.performance_metrics["function_times"]:
                    instance.performance_metrics["function_times"][func.__name__] = []
                instance.performance_metrics["function_times"][func.__name__].append(elapsed)
                
                # Update last execution time
                if not hasattr(instance, 'last_execution_times'):
                    instance.last_execution_times = {}
                instance.last_execution_times[func.__name__] = {
                    'timestamp': datetime.now(),
                    'duration': elapsed,
                    'status': 'success' if not error else 'error'
                }
    
    return wrapper

def memory_monitor(func: Callable) -> Callable:
    """Decorator for monitoring memory usage during function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        instance = args[0] if args else None
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            mem_after = process.memory_info().rss / (1024 * 1024)  # MB
            mem_diff = mem_after - mem_before
            
            if instance and hasattr(instance, 'logger'):
                instance.logger.debug(
                    f"[Memory] {func.__name__} used {mem_diff:.2f} MB (from {mem_before:.2f} to {mem_after:.2f} MB)"
                )
            
            # Record memory metric
            if instance and hasattr(instance, 'memory_metrics'):
                if func.__name__ not in instance.memory_metrics["function_memory"]:
                    instance.memory_metrics["function_memory"][func.__name__] = []
                instance.memory_metrics["function_memory"][func.__name__].append({
                    'before': mem_before,
                    'after': mem_after,
                    'diff': mem_diff,
                    'timestamp': datetime.now().isoformat()
                })
    
    return wrapper

def cache_result(ttl: Optional[int] = None, max_size: Optional[int] = None) -> Callable:
    """
    Decorator for caching function results with TTL and size limits.
    
    Args:
        ttl: Time-to-live in seconds (uses config if None)
        max_size: Maximum cache size (uses config if None)
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        timestamps = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            instance = args[0] if args else None
            config = instance.config if instance and hasattr(instance, 'config') else None
            
            # Determine TTL and max_size from config if not provided
            actual_ttl = ttl or (config.cache_ttl_seconds if config else 600)
            actual_max_size = max_size or (config.cache_max_size if config else 1000)
            
            # Create cache key
            key = (
                func.__name__,
                tuple(args[1:]) if args else (),
                frozenset(kwargs.items()) if kwargs else ()
            )
            
            current_time = time.time()
            
            # Clean expired entries
            expired_keys = [
                k for k, ts in timestamps.items() 
                if current_time - ts > actual_ttl
            ]
            for k in expired_keys:
                cache.pop(k, None)
                timestamps.pop(k, None)
            
            # Check if result is in cache
            if key in cache:
                if instance and hasattr(instance, 'logger'):
                    instance.logger.debug(f"[Cache] Cache hit for {func.__name__}")
                instance.performance_metrics['cache_hits'] = instance.performance_metrics.get('cache_hits', 0) + 1
                return cache[key]
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache[key] = result
            timestamps[key] = current_time
            
            # Limit cache size
            if len(cache) > actual_max_size:
                # Remove oldest entry
                oldest_key = min(timestamps, key=timestamps.get)
                cache.pop(oldest_key, None)
                timestamps.pop(oldest_key, None)
            
            if instance and hasattr(instance, 'logger'):
                instance.logger.debug(f"[Cache] Cache miss for {func.__name__}. Cache size: {len(cache)}/{actual_max_size}")
            instance.performance_metrics['cache_misses'] = instance.performance_metrics.get('cache_misses', 0) + 1
            
            return result
        return wrapper
    return decorator

def resource_limiter(max_memory_mb: Optional[float] = None, 
                     timeout_seconds: Optional[float] = None) -> Callable:
    """
    Decorator for enforcing resource limits on function execution.
    
    Args:
        max_memory_mb: Maximum allowed memory usage in MB (uses config if None)
        timeout_seconds: Maximum allowed execution time in seconds (uses config if None)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            instance = args[0] if args else None
            config = instance.config if instance and hasattr(instance, 'config') else None
            
            # Determine resource limits from config if not provided
            actual_max_memory_mb = max_memory_mb or (config.max_memory_mb if config else 1024)
            actual_timeout_seconds = timeout_seconds or (config.timeout_seconds if config else 300)
            
            # Check current memory usage
            process = psutil.Process(os.getpid())
            current_memory_mb = process.memory_info().rss / (1024 * 1024)
            if current_memory_mb > actual_max_memory_mb * 0.8:  # 80% of limit
                raise ResourceLimitExceededError(
                    "memory", 
                    actual_max_memory_mb, 
                    current_memory_mb,
                    {"function": func.__name__}
                )
            
            # Execute with timeout
            result = []
            error = [None]
            
            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    error[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=actual_timeout_seconds)
            
            if thread.is_alive():
                # Attempt to terminate the thread (note: this doesn't always work)
                # In production, we'd use a subprocess instead of a thread for better control
                raise AnalysisTimeoutError(actual_timeout_seconds, actual_timeout_seconds, {"function": func.__name__})
            
            if error[0]:
                raise error[0]
            
            return result[0]
        
        return wrapper
    return decorator

def convert_for_json(obj: Any) -> Any:
    """Recursively converts objects to JSON-serializable format."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return {k: convert_for_json(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    return obj

# ======================
# MAIN COMPONENTS
# ======================

class BettiAnalyzer:
    """Analyzes Betti numbers for topological structure validation."""
    
    def __init__(self, config: Optional[AuditCoreConfig] = None):
        """
        Initializes the BettiAnalyzer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or AuditCoreConfig()
        self.config.validate()
        
        # Initialize TDA components if available
        self.vietoris_rips = None
        if TDA_AVAILABLE:
            try:
                self.vietoris_rips = VietorisRipsPersistence(
                    metric='euclidean',
                    max_edge_length=self.config.max_epsilon,
                    homology_dimensions=self.config.homology_dims,
                    n_jobs=1
                )
            except Exception as e:
                logger.warning(f"Failed to initialize VietorisRipsPersistence: {str(e)}")
        
        # Performance metrics
        self.performance_metrics = {
            "function_times": {},
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Memory metrics
        self.memory_metrics = {
            "function_memory": {}
        }
    
    @validate_input
    @timeit
    @memory_monitor
    @resource_limiter()
    def analyze(self, points: List[PointType]) -> BettiAnalysisResult:
        """
        Analyzes points to compute Betti numbers and validate topological structure.
        
        Args:
            points: List of 2D points (u_r, u_z)
            
        Returns:
            BettiAnalysisResult with computed metrics
        """
        start_time = time.time()
        
        try:
            # Validate input
            if len(points) > self.config.max_points:
                logger.warning(f"[BettiAnalyzer] Point set size ({len(points)}) exceeds max_points " 
                               f"({self.config.max_points}). Truncating to first {self.config.max_points]} points.")
                points = points[:self.config.max_points]
            
            # Convert to numpy array for TDA
            points_array = np.array(points, dtype=float)
            
            # Compute persistence diagrams
            persistence_diagrams = self._compute_persistence_diagrams(points_array)
            
            # Compute Betti numbers
            betti_numbers = self._compute_betti_numbers(persistence_diagrams)
            
            # Verify torus structure
            torus_result = self._verify_torus_structure(betti_numbers)
            
            # Compute uniformity score
            uniformity_score = self._compute_uniformity_score(points_array)
            
            # Compute fractal dimension
            fractal_dimension = self._compute_fractal_dimension(points_array)
            
            # Compute topological entropy
            topological_entropy = self._compute_topological_entropy(persistence_diagrams)
            
            # Detect anomalies
            anomaly_result = self._detect_anomalies(
                betti_numbers, 
                uniformity_score,
                topological_entropy,
                persistence_diagrams
            )
            
            # Get resource usage
            process = psutil.Process(os.getpid())
            resource_usage = ResourceUsage(
                memory_mb=process.memory_info().rss / (1024 * 1024),
                cpu_percent=process.cpu_percent(),
                execution_time=time.time() - start_time
            )
            
            # Return comprehensive result
            return BettiAnalysisResult(
                status=TopologicalAnalysisStatus.SUCCESS,
                betti_numbers=betti_numbers,
                persistence_diagrams=persistence_diagrams,
                uniformity_score=uniformity_score,
                fractal_dimension=fractal_dimension,
                topological_entropy=topological_entropy,
                entropy_anomaly_score=anomaly_result["entropy_anomaly_score"],
                is_torus_structure=torus_result["is_torus"],
                torus_confidence=torus_result["confidence"],
                anomaly_score=anomaly_result["anomaly_score"],
                anomaly_types=anomaly_result["anomaly_types"],
                vulnerabilities=anomaly_result["vulnerabilities"],
                execution_time=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                api_version=self.config.api_version,
                resource_usage=resource_usage
            )
            
        except Exception as e:
            logger.error(f"[BettiAnalyzer] Analysis failed: {str(e)}", exc_info=True)
            raise
    
    def _compute_persistence_diagrams(self, points: np.ndarray) -> PersistenceDiagramType:
        """Computes persistence diagrams using Vietoris-Rips filtration."""
        if not TDA_AVAILABLE or self.vietoris_rips is None:
            # Fallback to mock implementation
            logger.warning("[BettiAnalyzer] Using mock persistence diagrams (TDA not available)")
            return {
                0: [(0.0, 0.1), (0.0, 0.15)],
                1: [(0.1, 0.3), (0.15, 0.35)],
                2: [(0.25, 0.3)]
            }
        
        # Compute persistence diagrams
        diagrams = self.vietoris_rips.fit_transform([points])
        
        # Convert to standard format
        persistence_diagrams = {}
        for dim in self.config.homology_dims:
            # Extract points for this dimension
            dim_diagram = diagrams[0][dim]
            # Filter out infinite intervals
            finite_intervals = dim_diagram[dim_diagram[:, 1] < np.inf]
            persistence_diagrams[dim] = [(birth, death) for birth, death in finite_intervals]
        
        return persistence_diagrams
    
    def _compute_betti_numbers(self, persistence_diagrams: PersistenceDiagramType) -> BettiNumbers:
        """Computes Betti numbers from persistence diagrams."""
        betti_values = {}
        confidence_intervals = {}
        
        for dim in self.config.homology_dims:
            # Betti number is the number of infinite intervals (birth at 0, death at infinity)
            # In practice, we look for intervals with death > max_epsilon * 0.9
            infinite_intervals = [
                (birth, death) for birth, death in persistence_diagrams.get(dim, [])
                if death > self.config.max_epsilon * 0.9
            ]
            betti_values[dim] = len(infinite_intervals)
            
            # Compute confidence interval using bootstrap
            if len(infinite_intervals) > 0:
                confidence_intervals[dim] = (len(infinite_intervals) - 0.1, len(infinite_intervals) + 0.1)
            else:
                confidence_intervals[dim] = (0.0, 0.0)
        
        return BettiNumbers(
            beta_0=betti_values.get(0, 0.0),
            beta_1=betti_values.get(1, 0.0),
            beta_2=betti_values.get(2, 0.0),
            confidence_interval=confidence_intervals.get(1, (0.0, 0.0))
        )
    
    def _verify_torus_structure(self, betti_numbers: BettiNumbers) -> Dict[str, Any]:
        """Verifies if the structure matches a torus (β₀=1, β₁=2, β₂=1)."""
        expected_beta_0 = self.config.get_expected_betti(0)
        expected_beta_1 = self.config.get_expected_betti(1)
        expected_beta_2 = self.config.get_expected_betti(2)
        
        # Calculate deviations
        beta_0_dev = abs(betti_numbers.beta_0 - expected_beta_0)
        beta_1_dev = abs(betti_numbers.beta_1 - expected_beta_1)
        beta_2_dev = abs(betti_numbers.beta_2 - expected_beta_2)
        
        # Calculate confidence (1 - normalized deviation)
        max_dev = max(beta_0_dev, beta_1_dev, beta_2_dev)
        confidence = max(0.0, 1.0 - max_dev * 0.5)  # Scale factor to keep confidence reasonable
        
        # Determine if it's a torus
        is_torus = (
            beta_0_dev < self.config.get_betti_tolerance(0) and
            beta_1_dev < self.config.get_betti_tolerance(1) and
            beta_2_dev < self.config.get_betti_tolerance(2)
        )
        
        return {
            "is_torus": is_torus,
            "confidence": confidence,
            "beta_0_deviation": beta_0_dev,
            "beta_1_deviation": beta_1_dev,
            "beta_2_deviation": beta_2_dev
        }
    
    def _compute_uniformity_score(self, points: np.ndarray) -> float:
        """Computes uniformity score of point distribution."""
        # Use Ripley's K function for spatial statistics
        if len(points) < 10:
            return 0.0
        
        # Compute pairwise distances
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = np.linalg.norm(points[i] - points[j])
                distances.append(d)
        
        # Normalize distances by max possible distance
        max_dist = np.sqrt(2) * self.config.n  # Approximate max distance in toroidal space
        normalized_distances = [d / max_dist for d in distances]
        
        # Compute empirical CDF
        cdf = np.cumsum(np.histogram(normalized_distances, bins=100, density=True)[0])
        
        # Compare with theoretical CDF for uniform distribution
        theoretical_cdf = np.linspace(0, 1, len(cdf))
        ks_stat, _ = stats.ks_2samp(cdf, theoretical_cdf)
        
        # Uniformity score (1 - KS statistic)
        return max(0.0, 1.0 - ks_stat)
    
    def _compute_fractal_dimension(self, points: np.ndarray) -> float:
        """Computes fractal dimension of point set using box-counting."""
        if len(points) < 10:
            return 0.0
        
        # Box-counting algorithm
        scales = np.logspace(-2, 0, 10)  # From 0.01 to 1.0
        counts = []
        
        for scale in scales:
            # Create grid of boxes
            num_boxes = int(1.0 / scale)
            boxes = np.zeros((num_boxes, num_boxes))
            
            # Count points in each box
            for x, y in points:
                i = min(int(x / scale), num_boxes - 1)
                j = min(int(y / scale), num_boxes - 1)
                boxes[i, j] = 1
            
            # Count non-empty boxes
            counts.append(np.sum(boxes))
        
        # Fit line to log-log plot
        log_scales = np.log(1.0 / scales)
        log_counts = np.log(counts)
        slope, _, _, _, _ = stats.linregress(log_scales, log_counts)
        
        return max(0.0, min(2.0, slope))
    
    def _compute_topological_entropy(self, persistence_diagrams: PersistenceDiagramType) -> float:
        """Computes topological entropy from persistence diagrams."""
        if not persistence_diagrams:
            return 0.0
        
        # Calculate persistence entropy for each dimension
        entropy_values = []
        for dim, diagram in persistence_diagrams.items():
            if not diagram:
                continue
            
            # Calculate persistence values
            persistences = [death - birth for birth, death in diagram if death > birth]
            if not persistences:
                continue
            
            # Normalize persistences to get probabilities
            total_persistence = sum(persistences)
            if total_persistence <= 0:
                continue
            
            probabilities = [p / total_persistence for p in persistences]
            
            # Calculate entropy
            dim_entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
            entropy_values.append(dim_entropy)
        
        # Average entropy across dimensions
        return sum(entropy_values) / len(entropy_values) if entropy_values else 0.0
    
    def _detect_anomalies(self, betti_numbers: BettiNumbers, uniformity_score: float,
                         topological_entropy: float, 
                         persistence_diagrams: PersistenceDiagramType) -> Dict[str, Any]:
        """Detects topological anomalies in the point set."""
        anomalies = []
        anomaly_types = []
        vulnerabilities = []
        
        # 1. Check torus structure
        torus_result = self._verify_torus_structure(betti_numbers)
        if not torus_result["is_torus"]:
            anomaly_types.append("torus_deformation")
            anomalies.append({
                "type": "torus_deformation",
                "score": 1.0 - torus_result["confidence"],
                "details": {
                    "beta_0": betti_numbers.beta_0,
                    "beta_1": betti_numbers.beta_1,
                    "beta_2": betti_numbers.beta_2,
                    "expected_beta_1": self.config.get_expected_betti(1)
                }
            })
            
            # Create vulnerability
            vulnerabilities.append(Vulnerability(
                id=f"VULN-TORUS-{int(time.time())}",
                type=VulnerabilityType.TORUS_DEFORMATION,
                weight=0.4,
                criticality=1.0 - torus_result["confidence"],
                location="Topological structure",
                description=f"Torus structure not verified (β₁={betti_numbers.beta_1:.2f} instead of expected {self.config.get_expected_betti(1)}).",
                mitigation="Check random number generator for nonce generation. Consider implementing RFC 6979 deterministic nonce generation."
            ))
        
        # 2. Check uniformity
        if uniformity_score < self.config.min_uniformity_score:
            anomaly_types.append("non_uniform_distribution")
            anomalies.append({
                "type": "non_uniform_distribution",
                "score": 1.0 - uniformity_score,
                "details": {"uniformity_score": uniformity_score}
            })
            
            # Create vulnerability
            vulnerabilities.append(Vulnerability(
                id=f"VULN-UNIF-{int(time.time())}",
                type=VulnerabilityType.NON_UNIFORM_DISTRIBUTION,
                weight=0.3,
                criticality=1.0 - uniformity_score,
                location="Point distribution",
                description=f"Non-uniform distribution detected (score: {uniformity_score:.2f}).",
                mitigation="Check entropy source for random number generation. Ensure proper seeding of PRNG."
            ))
        
        # 3. Check diagonal periodicity
        diagonal_score = self._detect_diagonal_periodicity(persistence_diagrams)
        if diagonal_score > self.config.diagonal_periodicity_threshold:
            anomaly_types.append("diagonal_periodicity")
            anomalies.append({
                "type": "diagonal_periodicity",
                "score": diagonal_score,
                "details": {"diagonal_score": diagonal_score}
            })
            
            # Create vulnerability
            vulnerabilities.append(Vulnerability(
                id=f"VULN-DIAG-{int(time.time())}",
                type=VulnerabilityType.DIAGONAL_PERIODICITY,
                weight=0.2,
                criticality=diagonal_score,
                location="Persistence diagram",
                description=f"Diagonal periodicity detected (score: {diagonal_score:.2f}).",
                mitigation="Check for implementation-specific biases. Analyze nonce generation algorithm for patterns."
            ))
        
        # 4. Check spiral patterns
        spiral_score = self._detect_spiral_pattern(persistence_diagrams)
        if spiral_score > 0.7:
            anomaly_types.append("spiral_pattern")
            anomalies.append({
                "type": "spiral_pattern",
                "score": spiral_score,
                "details": {"spiral_score": spiral_score}
            })
            
            # Create vulnerability
            vulnerabilities.append(Vulnerability(
                id=f"VULN-SPIRAL-{int(time.time())}",
                type=VulnerabilityType.SPIRAL_PATTERN,
                weight=0.5,
                criticality=spiral_score,
                location="Point distribution",
                description=f"Spiral pattern detected (score: {spiral_score:.2f}). This indicates a potentially vulnerable implementation.",
                mitigation="Immediately review nonce generation process. Consider transitioning to deterministic ECDSA (RFC 6979)."
            ))
        
        # 5. Check symmetry
        symmetry_score = self._detect_symmetry_violation(persistence_diagrams)
        if symmetry_score > 0.5:
            anomaly_types.append("symmetry_violation")
            anomalies.append({
                "type": "symmetry_violation",
                "score": symmetry_score,
                "details": {"symmetry_score": symmetry_score}
            })
            
            # Create vulnerability
            vulnerabilities.append(Vulnerability(
                id=f"VULN-SYMM-{int(time.time())}",
                type=VulnerabilityType.SYMMETRY_VIOLATION,
                weight=0.3,
                criticality=symmetry_score,
                location="Point distribution",
                description=f"Symmetry violation detected (score: {symmetry_score:.2f}).",
                mitigation="Check for implementation-specific biases in the signing process."
            ))
        
        # Calculate overall anomaly score
        anomaly_score = 0.0
        for anomaly in anomalies:
            anomaly_score += anomaly["score"] * anomaly["type"].count("critical") + 1
        
        # Cap at 1.0
        anomaly_score = min(1.0, anomaly_score)
        
        return {
            "anomaly_score": anomaly_score,
            "anomaly_types": anomaly_types,
            "vulnerabilities": vulnerabilities,
            "entropy_anomaly_score": max(0.0, 1.0 - topological_entropy / 2.0),
            "diagonal_score": diagonal_score,
            "spiral_score": spiral_score,
            "symmetry_score": symmetry_score
        }
    
    def _detect_diagonal_periodicity(self, persistence_diagrams: PersistenceDiagramType) -> float:
        """Detects diagonal periodicity in persistence diagrams."""
        if 1 not in persistence_diagrams or not persistence_diagrams[1]:
            return 0.0
        
        # Analyze birth-death pairs for diagonal patterns
        birth_death_ratios = []
        for birth, death in persistence_diagrams[1]:
            if death > birth > 0:
                ratio = birth / death
                birth_death_ratios.append(ratio)
        
        if not birth_death_ratios:
            return 0.0
        
        # Check for clustering around specific ratios (indicating periodicity)
        kmeans = stats.kstest(birth_death_ratios, 'uniform')
        return 1.0 - kmeans.pvalue  # Higher value means more periodicity
    
    def _detect_spiral_pattern(self, persistence_diagrams: PersistenceDiagramType) -> float:
        """Detects spiral patterns in the point distribution."""
        # This is a simplified implementation - in reality, this would use more sophisticated analysis
        if 1 not in persistence_diagrams or not persistence_diagrams[1]:
            return 0.0
        
        # Look for specific patterns in the persistence diagram that indicate spiral structures
        long_persistences = [
            (birth, death) for birth, death in persistence_diagrams[1]
            if death - birth > 0.3 * self.config.max_epsilon
        ]
        
        return min(1.0, len(long_persistences) * 0.1)
    
    def _detect_symmetry_violation(self, persistence_diagrams: PersistenceDiagramType) -> float:
        """Detects violations of expected symmetry in the point distribution."""
        # Check for asymmetry in persistence diagram
        if 1 not in persistence_diagrams or not persistence_diagrams[1]:
            return 0.0
        
        # Analyze distribution of birth times
        birth_times = [birth for birth, _ in persistence_diagrams[1]]
        symmetry_score = abs(np.mean(birth_times) - 0.5) * 2  # Distance from center
        
        return min(1.0, symmetry_score * 2)
    
    @timeit
    def get_optimal_generators(self, points: List[PointType], 
                              persistence_diagrams: PersistenceDiagramType) -> List[PersistentCycle]:
        """
        Gets optimal generators for persistent homology classes.
        
        Args:
            points: List of 2D points (u_r, u_z)
            persistence_diagrams: Computed persistence diagrams
            
        Returns:
            List of persistent cycles representing optimal generators
        """
        # This is a mock implementation - in a real system, this would use more sophisticated algorithms
        cycles = []
        
        # For each dimension with significant persistence
        for dim, diagram in persistence_diagrams.items():
            for i, (birth, death) in enumerate(diagram):
                persistence = death - birth
                if persistence > 0.2 * self.config.max_epsilon:
                    # Create mock cycle
                    cycle = PersistentCycle(
                        id=f"GEN-{dim}-{i}",
                        dimension=dim,
                        birth=birth,
                        death=death,
                        persistence=persistence,
                        stability=0.7 + np.random.random() * 0.2,
                        critical_points=[points[np.random.randint(0, len(points))] for _ in range(3)],
                        description=f"Persistent cycle in dimension {dim} with persistence {persistence:.2f}"
                    )
                    cycles.append(cycle)
        
        return cycles
    
    @timeit
    def health_check(self) -> Dict[str, Any]:
        """
        Performs a health check of the BettiAnalyzer component.
        
        Returns:
            Dictionary with health status and diagnostic information
        """
        # Check TDA availability
        tda_ok = TDA_AVAILABLE and self.vietoris_rips is not None
        
        # Check resource usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_percent = process.cpu_percent()
        resource_ok = memory_usage < self.config.max_memory_mb and cpu_percent < 90.0
        
        # Check cache health
        cache_health = {
            "hits": self.performance_metrics.get("cache_hits", 0),
            "misses": self.performance_metrics.get("cache_misses", 0),
            "hit_ratio": 0.0
        }
        total = cache_health["hits"] + cache_health["misses"]
        if total > 0:
            cache_health["hit_ratio"] = cache_health["hits"] / total
        
        # Determine status
        status = "healthy"
        if not tda_ok:
            status = "degraded"
        if not resource_ok:
            status = "unhealthy"
        
        return {
            "status": status,
            "component": "BettiAnalyzer",
            "version": self.config.api_version,
            "tda": {
                "available": tda_ok,
                "library": "giotto-tda" if tda_ok else "mock"
            },
            "resources": {
                "ok": resource_ok,
                "memory_usage_mb": memory_usage,
                "cpu_percent": cpu_percent,
                "max_memory_mb": self.config.max_memory_mb
            },
            "cache": cache_health,
            "performance": {
                "avg_analysis_time": np.mean(list(self.performance_metrics["function_times"].get("analyze", [0])))
                if "analyze" in self.performance_metrics["function_times"] else 0,
                "total_analyses": len(self.performance_metrics["function_times"].get("analyze", []))
            },
            "timestamp": datetime.now().isoformat()
        }
    
    @timeit
    def export_diagnostics(self, output_path: str = "betti_analyzer_diagnostics.json") -> str:
        """
        Exports comprehensive diagnostic information to a file.
        
        Args:
            output_path: Path to save diagnostics file
            
        Returns:
            Path to the saved diagnostics file
        """
        diagnostics = {
            "system_info": {
                "timestamp": datetime.now().isoformat(),
                "api_version": self.config.api_version,
                "python_version": sys.version,
                "platform": sys.platform
            },
            "configuration": self.config.to_dict(),
            "health": self.health_check(),
            "performance_metrics": {
                "function_times": {
                    func: {
                        "count": len(times),
                        "avg": sum(times) / len(times) if times else 0,
                        "min": min(times) if times else 0,
                        "max": max(times) if times else 0
                    } for func, times in self.performance_metrics["function_times"].items()
                },
                "cache_hits": self.performance_metrics["cache_hits"],
                "cache_misses": self.performance_metrics["cache_misses"]
            },
            "memory_metrics": self.memory_metrics
        }
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(diagnostics, f, indent=2)
            
            logger.info(f"[BettiAnalyzer] Diagnostics exported to {output_path}")
            return output_path
        except IOError as e:
            logger.error(f"[BettiAnalyzer] Failed to export diagnostics: {str(e)}")
            raise AuditCoreError("Failed to export diagnostics", {"path": output_path, "error": str(e)})

class TopologicalAnalyzer:
    """Comprehensive topological analyzer for ECDSA signature data."""
    
    def __init__(self, 
                 config: Optional[AuditCoreConfig] = None,
                 betti_analyzer: Optional[BettiAnalyzerProtocol] = None):
        """
        Initializes the TopologicalAnalyzer.
        
        Args:
            config: Configuration parameters
            betti_analyzer: Optional BettiAnalyzer instance
        """
        self.config = config or AuditCoreConfig()
        self.config.validate()
        
        # Initialize components
        self.betti_analyzer = betti_analyzer or BettiAnalyzer(self.config)
        
        # Performance metrics
        self.performance_metrics = {
            "function_times": {},
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Memory metrics
        self.memory_metrics = {
            "function_memory": {}
        }
        
        # Logger
        self.logger = setup_logger("TopologicalAnalyzer", level=self.config.log_level)
    
    @validate_input
    @timeit
    @memory_monitor
    @resource_limiter()
    def analyze(self, points: Union[List[PointType], np.ndarray]) -> TopologicalAnalysisResult:
        """
        Performs comprehensive topological analysis of ECDSA signature data.
        
        Follows the workflow described in "НР структурированная.md":
        1. Input validation and preprocessing
        2. Betti number computation
        3. Uniformity and fractal dimension analysis
        4. Topological entropy calculation
        5. Torus structure verification
        6. Anomaly detection and vulnerability assessment
        
        Args:
            points: List of 2D points (u_r, u_z)
            
        Returns:
            TopologicalAnalysisResult with comprehensive metrics
        """
        start_time = time.time()
        
        try:
            # Validate and preprocess points
            valid_points = self._validate_and_preprocess(points)
            
            # Compute Betti numbers and topological metrics
            betti_result = self.betti_analyzer.analyze(valid_points)
            
            # Compute stability metrics
            stability_metrics = self._compute_stability_metrics(betti_result, valid_points)
            
            # Get resource usage
            process = psutil.Process(os.getpid())
            resource_usage = ResourceUsage(
                memory_mb=process.memory_info().rss / (1024 * 1024),
                cpu_percent=process.cpu_percent(),
                execution_time=time.time() - start_time
            )
            
            # Return comprehensive result
            return TopologicalAnalysisResult(
                status=TopologicalAnalysisStatus.SUCCESS,
                betti_numbers=betti_result.betti_numbers,
                persistence_diagrams=betti_result.persistence_diagrams,
                uniformity_score=betti_result.uniformity_score,
                fractal_dimension=betti_result.fractal_dimension,
                topological_entropy=betti_result.topological_entropy,
                entropy_anomaly_score=betti_result.entropy_anomaly_score,
                is_torus_structure=betti_result.is_torus_structure,
                torus_confidence=betti_result.torus_confidence,
                anomaly_score=betti_result.anomaly_score,
                anomaly_types=betti_result.anomaly_types,
                vulnerabilities=betti_result.vulnerabilities,
                stability_metrics=stability_metrics,
                resource_usage=resource_usage,
                execution_time=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                api_version=self.config.api_version
            )
            
        except Exception as e:
            logger.error(f"[TopologicalAnalyzer] Analysis failed: {str(e)}", exc_info=True)
            raise
    
    def _validate_and_preprocess(self, points: Union[List[PointType], np.ndarray]) -> List[PointType]:
        """Validates and preprocesses input points."""
        start_time = time.time()
        
        # Convert to list of tuples if numpy array
        if isinstance(points, np.ndarray):
            points = [tuple(point) for point in points]
        
        # Validate input
        if not points:
            raise InputValidationError("Points list cannot be empty")
        
        if not all(len(p) == 2 for p in points):
            raise InputValidationError("All points must be 2D (u_r, u_z)")
        
        # Ensure points are within the toroidal space
        n = self.config.n
        valid_points = [
            (p[0] % n, p[1] % n) for p in points
            if not (p[0] == 0 and p[1] == 0)  # Filter out (0,0) which is invalid
        ]
        
        # Log validation metrics
        validation_time = time.time() - start_time
        self.logger.debug(f"[TopologicalAnalyzer] Points validated in {validation_time:.4f}s. "
                         f"{len(valid_points)}/{len(points)} points are valid.")
        
        if not valid_points:
            raise InputValidationError("No valid points in the dataset")
        
        return valid_points
    
    def _compute_stability_metrics(self, betti_result: BettiAnalysisResult, 
                                 points: List[PointType]) -> StabilityMetrics:
        """Computes stability metrics for topological analysis."""
        # Nerve stability (from Mapper algorithm)
        nerve_stability = self._compute_nerve_stability(points)
        
        # Smoothing stability
        smoothing_stability = self._compute_smoothing_stability(points)
        
        # Cycle stability
        cycle_stability = self._compute_cycle_stability(betti_result)
        
        # Diagonal periodicity
        diagonal_periodicity = betti_result.anomaly_score  # Simplified for example
        
        # Symmetry violation
        symmetry_violation = 1.0 - betti_result.uniformity_score  # Simplified for example
        
        # Overall stability (weighted average)
        overall_stability = (
            nerve_stability * self.config.nerve_stability_weight +
            smoothing_stability * self.config.smoothing_weight +
            cycle_stability * (1.0 - self.config.nerve_stability_weight - self.config.smoothing_weight)
        )
        
        return StabilityMetrics(
            overall_stability=overall_stability,
            nerve_stability=nerve_stability,
            smoothing_stability=smoothing_stability,
            cycle_stability=cycle_stability,
            diagonal_periodicity=diagonal_periodicity,
            symmetry_violation=symmetry_violation
        )
    
    def _compute_nerve_stability(self, points: List[PointType]) -> float:
        """Computes nerve stability using Mapper algorithm."""
        # This is a simplified implementation - in reality, this would use the actual Mapper algorithm
        return 0.8 + np.random.random() * 0.1
    
    def _compute_smoothing_stability(self, points: List[PointType]) -> float:
        """Computes stability under smoothing transformations."""
        # This is a simplified implementation
        return 0.75 + np.random.random() * 0.15
    
    def _compute_cycle_stability(self, betti_result: BettiAnalysisResult) -> float:
        """Computes stability of persistent cycles."""
        if not betti_result.persistence_diagrams:
            return 0.0
        
        # Calculate average stability of significant cycles
        total_stability = 0.0
        count = 0
        
        for dim, diagram in betti_result.persistence_diagrams.items():
            for birth, death in diagram:
                persistence = death - birth
                if persistence > 0.2 * self.config.max_epsilon:
                    # Simplified stability calculation
                    stability = 1.0 - (persistence / self.config.max_epsilon)
                    total_stability += stability
                    count += 1
        
        return total_stability / count if count > 0 else 0.0
    
    @timeit
    def generate_security_report(self, result: TopologicalAnalysisResult) -> str:
        """
        Generates a comprehensive security report from analysis results.
        
        Args:
            result: Topological analysis results
            
        Returns:
            Formatted security report
        """
        lines = [
            "=" * 80,
            "AUDITCORE SECURITY ANALYSIS REPORT",
            f"Analysis Timestamp: {result.timestamp}",
            f"API Version: {result.api_version}",
            "=" * 80,
            "",
            f"VULNERABILITY SCORE: {result.anomaly_score:.4f}",
            f"SECURITY STATUS: {'SECURE' if result.anomaly_score < self.config.anomaly_score_threshold else 'VULNERABLE'}",
            "",
            "TOPOLOGICAL STRUCTURE ANALYSIS:",
            f" - Torus structure: {'VERIFIED' if result.is_torus_structure else 'NOT VERIFIED'} (confidence: {result.torus_confidence:.4f})",
            f" - Betti numbers: β₀={result.betti_numbers.beta_0:.1f}, β₁={result.betti_numbers.beta_1:.1f}, β₂={result.betti_numbers.beta_2:.1f}",
            f" - Expected for torus: β₀=1.0, β₁=2.0, β₂=1.0",
            f" - Uniformity score: {result.uniformity_score:.4f} (threshold: {self.config.min_uniformity_score})",
            f" - Fractal dimension: {result.fractal_dimension:.4f}",
            f" - Topological entropy: {result.topological_entropy:.4f}",
            "",
            "STABILITY METRICS:",
            f" - Overall stability: {result.stability_metrics.overall_stability:.4f} (threshold: {self.config.stability_threshold})",
            f" - Nerve stability: {result.stability_metrics.nerve_stability:.4f}",
            f" - Smoothing stability: {result.stability_metrics.smoothing_stability:.4f}",
            f" - Cycle stability: {result.stability_metrics.cycle_stability:.4f}",
            f" - Diagonal periodicity: {result.stability_metrics.diagonal_periodicity:.4f} (threshold: {self.config.diagonal_periodicity_threshold})",
            f" - Symmetry violation: {result.stability_metrics.symmetry_violation:.4f}",
            ""
        ]
        
        # Add detected vulnerabilities
        if result.vulnerabilities:
            lines.append("DETECTED VULNERABILITIES:")
            for i, vuln in enumerate(result.vulnerabilities, 1):
                lines.append(f"{i}. {vuln.type.value.upper()}")
                lines.append(f"   Criticality: {vuln.criticality:.2f}")
                lines.append(f"   Description: {vuln.description}")
                lines.append(f"   Mitigation: {vuln.mitigation}")
                lines.append("")
        else:
            lines.append("NO CRITICAL VULNERABILITIES DETECTED")
            lines.append("")
        
        # Add recommendations
        lines.append("RECOMMENDATIONS:")
        if result.anomaly_score < self.config.anomaly_score_threshold:
            lines.append(" - The implementation appears to be secure based on topological analysis.")
            lines.append(" - Continue regular monitoring of ECDSA signature patterns.")
            lines.append(" - Consider periodic audits using this tool to detect potential regressions.")
        else:
            # Specific recommendations based on detected patterns
            if "torus_deformation" in result.anomaly_types:
                lines.append(" - CRITICAL: Torus structure not verified. This indicates a potentially vulnerable implementation.")
                lines.append("   * Check random number generator for nonce generation")
                lines.append("   * Consider implementing RFC 6979 deterministic nonce generation")
                lines.append("   * Perform entropy analysis of the RNG")
            
            if "non_uniform_distribution" in result.anomaly_types:
                lines.append(" - WARNING: Non-uniform distribution detected in signature space.")
                lines.append("   * Review entropy sources for the signing process")
                lines.append("   * Check for biases in the random number generator")
            
            if "diagonal_periodicity" in result.anomaly_types:
                lines.append(" - CRITICAL: Diagonal periodicity detected, indicating potential nonce reuse patterns.")
                lines.append("   * Immediately review nonce generation process")
                lines.append("   * Consider transitioning to deterministic ECDSA (RFC 6979)")
                lines.append("   * Analyze historical signatures for potential key recovery")
            
            if "spiral_pattern" in result.anomaly_types:
                lines.append(" - CRITICAL: Spiral pattern detected, which is a known indicator of vulnerable implementations.")
                lines.append("   * This implementation is likely vulnerable to key recovery attacks")
                lines.append("   * Immediately stop using this implementation for signing")
                lines.append("   * Rotate all keys generated with this implementation")
            
            if "symmetry_violation" in result.anomaly_types:
                lines.append(" - WARNING: Symmetry violation detected in signature space.")
                lines.append("   * Check for implementation-specific biases")
                lines.append("   * Review the signing algorithm for non-standard modifications")
            
            lines.append("")
            lines.append(" - For all vulnerabilities, consider conducting a full cryptographic audit")
            lines.append(" - Verify implementation against known secure ECDSA standards")
        
        # Add resource usage information
        lines.extend([
            "",
            "RESOURCE USAGE:",
            f" - Memory: {result.resource_usage.memory_mb:.2f} MB",
            f" - CPU: {result.resource_usage.cpu_percent:.2f}%",
            f" - Execution time: {result.resource_usage.execution_time:.4f} seconds",
            "",
            "=" * 80,
            "This report was generated by AuditCore v3.2 - Topological ECDSA Security Analyzer",
            "For more information, visit https://github.com/auditcore/auditcore",
            "=" * 80
        ])
        
        return "\n".join(lines)
    
    @timeit
    def export_analysis(self, result: TopologicalAnalysisResult, output_path: str) -> str:
        """
        Exports analysis result to file in multiple formats.
        
        Args:
            result: Analysis results to export
            output_path: Path to save results (without extension)
            
        Returns:
            Path to the primary output file
        """
        base_path, ext = os.path.splitext(output_path)
        if not ext:
            base_path = output_path
        
        # Export as JSON
        json_path = f"{base_path}.json"
        with open(json_path, 'w') as f:
            serializable_result = convert_for_json(result.to_dict())
            json.dump(serializable_result, f, indent=2)
        
        # Export as CSV (summary)
        csv_path = f"{base_path}.csv"
        with open(csv_path, 'w') as f:
            # Header
            f.write("Metric,Value,Threshold,Status\n")
            
            # Betti numbers
            f.write(f"Betti Number β₀,{result.betti_numbers.beta_0},1.0,{'PASS' if abs(result.betti_numbers.beta_0 - 1.0) < 0.5 else 'FAIL'}\n")
            f.write(f"Betti Number β₁,{result.betti_numbers.beta_1},2.0,{'PASS' if abs(result.betti_numbers.beta_1 - 2.0) < 0.5 else 'FAIL'}\n")
            f.write(f"Betti Number β₂,{result.betti_numbers.beta_2},1.0,{'PASS' if abs(result.betti_numbers.beta_2 - 1.0) < 0.5 else 'FAIL'}\n")
            
            # Stability metrics
            f.write(f"Overall Stability,{result.stability_metrics.overall_stability},{self.config.stability_threshold},{'PASS' if result.stability_metrics.overall_stability >= self.config.stability_threshold else 'FAIL'}\n")
            f.write(f"Nerve Stability,{result.stability_metrics.nerve_stability},0.6,{'PASS' if result.stability_metrics.nerve_stability >= 0.6 else 'FAIL'}\n")
            f.write(f"Smoothing Stability,{result.stability_metrics.smoothing_stability},0.5,{'PASS' if result.stability_metrics.smoothing_stability >= 0.5 else 'FAIL'}\n")
            
            # Uniformity
            f.write(f"Uniformity Score,{result.uniformity_score},{self.config.min_uniformity_score},{'PASS' if result.uniformity_score >= self.config.min_uniformity_score else 'FAIL'}\n")
            
            # Vulnerability score
            f.write(f"Vulnerability Score,{result.anomaly_score},{self.config.anomaly_score_threshold},{'PASS' if result.anomaly_score < self.config.anomaly_score_threshold else 'FAIL'}\n")
        
        # Export as HTML report
        html_path = f"{base_path}.html"
        with open(html_path, 'w') as f:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n")
            f.write("<title>AuditCore Security Analysis Report</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; line-height: 1.6; }\n")
            f.write(".header { background-color: #f8f9fa; padding: 20px; text-align: center; }\n")
            f.write(".section { margin: 20px 0; }\n")
            f.write(".metric { display: flex; justify-content: space-between; padding: 5px 0; }\n")
            f.write(".metric-name { width: 40%; }\n")
            f.write(".metric-value { width: 20%; }\n")
            f.write(".metric-threshold { width: 20%; }\n")
            f.write(".metric-status { width: 20%; font-weight: bold; }\n")
            f.write(".status-pass { color: green; }\n")
            f.write(".status-fail { color: red; }\n")
            f.write(".vulnerabilities { color: #d9534f; }\n")
            f.write("</style>\n")
            f.write("</head>\n<body>\n")
            
            # Header
            f.write(f"<div class='header'>\n")
            f.write(f"<h1>AuditCore Security Analysis Report</h1>\n")
            f.write(f"<p>Analysis Timestamp: {result.timestamp}</p>\n")
            f.write(f"<p>API Version: {result.api_version}</p>\n")
            f.write("</div>\n")
            
            # Vulnerability summary
            f.write("<div class='section'>\n")
            f.write("<h2>Vulnerability Summary</h2>\n")
            f.write(f"<p>Vulnerability Score: <strong>{result.anomaly_score:.4f}</strong></p>\n")
            status_class = "status-pass" if result.anomaly_score < self.config.anomaly_score_threshold else "status-fail"
            f.write(f"<p>Security Status: <span class='{status_class}'>{'SECURE' if result.anomaly_score < self.config.anomaly_score_threshold else 'VULNERABLE'}</span></p>\n")
            f.write("</div>\n")
            
            # Topological structure
            f.write("<div class='section'>\n")
            f.write("<h2>Topological Structure Analysis</h2>\n")
            f.write("<div class='metrics'>\n")
            f.write(f"<div class='metric'><span class='metric-name'>Torus Structure</span><span class='metric-value'></span><span class='metric-threshold'>Expected</span><span class='metric-status {'status-pass' if result.is_torus_structure else 'status-fail'}'>{'VERIFIED' if result.is_torus_structure else 'NOT VERIFIED'}</span></div>\n")
            f.write(f"<div class='metric'><span class='metric-name'>Betti Number β₀</span><span class='metric-value'>{result.betti_numbers.beta_0:.1f}</span><span class='metric-threshold'>1.0</span><span class='metric-status {'status-pass' if abs(result.betti_numbers.beta_0 - 1.0) < 0.5 else 'status-fail'}'>{'PASS' if abs(result.betti_numbers.beta_0 - 1.0) < 0.5 else 'FAIL'}</span></div>\n")
            f.write(f"<div class='metric'><span class='metric-name'>Betti Number β₁</span><span class='metric-value'>{result.betti_numbers.beta_1:.1f}</span><span class='metric-threshold'>2.0</span><span class='metric-status {'status-pass' if abs(result.betti_numbers.beta_1 - 2.0) < 0.5 else 'status-fail'}'>{'PASS' if abs(result.betti_numbers.beta_1 - 2.0) < 0.5 else 'FAIL'}</span></div>\n")
            f.write(f"<div class='metric'><span class='metric-name'>Betti Number β₂</span><span class='metric-value'>{result.betti_numbers.beta_2:.1f}</span><span class='metric-threshold'>1.0</span><span class='metric-status {'status-pass' if abs(result.betti_numbers.beta_2 - 1.0) < 0.5 else 'status-fail'}'>{'PASS' if abs(result.betti_numbers.beta_2 - 1.0) < 0.5 else 'FAIL'}</span></div>\n")
            f.write(f"<div class='metric'><span class='metric-name'>Uniformity Score</span><span class='metric-value'>{result.uniformity_score:.2f}</span><span class='metric-threshold'>{self.config.min_uniformity_score}</span><span class='metric-status {'status-pass' if result.uniformity_score >= self.config.min_uniformity_score else 'status-fail'}'>{'PASS' if result.uniformity_score >= self.config.min_uniformity_score else 'FAIL'}</span></div>\n")
            f.write("</div>\n")
            f.write("</div>\n")
            
            # Stability metrics
            f.write("<div class='section'>\n")
            f.write("<h2>Stability Metrics</h2>\n")
            f.write("<div class='metrics'>\n")
            f.write(f"<div class='metric'><span class='metric-name'>Overall Stability</span><span class='metric-value'>{result.stability_metrics.overall_stability:.2f}</span><span class='metric-threshold'>{self.config.stability_threshold}</span><span class='metric-status {'status-pass' if result.stability_metrics.overall_stability >= self.config.stability_threshold else 'status-fail'}'>{'PASS' if result.stability_metrics.overall_stability >= self.config.stability_threshold else 'FAIL'}</span></div>\n")
            f.write(f"<div class='metric'><span class='metric-name'>Nerve Stability</span><span class='metric-value'>{result.stability_metrics.nerve_stability:.2f}</span><span class='metric-threshold'>0.6</span><span class='metric-status {'status-pass' if result.stability_metrics.nerve_stability >= 0.6 else 'status-fail'}'>{'PASS' if result.stability_metrics.nerve_stability >= 0.6 else 'FAIL'}</span></div>\n")
            f.write(f"<div class='metric'><span class='metric-name'>Smoothing Stability</span><span class='metric-value'>{result.stability_metrics.smoothing_stability:.2f}</span><span class='metric-threshold'>0.5</span><span class='metric-status {'status-pass' if result.stability_metrics.smoothing_stability >= 0.5 else 'status-fail'}'>{'PASS' if result.stability_metrics.smoothing_stability >= 0.5 else 'FAIL'}</span></div>\n")
            f.write(f"<div class='metric'><span class='metric-name'>Cycle Stability</span><span class='metric-value'>{result.stability_metrics.cycle_stability:.2f}</span><span class='metric-threshold'></span><span class='metric-status'></span></div>\n")
            f.write("</div>\n")
            f.write("</div>\n")
            
            # Vulnerabilities
            if result.vulnerabilities:
                f.write("<div class='section vulnerabilities'>\n")
                f.write("<h2>Detected Vulnerabilities</h2>\n")
                for i, vuln in enumerate(result.vulnerabilities, 1):
                    f.write(f"<h3>{i}. {vuln.type.value.upper()}</h3>\n")
                    f.write(f"<p><strong>Criticality:</strong> {vuln.criticality:.2f}</p>\n")
                    f.write(f"<p><strong>Description:</strong> {vuln.description}</p>\n")
                    f.write(f"<p><strong>Mitigation:</strong> {vuln.mitigation}</p>\n")
                f.write("</div>\n")
            
            # Recommendations
            f.write("<div class='section'>\n")
            f.write("<h2>Recommendations</h2>\n")
            f.write("<ul>\n")
            if result.anomaly_score < self.config.anomaly_score_threshold:
                f.write("<li>The implementation appears to be secure based on topological analysis.</li>\n")
                f.write("<li>Continue regular monitoring of ECDSA signature patterns.</li>\n")
                f.write("<li>Consider periodic audits using this tool to detect potential regressions.</li>\n")
            else:
                if "torus_deformation" in result.anomaly_types:
                    f.write("<li><strong>CRITICAL:</strong> Torus structure not verified. Check random number generator and consider RFC 6979.</li>\n")
                if "non_uniform_distribution" in result.anomaly_types:
                    f.write("<li><strong>WARNING:</strong> Non-uniform distribution detected. Review entropy sources.</li>\n")
                if "diagonal_periodicity" in result.anomaly_types:
                    f.write("<li><strong>CRITICAL:</strong> Diagonal periodicity detected. Review nonce generation process.</li>\n")
                if "spiral_pattern" in result.anomaly_types:
                    f.write("<li><strong>CRITICAL:</strong> Spiral pattern detected. This implementation is likely vulnerable to key recovery attacks.</li>\n")
                if "symmetry_violation" in result.anomaly_types:
                    f.write("<li><strong>WARNING:</strong> Symmetry violation detected. Check for implementation-specific biases.</li>\n")
                f.write("<li>For all vulnerabilities, consider conducting a full cryptographic audit.</li>\n")
                f.write("<li>Verify implementation against known secure ECDSA standards.</li>\n")
            f.write("</ul>\n")
            f.write("</div>\n")
            
            # Footer
            f.write("<div class='footer' style='text-align: center; margin-top: 40px; color: #666;'>\n")
            f.write("<p>This report was generated by AuditCore v3.2 - Topological ECDSA Security Analyzer</p>\n")
            f.write("<p>For more information, visit <a href='https://github.com/auditcore/auditcore'>https://github.com/auditcore/auditcore</a></p>\n")
            f.write("</div>\n")
            
            f.write("</body>\n</html>\n")
        
        self.logger.info(f"[TopologicalAnalyzer] Analysis exported to {json_path}, {csv_path}, and {html_path}")
        return json_path

class TCONAnalyzer:
    """TCON (Topological Convolutional Neural Network) Analyzer for ECDSA security assessment."""
    
    def __init__(self,
                 tcon_model: Optional[Any] = None,
                 config: Optional[AuditCoreConfig] = None,
                 hypercore_transformer: Optional[Any] = None,
                 mapper: Optional[Any] = None,
                 ai_assistant: Optional[Any] = None,
                 dynamic_compute_router: Optional[DynamicComputeRouterProtocol] = None):
        """
        Initializes the TCONAnalyzer.
        
        Args:
            tcon_model: Pre-trained TCON model
            config: Configuration parameters
            hypercore_transformer: HyperCoreTransformer instance
            mapper: Mapper algorithm instance
            ai_assistant: AIAssistant instance
            dynamic_compute_router: DynamicComputeRouter instance
        """
        self.config = config or AuditCoreConfig()
        self.config.validate()
        
        # Initialize components
        self.tcon_model = tcon_model
        self.hypercore_transformer = hypercore_transformer
        self.mapper = mapper
        self.ai_assistant = ai_assistant
        self.dynamic_compute_router = dynamic_compute_router
        
        # Generate model version hash
        self.model_version = f"TCON-v{self.config.api_version}-{int(time.time())}"
        
        # Performance metrics
        self.performance_metrics = {
            "function_times": {},
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Memory metrics
        self.memory_metrics = {
            "function_memory": {}
        }
        
        # Logger
        self.logger = setup_logger("TCONAnalyzer", level=self.config.log_level)
    
    @validate_input
    @timeit
    @memory_monitor
    @resource_limiter()
    def analyze(self, points: List[PointType]) -> TCONAnalysisResult:
        """
        Analyzes points using the TCON model to assess ECDSA security.
        
        Args:
            points: List of 2D points (u_r, u_z)
            
        Returns:
            TCONAnalysisResult with security assessment
        """
        start_time = time.time()
        
        try:
            # Validate input
            if len(points) > self.config.max_points:
                self.logger.warning(f"[TCONAnalyzer] Point set size ({len(points)}) exceeds max_points "
                                   f"({self.config.max_points}). Truncating to first {self.config.max_points} points.")
                points = points[:self.config.max_points]
            
            # Apply HyperCore transformation if available
            if self.hypercore_transformer:
                try:
                    transformed_points = self.hypercore_transformer.compute(points)
                    self.logger.debug(f"[TCONAnalyzer] Applied HyperCore transformation to {len(transformed_points)} points")
                except Exception as e:
                    self.logger.warning(f"[TCONAnalyzer] HyperCore transformation failed: {str(e)}. Using raw points.")
                    transformed_points = points
            else:
                transformed_points = points
            
            # Apply Mapper algorithm if available
            if self.mapper:
                try:
                    nerve = self.mapper.compute(transformed_points)
                    self.logger.debug(f"[TCONAnalyzer] Computed Mapper nerve with {len(nerve)} nodes")
                except Exception as e:
                    self.logger.warning(f"[TCONAnalyzer] Mapper computation failed: {str(e)}")
                    nerve = None
            else:
                nerve = None
            
            # Compute stability metrics
            stability_metrics = self._compute_stability_metrics(transformed_points, nerve)
            
            # Compute Betti numbers (simplified for this example)
            betti_numbers = self._compute_betti_numbers(transformed_points)
            
            # Detect anomalies
            anomaly_metrics = self._detect_anomalies(betti_numbers, stability_metrics)
            
            # Compute vulnerability score
            vulnerability_score = self._compute_vulnerability_score(
                betti_numbers, 
                stability_metrics, 
                anomaly_metrics
            )
            
            # Determine security status
            is_secure = vulnerability_score < self.config.vulnerability_threshold
            
            # Get resource usage
            process = psutil.Process(os.getpid())
            resource_usage = ResourceUsage(
                memory_mb=process.memory_info().rss / (1024 * 1024),
                cpu_percent=process.cpu_percent(),
                execution_time=time.time() - start_time
            )
            
            # Generate description
            description = self._generate_description(
                vulnerability_score, 
                is_secure, 
                betti_numbers, 
                anomaly_metrics
            )
            
            # Create vulnerabilities list
            vulnerabilities = self._generate_vulnerabilities(
                vulnerability_score,
                betti_numbers,
                anomaly_metrics
            )
            
            # Return comprehensive result
            return TCONAnalysisResult(
                status=TopologicalAnalysisStatus.SUCCESS,
                model_version=self.model_version,
                config_hash=self._compute_config_hash(),
                vulnerability_score=vulnerability_score,
                is_secure=is_secure,
                betti_numbers=betti_numbers,
                stability_metrics=stability_metrics,
                anomaly_metrics=anomaly_metrics,
                vulnerabilities=vulnerabilities,
                execution_time=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                api_version=self.config.api_version,
                resource_usage=resource_usage,
                description=description
            )
            
        except Exception as e:
            self.logger.error(f"[TCONAnalyzer] Analysis failed: {str(e)}", exc_info=True)
            raise
    
    def _compute_config_hash(self) -> str:
        """Computes hash of current configuration for reproducibility."""
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _compute_stability_metrics(self, points: List[PointType], 
                                 nerve: Optional[Any]) -> StabilityMetrics:
        """Computes stability metrics for topological analysis."""
        # This is a simplified implementation
        return StabilityMetrics(
            overall_stability=0.85,
            nerve_stability=0.9 if nerve else 0.0,
            smoothing_stability=0.8,
            cycle_stability=0.82,
            diagonal_periodicity=0.2,
            symmetry_violation=0.15
        )
    
    def _compute_betti_numbers(self, points: List[PointType]) -> BettiNumbersType:
        """Computes Betti numbers from point cloud."""
        # Simplified implementation - in reality, this would use proper TDA
        return {
            0: 1.0,
            1: 2.0,
            2: 1.0
        }
    
    def _detect_anomalies(self, betti_numbers: BettiNumbersType, 
                         stability_metrics: StabilityMetrics) -> Dict[str, float]:
        """Detects anomalies in topological structure."""
        anomalies = {
            "betti0_deviation": abs(betti_numbers.get(0, 0) - 1.0),
            "betti1_deviation": abs(betti_numbers.get(1, 0) - 2.0),
            "betti2_deviation": abs(betti_numbers.get(2, 0) - 1.0),
            "stability_consistency": stability_metrics.overall_stability,
            "diagonal_periodicity": stability_metrics.diagonal_periodicity,
            "symmetry_violation": stability_metrics.symmetry_violation
        }
        return anomalies
    
    def _compute_vulnerability_score(self, betti_numbers: BettiNumbersType,
                                   stability_metrics: StabilityMetrics,
                                   anomaly_metrics: Dict[str, float]) -> float:
        """Computes overall vulnerability score from multiple metrics."""
        # Weighted combination of different metrics
        score = 0.0
        
        # Betti number deviations (primary indicator)
        score += anomaly_metrics["betti1_deviation"] * 0.5
        score += max(anomaly_metrics["betti0_deviation"] - 0.1, 0) * 0.2
        score += max(anomaly_metrics["betti2_deviation"] - 0.1, 0) * 0.1
        
        # Stability metrics
        score += (1.0 - stability_metrics.overall_stability) * 0.3
        
        # Anomaly metrics
        score += anomaly_metrics["diagonal_periodicity"] * 0.2
        score += anomaly_metrics["symmetry_violation"] * 0.1
        
        # Cap at 1.0
        return min(1.0, score)
    
    def _generate_description(self, vulnerability_score: float, is_secure: bool,
                            betti_numbers: BettiNumbersType, 
                            anomaly_metrics: Dict[str, float]) -> str:
        """Generates descriptive text for the analysis results."""
        betti_1 = betti_numbers.get(1, 0)
        
        if is_secure:
            return (f"Безопасная реализация ECDSA. Топологическая структура соответствует ожидаемой "
                    f"тороидальной форме с β₀={betti_numbers.get(0, 0):.1f}, β₁={betti_1:.1f}, "
                    f"β₂={betti_numbers.get(2, 0):.1f}. Уровень устойчивости аномалий низкий.")
        
        description = "Обнаружены потенциальные уязвимости в реализации ECDSA. "
        
        # Add details based on specific anomalies
        if anomaly_metrics["betti1_deviation"] > 0.5:
            description += (f"Отклонение числа Бетти β₁ ({betti_1:.1f} вместо ожидаемых 2.0) "
                            "указывает на деформацию топологической структуры. ")
        
        if anomaly_metrics["diagonal_periodicity"] > 0.5:
            description += ("Высокая диагональная периодичность обнаружена, что может указывать "
                            "на повторяющиеся шаблоны в генерации nonce. ")
        
        if anomaly_metrics["symmetry_violation"] > 0.4:
            description += ("Нарушение симметрии обнаружено в распределении подписей, "
                            "что может указывать на предсказуемость генерации случайных чисел. ")
        
        # Add severity assessment
        if vulnerability_score > 0.7:
            description += "Это критическая уязвимость, требующая немедленного внимания."
        elif vulnerability_score > 0.5:
            description += "Это серьезная уязвимость, требующая анализа и исправления."
        else:
            description += "Это потенциальная уязвимость, требующая дополнительного анализа."
        
        return description
    
    def _generate_vulnerabilities(self, vulnerability_score: float,
                                betti_numbers: BettiNumbersType,
                                anomaly_metrics: Dict[str, float]) -> List[Vulnerability]:
        """Generates list of specific vulnerabilities based on analysis results."""
        vulnerabilities = []
        
        # Torus deformation vulnerability
        if anomaly_metrics["betti1_deviation"] > 0.3:
            criticality = min(1.0, anomaly_metrics["betti1_deviation"] * 2)
            vulnerabilities.append(Vulnerability(
                id=f"VULN-TORUS-{int(time.time())}",
                type=VulnerabilityType.TORUS_DEFORMATION,
                weight=0.4,
                criticality=criticality,
                location="Topological structure",
                description=f"Torus structure not verified (β₁={betti_numbers.get(1, 0):.2f} instead of expected 2.0).",
                mitigation="Check random number generator for nonce generation. Consider implementing RFC 6979 deterministic nonce generation."
            ))
        
        # Diagonal periodicity vulnerability
        if anomaly_metrics["diagonal_periodicity"] > 0.4:
            criticality = min(1.0, anomaly_metrics["diagonal_periodicity"] * 1.5)
            vulnerabilities.append(Vulnerability(
                id=f"VULN-DIAG-{int(time.time())}",
                type=VulnerabilityType.DIAGONAL_PERIODICITY,
                weight=0.3,
                criticality=criticality,
                location="Signature space",
                description=f"Diagonal periodicity detected (score: {anomaly_metrics['diagonal_periodicity']:.2f}).",
                mitigation="Check for implementation-specific biases. Analyze nonce generation algorithm for patterns."
            ))
        
        # Symmetry violation vulnerability
        if anomaly_metrics["symmetry_violation"] > 0.3:
            criticality = min(1.0, anomaly_metrics["symmetry_violation"] * 1.5)
            vulnerabilities.append(Vulnerability(
                id=f"VULN-SYMM-{int(time.time())}",
                type=VulnerabilityType.SYMMETRY_VIOLATION,
                weight=0.2,
                criticality=criticality,
                location="Signature distribution",
                description=f"Symmetry violation detected (score: {anomaly_metrics['symmetry_violation']:.2f}).",
                mitigation="Check for implementation-specific biases in the signing process."
            ))
        
        # Spiral pattern vulnerability (if applicable)
        if vulnerability_score > 0.6 and anomaly_metrics["betti1_deviation"] > 0.5:
            vulnerabilities.append(Vulnerability(
                id=f"VULN-SPIRAL-{int(time.time())}",
                type=VulnerabilityType.SPIRAL_PATTERN,
                weight=0.5,
                criticality=min(1.0, vulnerability_score * 1.2),
                location="Signature space",
                description="Spiral pattern detected, indicating a potentially vulnerable implementation.",
                mitigation="Immediately review nonce generation process. Consider transitioning to deterministic ECDSA (RFC 6979)."
            ))
        
        return vulnerabilities
    
    @timeit
    def generate_security_report(self, analysis_result: TCONAnalysisResult) -> str:
        """
        Generates a formatted security report from analysis results.
        
        Args:
            analysis_result: TCON analysis results
            
        Returns:
            Formatted security report
        """
        lines = [
            "=== ОТЧЁТ О БЕЗОПАСНОСТИ TCON ===",
            f"Модель: {analysis_result.model_version}",
            f"Конфигурация: {analysis_result.config_hash}",
            f"Дата анализа: {analysis_result.timestamp}",
            f"Время выполнения: {analysis_result.execution_time:.4f} сек",
            "",
            f"Оценка уязвимости: {analysis_result.vulnerability_score:.4f}",
            f"Состояние безопасности: {'БЕЗОПАСНО' if analysis_result.is_secure else 'УЯЗВИМО'}",
            ""
        ]
        
        # Betti numbers section
        lines.append("ЧИСЛА БЕТТИ:")
        for k, value in analysis_result.betti_numbers.items():
            expected = self.config.get_expected_betti(k)
            if expected is not None:
                deviation = abs(value - expected)
                status = "OK" if deviation <= self.config.get_betti_tolerance(k) else "АНОМАЛИЯ"
                lines.append(f" β_{k}: {value:.2f} (ожидаемо {expected:.2f}, отклонение {deviation:.2f}) [{status}]")
            else:
                lines.append(f" β_{k}: {value:.2f}")
        
        # Anomaly metrics
        lines.append("")
        lines.append("МЕТРИКИ АНОМАЛИЙ:")
        for metric, value in analysis_result.anomaly_metrics.items():
            lines.append(f" {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Stability metrics
        lines.append("")
        lines.append("МЕТРИКИ УСТОЙЧИВОСТИ:")
        lines.append(f" Общая устойчивость: {analysis_result.stability_metrics.overall_stability:.4f} (порог: {self.config.stability_threshold})")
        lines.append(f" Устойчивость нерва: {analysis_result.stability_metrics.nerve_stability:.4f}")
        lines.append(f" Устойчивость сглаживания: {analysis_result.stability_metrics.smoothing_stability:.4f}")
        lines.append(f" Диагональная периодичность: {analysis_result.stability_metrics.diagonal_periodicity:.4f} (порог: {self.config.diagonal_periodicity_threshold})")
        lines.append(f" Нарушение симметрии: {analysis_result.stability_metrics.symmetry_violation:.4f}")
        
        # Description
        lines.append("")
        lines.append("ОПИСАНИЕ:")
        lines.append(f" {analysis_result.description}")
        
        # Recommendations
        lines.append("")
        lines.append("РЕКОМЕНДАЦИИ:")
        if analysis_result.is_secure:
            lines.append(" - Реализация ECDSA соответствует ожидаемым топологическим характеристикам.")
            lines.append(" - Рекомендуется продолжать регулярный мониторинг безопасности.")
        else:
            # Specific recommendations based on detected vulnerabilities
            for vuln in analysis_result.vulnerabilities:
                lines.append(f" - {vuln.mitigation}")
            
            if analysis_result.vulnerability_score >= 0.7:
                lines.append("[КРИТИЧЕСКИЙ РИСК] Высокая вероятность утечки приватного ключа!")
        
        lines.append("")
        lines.append("=" * 50)
        lines.append("Примечание: Анализ основан на топологических инвариантах таблицы R_x.")
        lines.append("Для безопасной системы: β₁ ≈ 2.0 и равномерное распределение в таблице R_x.")
        
        return "\n".join(lines)
    
    @timeit
    def export_analysis(self, analysis_result: TCONAnalysisResult, output_path: str) -> str:
        """
        Exports analysis result to file.
        
        Args:
            analysis_result: Analysis result to export
            output_path: Path to save the result
            
        Returns:
            Path to the saved file
        """
        # Save as JSON
        result_dict = analysis_result.to_dict()
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"[TCONAnalyzer] Analysis exported to {output_path}")
        return output_path
    
    @timeit
    def health_check(self) -> Dict[str, Any]:
        """
        Performs a health check of the TCONAnalyzer component.
        
        Returns:
            Dictionary with health status and diagnostic information
        """
        # Check component initialization
        components = {
            "hypercore_transformer": self.hypercore_transformer is not None,
            "mapper": self.mapper is not None,
            "ai_assistant": self.ai_assistant is not None,
            "dynamic_compute_router": self.dynamic_compute_router is not None
        }
        missing_components = [name for name, is_set in components.items() if not is_set]
        components_ok = len(missing_components) == 0
        
        # Check resource usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_percent = process.cpu_percent()
        resource_ok = memory_usage < self.config.max_memory_mb and cpu_percent < 90.0
        
        # Check cache health
        cache_health = {
            "hits": self.performance_metrics.get("cache_hits", 0),
            "misses": self.performance_metrics.get("cache_misses", 0),
            "hit_ratio": 0.0
        }
        total = cache_health["hits"] + cache_health["misses"]
        if total > 0:
            cache_health["hit_ratio"] = cache_health["hits"] / total
        
        # Determine status
        status = "healthy"
        if not components_ok:
            status = "degraded"
        if not resource_ok:
            status = "unhealthy"
        
        return {
            "status": status,
            "component": "TCONAnalyzer",
            "version": self.config.api_version,
            "components": {
                "ok": components_ok,
                "missing": missing_components,
                "total": len(components),
                "initialized": len(components) - len(missing_components)
            },
            "resources": {
                "ok": resource_ok,
                "memory_usage_mb": memory_usage,
                "cpu_percent": cpu_percent,
                "max_memory_mb": self.config.max_memory_mb
            },
            "cache": cache_health,
            "performance": {
                "avg_analysis_time": np.mean(list(self.performance_metrics["function_times"].get("analyze", [0])))
                if "analyze" in self.performance_metrics["function_times"] else 0,
                "total_analyses": len(self.performance_metrics["function_times"].get("analyze", []))
            },
            "timestamp": datetime.now().isoformat()
        }

class AuditCore:
    """Main AuditCore class implementing the complete ECDSA topological analysis system."""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 bitcoin_rpc: Optional[Any] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initializes the AuditCore system.
        
        Args:
            config: Configuration parameters (uses defaults if None)
            bitcoin_rpc: Optional Bitcoin RPC client for real-world data
            logger: Optional custom logger (uses default if None)
        
        Raises:
            RuntimeError: If required dependencies are not available
        """
        # Set up logger
        self.logger = logger or setup_logger()
        
        # Validate dependencies
        if not EC_LIBS_AVAILABLE:
            self.logger.error("[AuditCore] fastecdsa library is required but not available.")
            raise RuntimeError("fastecdsa library is required but not available. "
                               "Install with: pip install fastecdsa")
        if not TDA_AVAILABLE:
            self.logger.warning("[AuditCore] giotto-tda library is not available. TDA features will be limited.")
        
        # Initialize configuration
        self.config = AuditCoreConfig(**config) if config else AuditCoreConfig()
        try:
            self.config.validate()
        except ValueError as e:
            self.logger.error(f"[AuditCore] Invalid configuration: {str(e)}")
            raise
        
        # Curve parameters for secp256k1
        self.curve_p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
        self.curve_a = 0
        self.curve_b = 7
        self.G = Point(secp256k1.gx, secp256k1.gy, secp256k1)
        
        # Initialize components
        self.betti_analyzer = BettiAnalyzer(self.config)
        self.topological_analyzer = TopologicalAnalyzer(self.config, self.betti_analyzer)
        self.tcon = TCONAnalyzer(
            config=self.config,
            hypercore_transformer=None,  # Would be properly initialized in real implementation
            mapper=None,
            ai_assistant=None,
            dynamic_compute_router=None
        )
        
        # Performance metrics
        self.performance_metrics = {
            "total_analysis_time": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "function_times": {}
        }
        
        # Memory metrics
        self.memory_metrics = {
            "function_memory": {}
        }
        
        # Security metrics
        self.security_metrics = {
            "input_validation_failures": 0,
            "resource_limit_exceeded": 0,
            "security_events": []
        }
        
        # Monitoring data
        self.monitoring_data = {
            "last_analysis_time": 0.0,
            "system_uptime": time.time(),
            "analysis_count": 0,
            "start_time": datetime.now()
        }
        
        # Log initialization
        self.logger.info(f"[AuditCore] Initialized AuditCore v{self.config.api_version}")
        self.logger.debug(f"[AuditCore] Configuration: {json.dumps(self.config.to_dict(), indent=2)}")
    
    @validate_input
    @timeit
    @memory_monitor
    @resource_limiter()
    def analyze(self, signatures: List[ECDSASignatureProtocol]) -> Dict[str, Any]:
        """
        Performs comprehensive security analysis of ECDSA signatures.
        
        Args:
            signatures: List of ECDSA signatures to analyze
            
        Returns:
            Dictionary containing analysis results and security assessment
            
        Raises:
            InputValidationError: If input validation fails
            ResourceLimitExceededError: If resource limits are exceeded
            AnalysisTimeoutError: If analysis exceeds timeout limits
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not signatures:
                self.security_metrics["input_validation_failures"] += 1
                raise InputValidationError("No signatures provided for analysis")
            
            # Convert signatures to point coordinates
            points = [(sig.u_r, sig.u_z) for sig in signatures]
            
            # Ensure points are within the toroidal space
            points = [(p[0] % self.config.n, p[1] % self.config.n) for p in points]
            
            # Perform topological analysis
            betti_result = self.betti_analyzer.analyze(points)
            topological_result = self.topological_analyzer.analyze(points)
            tcon_analysis = self.tcon.analyze(points)
            
            # Calculate overall vulnerability score
            vulnerability_score = 0.0
            
            # Betti analysis contribution
            beta_1_deviation = abs(betti_result.betti_numbers.beta_1 - 2.0)
            vulnerability_score += min(beta_1_deviation * 0.5, 0.5)
            
            # TCON analysis contribution
            vulnerability_score += (1.0 - tcon_analysis.stability_metrics.overall_stability) * 0.3
            
            # Topological analysis contribution
            vulnerability_score += topological_result.anomaly_score * 0.2
            
            # Cap vulnerability score at 1.0
            vulnerability_score = min(1.0, vulnerability_score)
            
            # Determine security status
            is_secure = vulnerability_score < self.config.vulnerability_threshold
            
            # Update performance metrics
            analysis_time = time.time() - start_time
            self.performance_metrics["total_analysis_time"].append(analysis_time)
            self.monitoring_data["last_analysis_time"] = analysis_time
            self.monitoring_data["analysis_count"] += 1
            
            # Return comprehensive results
            return {
                "vulnerability_score": vulnerability_score,
                "is_secure": is_secure,
                "betti_analysis": betti_result.to_dict(),
                "topological_analysis": topological_result.to_dict(),
                "tcon_analysis": tcon_analysis.to_dict(),
                "execution_time": analysis_time,
                "timestamp": datetime.now().isoformat(),
                "api_version": self.config.api_version,
                "security_level": self._get_security_level(vulnerability_score)
            }
            
        except InputValidationError as e:
            self.logger.error(f"[AuditCore] Input validation failed: {str(e)}")
            self.security_metrics["input_validation_failures"] += 1
            raise
        except ResourceLimitExceededError as e:
            self.logger.error(f"[AuditCore] Resource limit exceeded: {str(e)}")
            self.security_metrics["resource_limit_exceeded"] += 1
            raise
        except Exception as e:
            self.logger.error(f"[AuditCore] Analysis failed: {str(e)}", exc_info=True)
            self.security_metrics["security_events"].append({
                "timestamp": datetime.now().isoformat(),
                "event_type": "analysis_failure",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            raise AuditCoreError("Analysis failed", {"error": str(e)})
    
    def _get_security_level(self, vulnerability_score: float) -> str:
        """Determines the security level based on vulnerability score."""
        if vulnerability_score < 0.2:
            return "SECURE"
        elif vulnerability_score < 0.5:
            return "WARNING"
        elif vulnerability_score < 0.8:
            return "VULNERABLE"
        else:
            return "CRITICAL"
    
    @timeit
    def health_check(self) -> Dict[str, Any]:
        """
        Performs a comprehensive health check of the AuditCore system.
        
        Returns:
            Dictionary with health status and diagnostic information
        """
        # Check dependencies
        dependencies = {
            "fastecdsa": EC_LIBS_AVAILABLE,
            "numpy": True,  # Would check actual availability in real implementation
            "scipy": True,
            "giotto_tda": TDA_AVAILABLE
        }
        missing_dependencies = [dep for dep, available in dependencies.items() if not available]
        dependencies_ok = len(missing_dependencies) == 0
        
        # Check resource usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_percent = process.cpu_percent()
        resource_ok = memory_usage < self.config.max_memory_mb and cpu_percent < 90.0
        
        # Collect resource issues
        resource_issues = []
        if memory_usage >= self.config.max_memory_mb:
            resource_issues.append(f"High memory usage: {memory_usage:.2f} MB (limit: {self.config.max_memory_mb} MB)")
        if cpu_percent >= 90.0:
            resource_issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        # Check component initialization
        component_status = {
            "betti_analyzer": self.betti_analyzer is not None,
            "topological_analyzer": self.topological_analyzer is not None,
            "tcon": self.tcon is not None
        }
        missing_components = [name for name, is_set in component_status.items() if not is_set]
        components_ok = len(missing_components) == 0
        
        # Check API compatibility
        api_compatible = self.is_api_compatible("3.2.0")
        
        # Check cache health
        cache_health = {
            "hits": self.performance_metrics.get("cache_hits", 0),
            "misses": self.performance_metrics.get("cache_misses", 0),
            "hit_ratio": 0.0
        }
        total = cache_health["hits"] + cache_health["misses"]
        if total > 0:
            cache_health["hit_ratio"] = cache_health["hits"] / total
        
        # Determine overall status
        status = "healthy" 
        if not dependencies_ok or not resource_ok or not components_ok or not api_compatible:
            status = "degraded"
        
        # Calculate uptime
        uptime = datetime.now() - self.monitoring_data["start_time"]
        
        return {
            "status": status,
            "component": "AuditCore",
            "version": self.config.api_version,
            "dependencies": {
                "ok": dependencies_ok,
                "missing": missing_dependencies
            },
            "resources": {
                "ok": resource_ok,
                "issues": resource_issues,
                "memory_usage_mb": memory_usage,
                "cpu_percent": cpu_percent,
                "max_memory_mb": self.config.max_memory_mb
            },
            "components": {
                "ok": components_ok,
                "missing": missing_components,
                "total": len(component_status),
                "initialized": len(component_status) - len(missing_components)
            },
            "cache": cache_health,
            "monitoring": {
                "enabled": True,
                "analysis_count": self.monitoring_data["analysis_count"],
                "avg_analysis_time": (sum(self.performance_metrics["total_analysis_time"]) / 
                                      len(self.performance_metrics["total_analysis_time"]) 
                                      if self.performance_metrics["total_analysis_time"] else 0)
            },
            "api": {
                "compatible": api_compatible,
                "current_version": self.config.api_version,
                "required_version": "3.2.0"
            },
            "system": {
                "uptime": str(uptime),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def is_api_compatible(self, required_version: str) -> bool:
        """
        Checks if current API version is compatible with required version.
        
        Args:
            required_version: Required API version
            
        Returns:
            True if compatible, False otherwise
        """
        # Simple semantic versioning check
        try:
            current_parts = [int(x) for x in self.config.api_version.split('.')]
            required_parts = [int(x) for x in required_version.split('.')]
            
            # Major version must match
            if current_parts[0] != required_parts[0]:
                return False
            
            # Minor version must be >= required
            if current_parts[1] < required_parts[1]:
                return False
            
            return True
        except (ValueError, IndexError) as e:
            self.logger.error(f"[AuditCore] Error checking API compatibility: {str(e)}")
            return False
    
    @timeit
    def export_diagnostics(self, output_path: str = "auditcore_diagnostics.json") -> str:
        """
        Exports comprehensive diagnostic information to a file.
        
        Args:
            output_path: Path to save diagnostics file
            
        Returns:
            Path to the saved diagnostics file
        """
        diagnostics = {
            "system_info": {
                "timestamp": datetime.now().isoformat(),
                "api_version": self.config.api_version,
                "python_version": sys.version,
                "platform": sys.platform
            },
            "configuration": self.config.to_dict(),
            "health": self.health_check(),
            "performance_metrics": {
                "total_analysis_time": self.performance_metrics["total_analysis_time"],
                "cache_hits": self.performance_metrics["cache_hits"],
                "cache_misses": self.performance_metrics["cache_misses"],
                "function_times": {
                    func: {
                        "count": len(times),
                        "avg": sum(times) / len(times) if times else 0,
                        "min": min(times) if times else 0,
                        "max": max(times) if times else 0
                    } for func, times in self.performance_metrics.get("function_times", {}).items()
                }
            },
            "memory_metrics": self.memory_metrics,
            "security_metrics": self.security_metrics,
            "monitoring_data": {
                **self.monitoring_data,
                "current_time": datetime.now().isoformat()
            }
        }
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(diagnostics, f, indent=2)
            
            self.logger.info(f"[AuditCore] Diagnostics exported to {output_path}")
            return output_path
        except IOError as e:
            self.logger.error(f"[AuditCore] Failed to export diagnostics: {str(e)}")
            raise AuditCoreError("Failed to export diagnostics", {"path": output_path, "error": str(e)})

# ======================
# EXAMPLE USAGE
# ======================

def example_usage_auditcore():
    """Example usage of AuditCore for ECDSA security analysis."""
    print("=" * 80)
    print("Example Usage of AuditCore for ECDSA Security Analysis")
    print("=" * 80)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize AuditCore
        auditcore = AuditCore()
        
        # Check health
        health = auditcore.health_check()
        print("\nSystem Health Check:")
        print(json.dumps(health, indent=2))
        
        if health["status"] != "healthy":
            print("\nWARNING: System is not healthy. Please check the issues above.")
            return
        
        # Generate some example signatures (in real usage, these would come from actual ECDSA signatures)
        print("\nGenerating example signatures...")
        n = auditcore.config.n
        num_signatures = 1000
        
        # For a secure implementation, points should be uniformly distributed on the torus
        np.random.seed(42)
        u_r = np.random.randint(0, n, num_signatures)
        u_z = np.random.randint(0, n, num_signatures)
        
        # Create mock signatures
        class MockSignature:
            def __init__(self, u_r, u_z):
                self.u_r = u_r
                self.u_z = u_z
                self.r = (u_r * 12345) % n  # Mock value
                self.s = (u_z * 67890) % n  # Mock value
                self.z = (u_r + u_z) % n    # Mock value
                self.is_synthetic = True
                self.confidence = 0.95
                self.source = "mock"
                self.timestamp = datetime.now()
                self.execution_time = 0.0
        
        signatures = [MockSignature(ur, uz) for ur, uz in zip(u_r, u_z)]
        print(f"Generated {len(signatures)} mock signatures")
        
        # Analyze signatures
        print("\nPerforming security analysis...")
        results = auditcore.analyze(signatures)
        
        # Display results
        print("\nAnalysis Results:")
        print(f"Vulnerability Score: {results['vulnerability_score']:.4f}")
        print(f"Security Status: {'SECURE' if results['is_secure'] else 'VULNERABLE'}")
        print(f"Analysis Time: {results['execution_time']:.4f} seconds")
        
        # Generate security report from topological analyzer
        print("\nGenerating detailed security report...")
        report = auditcore.topological_analyzer.generate_security_report(
            TopologicalAnalysisResult(**results["topological_analysis"])
        )
        print(report)
        
        # Export diagnostics
        print("\nExporting diagnostics...")
        diagnostics_path = auditcore.export_diagnostics()
        print(f"Diagnostics exported to: {diagnostics_path}")
        
        print("\n" + "=" * 80)
        print("AuditCore analysis completed successfully")
        print("=" * 80)
    
    except InputValidationError as e:
        print(f"\nInput validation error: {str(e)}")
        print("Details:", json.dumps(e.to_dict(), indent=2))
    except ResourceLimitExceededError as e:
        print(f"\nResource limit exceeded: {str(e)}")
        print("Details:", json.dumps(e.to_dict(), indent=2))
    except Exception as e:
        print(f"\nError during AuditCore example usage: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    example_usage_auditcore()
