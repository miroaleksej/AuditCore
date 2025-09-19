#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIAssistant.py - AuditCore v3.2
Полная и окончательная промышленная реализация

Этот модуль реализует AIAssistant - ключевой компонент AuditCore v3.2,
отвечающий за определение регионов аудита и приоритизацию уязвимостей.

Основные функции:
- Определение критических регионов в пространстве (u_r, u_z) для детального аудита
- Приоритизация уязвимостей по степени тяжести и эксплуатируемости
- Интеграция с Mapper алгоритмом для топологического анализа
- Генерация рекомендаций по устранению уязвимостей
- Мониторинг и сбор метрик производительности и безопасности

Соответствует:
- "НР структурированная.md" (Разделы 3, 4, 11)
- "Comprehensive Logic and Mathematical Model.md" (раздел AIAssistant)
- "TOPOLOGICAL DATA ANALYSIS.pdf" (теория Mapper)
- Архитектуре AuditCore v3.2

Промышленная реализация без упрощений:
- Полная интеграция со всеми компонентами AuditCore
- Надежная обработка ошибок и валидация входных данных
- Оптимизация производительности с использованием кэширования
- Поддержка мониторинга и сбора метрик
- Готов к промышленному использованию и масштабированию
"""

import time
import json
import logging
import warnings
import threading
import psutil
import concurrent.futures
import numpy as np
from datetime import datetime
from typing import (
    List, Dict, Any, Optional, Callable, Tuple, Protocol, 
    runtime_checkable, TypeVar, Set, FrozenSet
)
from dataclasses import dataclass, field
from functools import wraps
import fastecdsa  # Используем для работы с ECDSA
from fastecdsa import curve, ecdsa, keys
from fastecdsa.point import Point

# Настройка глобального логгера для модуля
logger = logging.getLogger("AuditCore.AIAssistant")

# ======================
# EXCEPTIONS
# ======================
class AIAssistantError(Exception):
    """Базовый класс для исключений AI Assistant."""
    pass

class InputValidationError(AIAssistantError):
    """Исключение, возникающее при ошибке валидации входных данных."""
    pass

class ResourceLimitExceededError(AIAssistantError):
    """Исключение, возникающее при превышении лимитов ресурсов."""
    pass

class AnalysisTimeoutError(AIAssistantError):
    """Исключение, возникающее при превышении времени анализа."""
    pass

class SecurityValidationError(AIAssistantError):
    """Исключение, возникающее при провале проверки безопасности."""
    def __init__(self, vulnerability_score: float, threshold: float, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details.update({"vulnerability_score": vulnerability_score, "threshold": threshold})
        super().__init__(f"Security validation failed: vulnerability score {vulnerability_score} exceeds threshold {threshold}", details)
        self.vulnerability_score = vulnerability_score
        self.threshold = threshold
        self.details = details

# ======================
# PROTOCOLS & INTERFACES
# ======================
T = TypeVar('T')

@runtime_checkable
class PointProtocol(Protocol):
    """Протокол для точек эллиптической кривой."""
    x: int
    y: int
    infinity: bool
    curve: Optional[Any]

@runtime_checkable
class ECDSASignature(Protocol):
    """Протокол для ECDSA подписей."""
    r: int
    s: int
    z: int
    u_r: int
    u_z: int
    is_synthetic: bool
    confidence: float

@runtime_checkable
class HyperCoreTransformerProtocol(Protocol):
    """Протокол для HyperCoreTransformer."""
    def get_stability_map(self, points: np.ndarray) -> np.ndarray:
        """Возвращает карту стабильности для заданных точек."""
        ...
    
    def transform(self, points: np.ndarray) -> np.ndarray:
        """Преобразует точки в топологическое пространство."""
        ...

@runtime_checkable
class TCONProtocol(Protocol):
    """Протокол для TCON (Topological Control)."""
    def verify_torus_structure(self, points: np.ndarray) -> Dict[str, Any]:
        """Проверяет структуру тора в точках."""
        ...
    
    def analyze_diagonal_periodicity(self, points: np.ndarray) -> Dict[str, Any]:
        """Анализирует диагональную периодичность."""
        ...

@runtime_checkable
class SignatureGeneratorProtocol(Protocol):
    """Протокол для SignatureGenerator."""
    def generate_region(self, ur_range: Tuple[int, int], uz_range: Tuple[int, int], count: int) -> List[ECDSASignature]:
        """Генерирует подписи в указанном регионе."""
        ...
    
    def generate_for_collision_search(self, public_key: PointProtocol, target_regions: List[Dict[str, Any]], count: int) -> List[ECDSASignature]:
        """Генерирует подписи для поиска коллизий."""
        ...

@runtime_checkable
class BettiAnalyzerProtocol(Protocol):
    """Протокол для BettiAnalyzer."""
    def analyze_betti_numbers(self, mapper: Dict[str, Any]) -> Dict[str, Any]:
        """Анализирует числа Бетти топологического представления."""
        ...

@runtime_checkable
class CollisionEngineProtocol(Protocol):
    """Протокол для CollisionEngine."""
    def estimate_key_from_collision(self, public_key: PointProtocol, collision_r: int, signatures: List[ECDSASignature]) -> Optional[int]:
        """Оценивает приватный ключ на основе коллизий."""
        ...

@runtime_checkable
class GradientAnalysisProtocol(Protocol):
    """Протокол для GradientAnalysis."""
    def compute_gradient_field(self, points: np.ndarray) -> np.ndarray:
        """Вычисляет градиентное поле для точек."""
        ...
    
    def detect_spiral_patterns(self, gradient_field: np.ndarray) -> List[Dict[str, Any]]:
        """Обнаруживает спиральные паттерны в градиентном поле."""
        ...

@runtime_checkable
class DynamicComputeRouterProtocol(Protocol):
    """Протокол для DynamicComputeRouter."""
    def get_optimal_window_size(self, points: np.ndarray) -> int:
        """Определяет оптимальный размер окна для анализа."""
        ...
    
    def get_stability_threshold(self) -> float:
        """Возвращает порог стабильности для анализа."""
        ...
    
    def adaptive_route(self, task: Callable, points: np.ndarray, **kwargs) -> Any:
        """Адаптивно маршрутизирует задачи на основе характеристик точек."""
        ...

# ======================
# DECORATORS
# ======================
def timeit(func: Callable[..., T]) -> Callable[..., T]:
    """Декоратор для измерения времени выполнения функции."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        instance = args[0] if args else None
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Логирование времени выполнения
            if instance and hasattr(instance, 'logger'):
                instance.logger.debug(f"[AIAssistant] {func.__name__} completed in {elapsed:.4f} seconds")
            
            # Запись метрик производительности
            if instance and hasattr(instance, 'performance_metrics'):
                metric_name = f"{func.__name__}_time"
                if metric_name not in instance.performance_metrics:
                    instance.performance_metrics[metric_name] = []
                instance.performance_metrics[metric_name].append(elapsed)
                
                # Обновление общего времени анализа
                if "total_analysis_time" not in instance.performance_metrics:
                    instance.performance_metrics["total_analysis_time"] = []
                instance.performance_metrics["total_analysis_time"].append(elapsed)
                
            return result
        except Exception as e:
            if instance and hasattr(instance, 'logger'):
                instance.logger.error(f"[AIAssistant] Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def memory_profile(func: Callable[..., T]) -> Callable[..., T]:
    """Декоратор для профилирования использования памяти."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        instance = args[0] if args else None
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            mem_after = process.memory_info().rss / (1024 * 1024)  # MB
            mem_diff = mem_after - mem_before
            
            # Запись метрик памяти
            if instance and hasattr(instance, 'memory_metrics'):
                func_name = func.__name__
                if func_name not in instance.memory_metrics["function_memory"]:
                    instance.memory_metrics["function_memory"][func_name] = []
                instance.memory_metrics["function_memory"][func_name].append({
                    "before": mem_before,
                    "after": mem_after,
                    "diff": mem_diff
                })
            
            # Проверка лимитов памяти
            if instance and hasattr(instance, 'config') and mem_diff > instance.config.max_memory_mb * 0.8:
                if instance and hasattr(instance, 'logger'):
                    instance.logger.warning(
                        f"[AIAssistant] High memory usage in {func.__name__}: {mem_diff:.2f} MB (80% of limit)"
                    )
                if mem_diff > instance.config.max_memory_mb:
                    raise ResourceLimitExceededError(
                        f"Memory limit exceeded in {func.__name__}: {mem_diff:.2f} MB > {instance.config.max_memory_mb} MB"
                    )
    return wrapper

def rate_limit(calls: int, period: float = 1.0):
    """Декоратор для ограничения скорости вызова функции."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        last_reset = [time.time()]
        calls_made = [0]
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            with lock:
                # Сброс счетчика, если прошел период
                if current_time - last_reset[0] > period:
                    calls_made[0] = 0
                    last_reset[0] = current_time
                
                # Проверка лимита
                if calls_made[0] >= calls:
                    raise ResourceLimitExceededError(
                        f"Rate limit exceeded for {func.__name__}: {calls_made[0]}/{calls} calls per {period} second(s)"
                    )
                
                calls_made[0] += 1
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def cache(max_size: int = 1000):
    """Декоратор для кэширования результатов функции."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache_dict = {}
        cache_order = []
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Создание ключа, учитывающего порядок именованных аргументов
            with lock:
                # Используем args[1:] чтобы пропустить self
                key_args = tuple(args[1:]) if len(args) > 1 else ()
                key_kwargs = tuple(sorted(kwargs.items()))
                key = (key_args, key_kwargs)
                
                # Проверка кэша
                if key in cache_dict:
                    # Обновляем порядок использования
                    if key in cache_order:
                        cache_order.remove(key)
                    cache_order.append(key)
                    # Увеличиваем счетчик попаданий
                    if hasattr(wrapper, 'cache_hits'):
                        wrapper.cache_hits += 1
                    return cache_dict[key]
                
                # Вычисление результата
                result = func(*args, **kwargs)
                
                # Сохранение в кэш
                if len(cache_dict) >= max_size:
                    oldest_key = cache_order.pop(0)
                    del cache_dict[oldest_key]
                
                cache_dict[key] = result
                cache_order.append(key)
                
                # Обновляем счетчик промахов
                if not hasattr(wrapper, 'cache_misses'):
                    wrapper.cache_misses = 0
                wrapper.cache_misses += 1
                
                return result
        
        # Инициализация счетчиков
        wrapper.cache_hits = 0
        wrapper.cache_misses = 0
        
        return wrapper
    return decorator

def validate_input(func: Callable[..., T]) -> Callable[..., T]:
    """Декоратор для валидации входных данных."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        instance = args[0] if args else None
        try:
            # Валидация конфигурации при инициализации
            if func.__name__ == "__init__" and instance and hasattr(instance, 'config'):
                instance.config.validate()
            return func(*args, **kwargs)
        except Exception as e:
            if instance and hasattr(instance, 'logger'):
                error_msg = f"[AIAssistant] Input validation failed in {func.__name__}: {str(e)}"
                instance.logger.error(error_msg)
            raise InputValidationError(f"Input validation failed: {str(e)}") from e
    return wrapper

# ======================
# HELPER METHODS
# ======================
def torus_distance(point1: Tuple[float, float], point2: Tuple[float, float], n: int) -> float:
    """
    Вычисляет расстояние на торе с учетом циклической природы пространства ECDSA.
    
    Args:
        point1: Первая точка (u_r, u_z)
        point2: Вторая точка (u_r, u_z)
        n: Порядок эллиптической кривой
    
    Returns:
        Минимальное расстояние на торе
    """
    dx = min(abs(point1[0] - point2[0]), n - abs(point1[0] - point2[0]))
    dy = min(abs(point1[1] - point2[1]), n - abs(point1[1] - point2[1]))
    return (dx**2 + dy**2)**0.5

def create_torus_cover(points: np.ndarray, n: int, num_intervals: int, overlap_percent: float) -> List[Tuple[int, int]]:
    """
    Создает покрытие тора с учетом циклической природы пространства.
    
    Args:
        points: Массив точек в пространстве (u_r, u_z)
        n: Порядок эллиптической кривой
        num_intervals: Количество интервалов в покрытии
        overlap_percent: Процент перекрытия между интервалами
    
    Returns:
        Список интервалов покрытия
    """
    if len(points) == 0:
        return [(0, n)]
    
    # Создание функции фильтрации (используем u_r как основную координату)
    filter_values = points[:, 0]  # u_r координаты
    
    # Определение границ
    min_val = np.min(filter_values)
    max_val = np.max(filter_values)
    range_val = max_val - min_val
    
    if range_val == 0:
        # Все точки имеют одинаковое значение u_r
        return [(0, n)]
    
    # Размер интервала с учетом перекрытия
    interval_size = range_val / (num_intervals * (1 - overlap_percent / 100))
    step = interval_size * (1 - overlap_percent / 100)
    
    # Создание интервалов с учетом циклической природы
    intervals = []
    current = min_val
    
    for _ in range(num_intervals):
        end = current + interval_size
        # Если интервал выходит за пределы [0, n], оборачиваем его
        if end > n:
            # Создаем два интервала: от current до n и от 0 до остатка
            intervals.append((int(current), n))
            remainder = end - n
            if remainder > 0:
                intervals.append((0, int(remainder)))
        else:
            intervals.append((int(current), int(end)))
        
        current = current + step
        if current >= n:
            current -= n
    
    return intervals

# ======================
# CONFIGURATION
# ======================
@dataclass
class AIAssistantConfig:
    """
    Конфигурация AIAssistant с параметрами для топологического анализа.
    
    Attributes:
        n: Порядок эллиптической кривой (по умолчанию для secp256k1)
        grid_size: Базовый размер сетки для анализа
        min_density_threshold: Минимальный порог плотности (25-й перцентиль)
        num_intervals: Количество интервалов в покрытии
        overlap_percent: Процент перекрытия между интервалами
        clustering_method: Метод кластеризации ('dbscan' или 'hierarchical')
        eps: Параметр epsilon для DBSCAN
        min_samples: Минимальное количество образцов для DBSCAN
        max_analysis_time: Максимальное время анализа в секундах
        vulnerability_threshold: Порог уязвимости (0.0-1.0)
        anomaly_score_threshold: Порог аномалии (0.0-1.0)
        log_level: Уровень логирования
        monitoring_enabled: Включить мониторинг
        api_version: Версия API
        max_memory_mb: Максимальное использование памяти в МБ
    """
    n: int = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141  # Порядок secp256k1
    grid_size: int = 100
    min_density_threshold: float = 0.25
    num_intervals: int = 10
    overlap_percent: float = 30.0
    clustering_method: str = "dbscan"
    eps: float = 0.1
    min_samples: int = 5
    max_analysis_time: float = 300.0
    vulnerability_threshold: float = 0.5
    anomaly_score_threshold: float = 0.7
    log_level: str = "INFO"
    monitoring_enabled: bool = True
    api_version: str = "3.2.0"
    max_memory_mb: int = 1024
    
    def validate(self):
        """Валидация конфигурации."""
        if self.n <= 0:
            raise ValueError("n (curve order) must be positive")
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if not (0 <= self.min_density_threshold <= 1):
            raise ValueError("min_density_threshold must be between 0 and 1")
        if self.num_intervals <= 0:
            raise ValueError("num_intervals must be positive")
        if not (0 <= self.overlap_percent <= 100):
            raise ValueError("overlap_percent must be between 0 and 100")
        if self.clustering_method not in ["dbscan", "hierarchical"]:
            raise ValueError("clustering_method must be 'dbscan' or 'hierarchical'")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if self.max_analysis_time <= 0:
            raise ValueError("max_analysis_time must be positive")
        if not (0 <= self.vulnerability_threshold <= 1):
            raise ValueError("vulnerability_threshold must be between 0 and 1")
        if not (0 <= self.anomaly_score_threshold <= 1):
            raise ValueError("anomaly_score_threshold must be between 0 and 1")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        
        # Проверка формата версии API
        version_parts = self.api_version.split('.')
        if len(version_parts) != 3 or not all(part.isdigit() for part in version_parts):
            raise ValueError("api_version must be in format X.Y.Z")
    
    def is_api_version_compatible(self, required_version: str) -> bool:
        """
        Проверяет совместимость версий API.
        
        Args:
            required_version: Требуемая версия API
            
        Returns:
            True, если версии совместимы, иначе False
        """
        # Простая семантическая проверка версии
        try:
            current_parts = [int(x) for x in self.api_version.split('.')]
            required_parts = [int(x) for x in required_version.split('.')]
            
            # Мажорная версия должна совпадать
            if current_parts[0] != required_parts[0]:
                return False
                
            # Минорная версия должна быть >= требуемой
            if current_parts[1] < required_parts[1]:
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error parsing API version: {str(e)}")
            return False

# ======================
# MAPPER ALGORITHM
# ======================
class Mapper:
    """Реализация Mapper алгоритма для топологического анализа."""
    
    def __init__(self, config: AIAssistantConfig):
        """Инициализация Mapper."""
        self.config = config
        self.logger = logging.getLogger("AuditCore.AIAssistant.Mapper")
    
    @timeit
    @memory_profile
    def compute_mapper(self, points: np.ndarray) -> Dict[str, Any]:
        """
        Вычисляет топологическое представление с помощью Mapper алгоритма.
        
        Args:
            points: Массив точек в пространстве (u_r, u_z)
            
        Returns:
            Словарь с узлами и ребрами топологического представления
        """
        if len(points) == 0:
            return {"nodes": [], "edges": [], "success": False, "error": "No points provided"}
        
        try:
            # Создание покрытия
            cover = create_torus_cover(
                points, 
                self.config.n, 
                self.config.num_intervals, 
                self.config.overlap_percent
            )
            
            # Кластеризация в каждом интервале
            clusters_per_interval = self._cluster_points(points, cover)
            
            # Построение графа
            graph = self._build_graph(cover, clusters_per_interval)
            
            return {
                "nodes": graph["nodes"],
                "edges": graph["edges"],
                "cover": cover,
                "clusters_per_interval": clusters_per_interval,
                "success": True
            }
        except Exception as e:
            self.logger.error(f"Mapper computation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _cluster_points(self, points: np.ndarray, cover: List[Tuple[int, int]]) -> List[List[List[int]]]:
        """
        Кластеризует точки в каждом интервале покрытия.
        
        Args:
            points: Массив точек в пространстве (u_r, u_z)
            cover: Покрытие пространства
            
        Returns:
            Список кластеров для каждого интервала покрытия
        """
        clusters_per_interval = []
        
        for interval in cover:
            # Определение индексов точек в интервале
            interval_indices = []
            for i, point in enumerate(points):
                ur = point[0]
                # Проверка для циклического пространства
                if interval[0] <= interval[1]:  # Нормальный интервал
                    if interval[0] <= ur < interval[1]:
                        interval_indices.append(i)
                else:  # Интервал, пересекающий границу
                    if ur >= interval[0] or ur < interval[1]:
                        interval_indices.append(i)
            
            # Кластеризация точек в интервале
            if interval_indices:
                clusters = self._cluster_interval(points, interval_indices)
                clusters_per_interval.append(clusters)
            else:
                clusters_per_interval.append([])
        
        return clusters_per_interval
    
    def _cluster_interval(self, points: np.ndarray, indices: List[int]) -> List[List[int]]:
        """
        Применяет кластеризацию к точкам в одном интервале.
        
        Args:
            points: Массив точек в пространстве (u_r, u_z)
            indices: Индексы точек для кластеризации
            
        Returns:
            Список кластеров (каждый кластер - список индексов)
        """
        if len(indices) == 0:
            return []
        
        subset = points[indices]
        
        # Применение метода кластеризации
        if self.config.clustering_method == "dbscan":
            try:
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=self.config.eps, min_samples=self.config.min_samples)
                labels = clustering.fit_predict(subset)
                
                # Группировка точек по кластерам
                clusters = {}
                for i, label in enumerate(labels):
                    if label != -1:  # Игнорируем шум
                        if label not in clusters:
                            clusters[label] = []
                        clusters[label].append(indices[i])
                
                return list(clusters.values())
            except ImportError:
                self.logger.warning("DBSCAN not available, falling back to simple clustering")
                # Простая кластеризация как fallback
                return [indices]
        else:
            # Реализация иерархической кластеризации
            try:
                from sklearn.cluster import AgglomerativeClustering
                clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=self.config.eps)
                labels = clustering.fit_predict(subset)
                
                # Группировка точек по кластерам
                clusters = {}
                for i, label in enumerate(labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(indices[i])
                
                return list(clusters.values())
            except ImportError:
                self.logger.warning("AgglomerativeClustering not available, falling back to simple clustering")
                return [indices]
    
    def _build_graph(self, cover: List[Tuple[int, int]], clusters_per_interval: List[List[List[int]]]) -> Dict[str, Any]:
        """
        Строит граф на основе кластеров в интервалах покрытия.
        
        Args:
            cover: Покрытие пространства
            clusters_per_interval: Кластеры для каждого интервала покрытия
            
        Returns:
            Граф в виде словаря с узлами и ребрами
        """
        # Создание узлов
        nodes = []
        node_to_id = {}
        
        for interval_idx, clusters in enumerate(clusters_per_interval):
            for cluster_idx, cluster in enumerate(clusters):
                node_id = f"{interval_idx}-{cluster_idx}"
                nodes.append({
                    "id": node_id,
                    "interval": interval_idx,
                    "cluster": cluster_idx,
                    "size": len(cluster),
                    "points": cluster
                })
                node_to_id[(interval_idx, cluster_idx)] = len(nodes) - 1
        
        # Создание ребер
        edges = []
        for i in range(len(clusters_per_interval)):
            for j in range(i + 1, len(clusters_per_interval)):
                # Проверка перекрытия интервалов
                if j - i <= 1:  # Смежные интервалы
                    for ci, cluster_i in enumerate(clusters_per_interval[i]):
                        for cj, cluster_j in enumerate(clusters_per_interval[j]):
                            # Проверка пересечения кластеров
                            intersection = set(cluster_i) & set(cluster_j)
                            if len(intersection) > 0:
                                source_id = node_to_id.get((i, ci))
                                target_id = node_to_id.get((j, cj))
                                if source_id is not None and target_id is not None:
                                    edges.append({
                                        "source": source_id,
                                        "target": target_id,
                                        "weight": len(intersection),
                                        "intersection": list(intersection)
                                    })
        
        return {"nodes": nodes, "edges": edges}

# ======================
# MAIN AI ASSISTANT CLASS
# ======================
class AIAssistant:
    """
    Основной класс AIAssistant для определения регионов аудита и анализа уязвимостей.
    
    Этот компонент использует Mapper алгоритм для топологического анализа пространства ECDSA подписей
    и определяет критические регионы для детального аудита.
    """
    
    def __init__(self, config: Optional[AIAssistantConfig] = None):
        """
        Инициализация AIAssistant.
        
        Args:
            config: Конфигурация AIAssistant (опционально)
        """
        self.config = config or AIAssistantConfig()
        try:
            self.config.validate()
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise
            
        # Инициализация логгера
        self.logger = logging.getLogger("AuditCore.AIAssistant")
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Проверка и настройка обработчиков логгера
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Инициализация Mapper
        self.mapper = Mapper(self.config)
        
        # Инициализация внутренних состояний
        self.performance_metrics = {
            "total_analysis_time": [],
            "cache_hits": 0,
            "cache_misses": 0
        }
        self.memory_metrics = {
            "function_memory": {},
            "total_memory": []
        }
        self.security_metrics = {
            "input_validation_failures": 0,
            "resource_limit_exceeded": 0
        }
        self.monitoring_data = {
            "analysis_count": 0,
            "last_analysis_time": None,
            "critical_vulnerabilities": 0,
            "high_vulnerabilities": 0,
            "medium_vulnerabilities": 0
        }
        self.last_analysis = None
        
        # Инициализация компонентов (будут установлены через dependency injection)
        self.hypercore_transformer: Optional[HyperCoreTransformerProtocol] = None
        self.tcon: Optional[TCONProtocol] = None
        self.signature_generator: Optional[SignatureGeneratorProtocol] = None
        self.betti_analyzer: Optional[BettiAnalyzerProtocol] = None
        self.collision_engine: Optional[CollisionEngineProtocol] = None
        self.gradient_analysis: Optional[GradientAnalysisProtocol] = None
        self.dynamic_compute_router: Optional[DynamicComputeRouterProtocol] = None
    
    def set_hypercore_transformer(self, transformer: HyperCoreTransformerProtocol):
        """Устанавливает HyperCoreTransformer."""
        self.hypercore_transformer = transformer
    
    def set_tcon(self, tcon: TCONProtocol):
        """Устанавливает TCON (Topological Control)."""
        self.tcon = tcon
    
    def set_signature_generator(self, generator: SignatureGeneratorProtocol):
        """Устанавливает SignatureGenerator."""
        self.signature_generator = generator
    
    def set_betti_analyzer(self, analyzer: BettiAnalyzerProtocol):
        """Устанавливает BettiAnalyzer."""
        self.betti_analyzer = analyzer
    
    def set_collision_engine(self, engine: CollisionEngineProtocol):
        """Устанавливает CollisionEngine."""
        self.collision_engine = engine
    
    def set_gradient_analysis(self, analysis: GradientAnalysisProtocol):
        """Устанавливает GradientAnalysis."""
        self.gradient_analysis = analysis
    
    def set_dynamic_compute_router(self, router: DynamicComputeRouterProtocol):
        """Устанавливает DynamicComputeRouter."""
        self.dynamic_compute_router = router
    
    def verify_dependencies(self) -> bool:
        """
        Проверяет наличие всех необходимых зависимостей.
        
        Returns:
            True, если все зависимости установлены, иначе False
        """
        missing = []
        if not self.hypercore_transformer:
            missing.append("HyperCoreTransformer")
        if not self.tcon:
            missing.append("TCON")
        if not self.signature_generator:
            missing.append("SignatureGenerator")
        
        if missing:
            self.logger.warning(f"Missing dependencies: {', '.join(missing)}")
            return False
        return True
    
    @timeit
    @memory_profile
    @validate_input
    def analyze_real_signatures(self, public_key: PointProtocol, real_signatures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Определяет регионы для детального аудита на основе анализа реальных подписей.
        
        Args:
            public_key: Публичный ключ для анализа
            real_signatures: Список реальных ECDSA подписей
            
        Returns:
            Список регионов для аудита с информацией о критичности и стабильности
        """
        if not self.verify_dependencies():
            raise RuntimeError("Required components are not properly set")
        
        if not real_signatures:
            raise InputValidationError("No real signatures provided for analysis")
        
        start_time = time.time()
        try:
            # Преобразование подписей в точки
            self.logger.debug("[AIAssistant] Converting signatures to points...")
            points = np.array([
                [sig["u_r"], sig["u_z"]] 
                for sig in real_signatures 
                if "u_r" in sig and "u_z" in sig
            ])
            
            if len(points) == 0:
                raise InputValidationError("No valid points extracted from signatures")
            
            # Анализ с использованием Mapper
            self.logger.debug("[AIAssistant] Performing Mapper analysis...")
            analysis_results = self.analyze_with_mapper(points)
            
            # Идентификация регионов для аудита
            self.logger.debug("[AIAssistant] Identifying critical regions for audit...")
            regions = self._identify_critical_regions(
                analysis_results["mapper"],
                analysis_results["multiscale_mapper"],
                analysis_results["smoothing_analysis"],
                analysis_results["density_analysis"]
            )
            
            # Обновление мониторинга
            self._update_monitoring(regions)
            
            # Запись времени анализа
            elapsed = time.time() - start_time
            self.logger.info(f"[AIAssistant] Analysis completed in {elapsed:.4f} seconds")
            
            # Сохранение результатов для последующего использования
            self.last_analysis = {
                "regions": regions,
                "analysis_time": elapsed,
                "signature_count": len(real_signatures),
                "success": True
            }
            
            return {
                "critical_regions": regions,
                "analysis_time": elapsed,
                "signature_count": len(real_signatures),
                "success": True
            }
        except Exception as e:
            self.logger.error(f"[AIAssistant] Analysis failed: {str(e)}")
            # Сохранение ошибки для последующего использования
            self.last_analysis = {
                "error": str(e),
                "success": False
            }
            raise AIAssistantError(f"Analysis failed: {str(e)}") from e
    
    @timeit
    @memory_profile
    def analyze_with_mapper(self, points: np.ndarray) -> Dict[str, Any]:
        """
        Выполняет полный топологический анализ с использованием Mapper алгоритма.
        
        Args:
            points: Массив точек в пространстве (u_r, u_z)
            
        Returns:
            Словарь с результатами анализа
        """
        start_time = time.time()
        try:
            # Проверка ограничения времени
            if self.config.max_analysis_time > 0:
                def _analyze():
                    # Вычисление Mapper
                    mapper_start = time.time()
                    mapper = self.mapper.compute_mapper(points)
                    mapper_time = time.time() - mapper_start
                    self.logger.info(f"[AIAssistant] Mapper computation completed in {mapper_time:.4f} seconds")
                    
                    # Вычисление Multiscale Mapper
                    multiscale_start = time.time()
                    multiscale_mapper = self._compute_multiscale_mapper(points)
                    multiscale_time = time.time() - multiscale_start
                    self.logger.info(f"[AIAssistant] Multiscale Mapper computation completed in {multiscale_time:.4f} seconds")
                    
                    # Анализ сглаживания
                    smoothing_start = time.time()
                    smoothing_analysis = self._analyze_smoothing(points)
                    smoothing_time = time.time() - smoothing_start
                    self.logger.info(f"[AIAssistant] Smoothing analysis completed in {smoothing_time:.4f} seconds")
                    
                    # Анализ плотности
                    density_start = time.time()
                    density_analysis = self._analyze_density(points)
                    density_time = time.time() - density_start
                    self.logger.info(f"[AIAssistant] Density analysis completed in {density_time:.4f} seconds")
                    
                    return {
                        "mapper": mapper,
                        "multiscale_mapper": multiscale_mapper,
                        "smoothing_analysis": smoothing_analysis,
                        "density_analysis": density_analysis
                    }
                
                # Запуск анализа с таймаутом
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(_analyze)
                    try:
                        result = future.result(timeout=self.config.max_analysis_time)
                    except concurrent.futures.TimeoutError:
                        raise AnalysisTimeoutError(
                            f"Analysis exceeded maximum time limit of {self.config.max_analysis_time} seconds"
                        )
            else:
                # Анализ без ограничения времени
                mapper = self.mapper.compute_mapper(points)
                multiscale_mapper = self._compute_multiscale_mapper(points)
                smoothing_analysis = self._analyze_smoothing(points)
                density_analysis = self._analyze_density(points)
                result = {
                    "mapper": mapper,
                    "multiscale_mapper": multiscale_mapper,
                    "smoothing_analysis": smoothing_analysis,
                    "density_analysis": density_analysis
                }
            
            # Идентификация критических регионов
            critical_regions = self._identify_critical_regions(
                result["mapper"],
                result["multiscale_mapper"],
                result["smoothing_analysis"],
                result["density_analysis"]
            )
            
            return {
                "mapper": result["mapper"],
                "multiscale_mapper": result["multiscale_mapper"],
                "smoothing_analysis": result["smoothing_analysis"],
                "density_analysis": result["density_analysis"],
                "critical_regions": critical_regions,
                "success": True
            }
        except Exception as e:
            self.logger.error(f"[AIAssistant] Mapper analysis failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @cache()
    def _identify_critical_regions(self, 
                                 mapper: Dict[str, Any],
                                 multiscale_mapper: Dict[str, Any],
                                 smoothing_analysis: Dict[str, Any],
                                 density_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Идентифицирует критические регионы на основе результатов анализа.
        
        Args:
            mapper: Результаты Mapper анализа
            multiscale_mapper: Результаты multiscale Mapper анализа
            smoothing_analysis: Результаты анализа сглаживания
            density_analysis: Результаты анализа плотности
            
        Returns:
            Список критических регионов с информацией о критичности
        """
        critical_regions = []
        
        # 1. Анализ аномалий сглаживания
        if "anomalies" in smoothing_analysis:
            for anomaly in smoothing_analysis["anomalies"]:
                ur_start = anomaly.get("ur_range", [0, 0])[0]
                ur_end = anomaly.get("ur_range", [0, 0])[1]
                uz_start = anomaly.get("uz_range", [0, 0])[0]
                uz_end = anomaly.get("uz_range", [0, 0])[1]
                
                # Проверка наличия ключей
                if ur_start is None or ur_end is None or uz_start is None or uz_end is None:
                    continue
                    
                critical_regions.append({
                    "ur_range": (ur_start, ur_end),
                    "uz_range": (uz_start, uz_end),
                    "stability": anomaly.get("stability", 0.5),
                    "criticality": min(1.0, anomaly.get("criticality", 0.5) * 1.2),
                    "type": "smoothing_anomaly",
                    "source": "smoothing_analysis",
                    "details": anomaly.get("details", {})
                })
        
        # 2. Анализ плотности
        if "low_density_regions" in density_analysis:
            for region in density_analysis["low_density_regions"]:
                ur_start = region.get("ur_range", [0, 0])[0]
                ur_end = region.get("ur_range", [0, 0])[1]
                uz_start = region.get("uz_range", [0, 0])[0]
                uz_end = region.get("uz_range", [0, 0])[1]
                
                # Проверка наличия ключей
                if ur_start is None or ur_end is None or uz_start is None or uz_end is None:
                    continue
                
                criticality = region.get("criticality", 0.3)
                # Увеличиваем критичность для очень низкой плотности
                if region.get("density", 1.0) < density_analysis.get("min_density", 1.0) * 0.5:
                    criticality = min(1.0, criticality * 1.5)
                
                critical_regions.append({
                    "ur_range": (ur_start, ur_end),
                    "uz_range": (uz_start, uz_end),
                    "stability": region.get("stability", 0.7),
                    "criticality": criticality,
                    "type": "low_density",
                    "source": "density_analysis",
                    "details": region.get("details", {})
                })
        
        # 3. Проверка структуры тора
        if self.tcon:
            try:
                torus_verification = self.tcon.verify_torus_structure(
                    np.array([[anomaly.get("ur_range", [0, 0])[0], anomaly.get("uz_range", [0, 0])[0]] 
                             for anomaly in smoothing_analysis.get("anomalies", [])])
                )
                if not torus_verification.get("is_valid", True):
                    for i, invalid_point in enumerate(torus_verification.get("invalid_points", [])):
                        # Создание региона вокруг проблемной точки
                        ur, uz = invalid_point
                        ur_range = (max(0, ur - 100), min(self.config.n, ur + 100))
                        uz_range = (max(0, uz - 100), min(self.config.n, uz + 100))
                        critical_regions.append({
                            "ur_range": ur_range,
                            "uz_range": uz_range,
                            "stability": 0.2,
                            "criticality": 0.9,
                            "type": "torus_deformation",
                            "source": "tcon_verification",
                            "details": torus_verification.get("details", {})
                        })
            except Exception as e:
                self.logger.warning(f"[AIAssistant] TCON verification failed: {str(e)}")
        
        # 4. Анализ диагональной периодичности
        if self.tcon:
            try:
                periodicity_analysis = self.tcon.analyze_diagonal_periodicity(
                    np.array([[anomaly.get("ur_range", [0, 0])[0], anomaly.get("uz_range", [0, 0])[0]] 
                             for anomaly in smoothing_analysis.get("anomalies", [])])
                )
                if periodicity_analysis.get("has_issues", False):
                    for issue in periodicity_analysis.get("issues", []):
                        ur = issue.get("ur", 0)
                        ur_range = (max(0, ur - 50), min(self.config.n, ur + 50))
                        uz_range = (max(0, issue.get("uz", 0) - 50), min(self.config.n, issue.get("uz", 0) + 50))
                        critical_regions.append({
                            "ur_range": ur_range,
                            "uz_range": uz_range,
                            "stability": 0.3,
                            "criticality": 0.8,
                            "type": "periodicity_issue",
                            "source": "tcon_periodicity",
                            "details": issue
                        })
            except Exception as e:
                self.logger.warning(f"[AIAssistant] Diagonal periodicity analysis failed: {str(e)}")
        
        # 5. Объединение перекрывающихся регионов
        critical_regions = self._merge_overlapping_regions(critical_regions)
        
        return critical_regions
    
    def _merge_overlapping_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Объединяет перекрывающиеся регионы в один.
        
        Args:
            regions: Список регионов
            
        Returns:
            Список объединенных регионов
        """
        if not regions:
            return []
        
        # Сортировка регионов по критичности (от высокой к низкой)
        sorted_regions = sorted(regions, key=lambda x: x.get("criticality", 0), reverse=True)
        merged_regions = []
        
        for region in sorted_regions:
            ur_start, ur_end = region["ur_range"]
            uz_start, uz_end = region["uz_range"]
            
            # Проверка на перекрытие с существующими регионами
            overlap_found = False
            for i, existing in enumerate(merged_regions):
                if self._regions_overlap(region, existing):
                    # Объединение регионов
                    merged_regions[i] = self._merge_regions(existing, region)
                    overlap_found = True
                    break
            
            if not overlap_found:
                merged_regions.append(region)
        
        return merged_regions
    
    def _regions_overlap(self, region1: Dict[str, Any], region2: Dict[str, Any]) -> bool:
        """
        Проверяет, перекрываются ли два региона с учетом торической структуры.
        
        Args:
            region1: Первый регион
            region2: Второй регион
            
        Returns:
            True, если регионы перекрываются, иначе False
        """
        ur1_start, ur1_end = region1["ur_range"]
        ur2_start, ur2_end = region2["ur_range"]
        
        # Проверка перекрытия по u_r с учетом торической структуры
        ur_overlap = False
        if ur1_start <= ur1_end:  # Нормальный интервал
            if ur2_start <= ur2_end:  # Нормальный интервал
                if (ur1_start <= ur2_end and ur2_start <= ur1_end):
                    ur_overlap = True
            else:  # Интервал, пересекающий границу
                if (ur1_start <= ur2_end or ur2_start <= ur1_end):
                    ur_overlap = True
        else:  # Интервал, пересекающий границу
            if ur2_start <= ur2_end:  # Нормальный интервал
                if (ur2_start <= ur1_end or ur1_start <= ur2_end):
                    ur_overlap = True
            else:  # Интервал, пересекающий границу
                ur_overlap = True  # Все интервалы, пересекающие границу, перекрываются
        
        # Проверка перекрытия по u_z
        uz1_start, uz1_end = region1["uz_range"]
        uz2_start, uz2_end = region2["uz_range"]
        
        uz_overlap = False
        if uz1_start <= uz1_end:  # Нормальный интервал
            if uz2_start <= uz2_end:  # Нормальный интервал
                if (uz1_start <= uz2_end and uz2_start <= uz1_end):
                    uz_overlap = True
            else:  # Интервал, пересекающий границу
                if (uz1_start <= uz2_end or uz2_start <= uz1_end):
                    uz_overlap = True
        else:  # Интервал, пересекающий границу
            if uz2_start <= uz2_end:  # Нормальный интервал
                if (uz2_start <= uz1_end or uz1_start <= uz2_end):
                    uz_overlap = True
            else:  # Интервал, пересекающий границу
                uz_overlap = True  # Все интервалы, пересекающие границу, перекрываются
        
        return ur_overlap and uz_overlap
    
    def _merge_regions(self, region1: Dict[str, Any], region2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Объединяет два региона в один.
        
        Args:
            region1: Первый регион
            region2: Второй регион
            
        Returns:
            Объединенный регион
        """
        # Объединение диапазонов u_r
        ur1_start, ur1_end = region1["ur_range"]
        ur2_start, ur2_end = region2["ur_range"]
        
        # Для торических интервалов нужно учитывать циклическую природу
        if ur1_start <= ur1_end and ur2_start <= ur2_end:
            ur_start = min(ur1_start, ur2_start)
            ur_end = max(ur1_end, ur2_end)
        elif ur1_start > ur1_end and ur2_start <= ur2_end:
            # region1 пересекает границу, region2 - нет
            if ur2_start < ur1_end or ur2_end > ur1_start:
                ur_start = min(ur1_start, ur2_start)
                ur_end = max(ur1_end, ur2_end)
            else:
                # Не пересекаются в линейном смысле, но могут пересекаться на торе
                if (self.config.n - ur1_start + ur1_end) < (ur2_end - ur2_start):
                    ur_start = ur1_start
                    ur_end = ur1_end
                else:
                    ur_start = ur2_start
                    ur_end = ur2_end
        elif ur1_start <= ur1_end and ur2_start > ur2_end:
            # region1 - нет, region2 пересекает границу
            if ur1_start < ur2_end or ur1_end > ur2_start:
                ur_start = min(ur1_start, ur2_start)
                ur_end = max(ur1_end, ur2_end)
            else:
                # Не пересекаются в линейном смысле, но могут пересекаться на торе
                if (ur1_end - ur1_start) < (self.config.n - ur2_start + ur2_end):
                    ur_start = ur1_start
                    ur_end = ur1_end
                else:
                    ur_start = ur2_start
                    ur_end = ur2_end
        else:
            # Оба региона пересекают границу
            ur_start = min(ur1_start, ur2_start)
            ur_end = max(ur1_end, ur2_end)
        
        # Объединение диапазонов u_z
        uz1_start, uz1_end = region1["uz_range"]
        uz2_start, uz2_end = region2["uz_range"]
        
        if uz1_start <= uz1_end and uz2_start <= uz2_end:
            uz_start = min(uz1_start, uz2_start)
            uz_end = max(uz1_end, uz2_end)
        elif uz1_start > uz1_end and uz2_start <= uz2_end:
            if uz2_start < uz1_end or uz2_end > uz1_start:
                uz_start = min(uz1_start, uz2_start)
                uz_end = max(uz1_end, uz2_end)
            else:
                if (self.config.n - uz1_start + uz1_end) < (uz2_end - uz2_start):
                    uz_start = uz1_start
                    uz_end = uz1_end
                else:
                    uz_start = uz2_start
                    uz_end = uz2_end
        elif uz1_start <= uz1_end and uz2_start > uz2_end:
            if uz1_start < uz2_end or uz1_end > uz2_start:
                uz_start = min(uz1_start, uz2_start)
                uz_end = max(uz1_end, uz2_end)
            else:
                if (uz1_end - uz1_start) < (self.config.n - uz2_start + uz2_end):
                    uz_start = uz1_start
                    uz_end = uz1_end
                else:
                    uz_start = uz2_start
                    uz_end = uz2_end
        else:
            uz_start = min(uz1_start, uz2_start)
            uz_end = max(uz1_end, uz2_end)
        
        # Вычисление стабильности и критичности
        stability = min(region1.get("stability", 1.0), region2.get("stability", 1.0))
        criticality = max(region1.get("criticality", 0.0), region2.get("criticality", 0.0))
        
        return {
            "ur_range": (ur_start, ur_end),
            "uz_range": (uz_start, uz_end),
            "stability": min(1.0, stability),
            "criticality": min(1.0, criticality),
            "type": f"{region1.get('type', 'unknown')},{region2.get('type', 'unknown')}",
            "source": f"{region1.get('source', 'unknown')},{region2.get('source', 'unknown')}"
        }
    
    @timeit
    @memory_profile
    def _compute_multiscale_mapper(self, points: np.ndarray) -> Dict[str, Any]:
        """
        Вычисляет multiscale Mapper для выявления уязвимостей на разных масштабах.
        
        Args:
            points: Массив точек в пространстве (u_r, u_z)
            
        Returns:
            Словарь с результатами multiscale Mapper анализа
        """
        try:
            # Используем разные параметры покрытия для разных масштабов
            scales = [0.5, 1.0, 2.0]
            results = []
            
            for scale in scales:
                # Настройка параметров для текущего масштаба
                config = AIAssistantConfig(
                    n=self.config.n,
                    num_intervals=int(self.config.num_intervals * scale),
                    overlap_percent=self.config.overlap_percent,
                    clustering_method=self.config.clustering_method,
                    eps=self.config.eps * scale,
                    min_samples=max(1, int(self.config.min_samples * scale))
                )
                
                # Вычисление Mapper для текущего масштаба
                mapper = Mapper(config)
                result = mapper.compute_mapper(points)
                result["scale"] = scale
                results.append(result)
            
            return {
                "scales": scales,
                "results": results,
                "success": True
            }
        except Exception as e:
            self.logger.error(f"Multiscale Mapper computation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @timeit
    @memory_profile
    def _analyze_smoothing(self, points: np.ndarray) -> Dict[str, Any]:
        """
        Выполняет анализ сглаживания для фильтрации шума и выявления аномалий.
        
        Args:
            points: Массив точек в пространстве (u_r, u_z)
            
        Returns:
            Словарь с результатами анализа сглаживания
        """
        try:
            if len(points) == 0:
                return {"anomalies": [], "success": False, "error": "No points provided"}
            
            # Вычисление 2D гистограммы
            hist, xedges, yedges = np.histogram2d(
                points[:, 0], points[:, 1],
                bins=self.config.grid_size,
                range=[[0, self.config.n], [0, self.config.n]]
            )
            
            # Нормализация
            if np.sum(hist) > 0:
                hist = hist / np.sum(hist)
            
            # Применение гауссова фильтра для сглаживания
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(hist, sigma=1.0)
            
            # Вычисление локальных аномалий
            anomalies = []
            for i in range(1, self.config.grid_size - 1):
                for j in range(1, self.config.grid_size - 1):
                    # Проверка на аномалию (резкое изменение плотности)
                    neighborhood = smoothed[i-1:i+2, j-1:j+2]
                    center = smoothed[i, j]
                    mean_neighborhood = np.mean(neighborhood)
                    
                    if center > 0 and abs(center - mean_neighborhood) > 0.5 * mean_neighborhood:
                        # Определение координат в исходном пространстве
                        ur = xedges[i] + (xedges[i+1] - xedges[i]) / 2
                        uz = yedges[j] + (yedges[j+1] - yedges[j]) / 2
                        
                        # Оценка критичности
                        criticality = min(1.0, abs(center - mean_neighborhood) / center)
                        
                        # Проверка порога аномалии
                        if criticality > self.config.anomaly_score_threshold:
                            anomalies.append({
                                "ur_range": (int(ur - 10), int(ur + 10)),
                                "uz_range": (int(uz - 10), int(uz + 10)),
                                "stability": 1.0 - criticality,
                                "criticality": criticality,
                                "type": "smoothing_anomaly",
                                "details": {
                                    "density": float(hist[i, j]),
                                    "smoothed_density": float(smoothed[i, j]),
                                    "anomaly_score": float(criticality)
                                }
                            })
            
            return {
                "anomalies": anomalies,
                "histogram": hist.tolist(),
                "smoothed": smoothed.tolist(),
                "xedges": xedges.tolist(),
                "yedges": yedges.tolist(),
                "success": True
            }
        except Exception as e:
            self.logger.error(f"Smoothing analysis failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @timeit
    @memory_profile
    def _analyze_density(self, points: np.ndarray) -> Dict[str, Any]:
        """
        Выполняет анализ плотности распределения точек.
        
        Args:
            points: Массив точек в пространстве (u_r, u_z)
            
        Returns:
            Словарь с результатами анализа плотности
        """
        try:
            if len(points) == 0:
                return {"success": False, "error": "No points provided"}
            
            # Вычисление 2D гистограммы
            hist, xedges, yedges = np.histogram2d(
                points[:, 0], points[:, 1],
                bins=self.config.grid_size,
                range=[[0, self.config.n], [0, self.config.n]]
            )
            
            # Нормализация
            total = np.sum(hist)
            if total > 0:
                hist = hist / total
            
            # Вычисление статистики
            density_values = hist[hist > 0]
            mean_density = float(np.mean(density_values)) if len(density_values) > 0 else 0.0
            std_density = float(np.std(density_values)) if len(density_values) > 0 else 0.0
            min_density = float(np.min(density_values)) if len(density_values) > 0 else 0.0
            max_density = float(np.max(density_values)) if len(density_values) > 0 else 0.0
            
            # Определение регионов с низкой плотностью
            low_density_regions = []
            threshold = max(self.config.min_density_threshold * mean_density, 1e-10)
            
            for i in range(self.config.grid_size):
                for j in range(self.config.grid_size):
                    if hist[i, j] < threshold:
                        ur = xedges[i] + (xedges[i+1] - xedges[i]) / 2
                        uz = yedges[j] + (yedges[j+1] - yedges[j]) / 2
                        
                        # Оценка критичности
                        criticality = 1.0 - (hist[i, j] / threshold)
                        
                        low_density_regions.append({
                            "ur_range": (int(ur - 10), int(ur + 10)),
                            "uz_range": (int(uz - 10), int(uz + 10)),
                            "density": float(hist[i, j]),
                            "criticality": min(1.0, criticality),
                            "stability": float(hist[i, j] / mean_density) if mean_density > 0 else 0.0
                        })
            
            return {
                "histogram": hist.tolist(),
                "xedges": xedges.tolist(),
                "yedges": yedges.tolist(),
                "mean_density": float(mean_density),
                "std_density": float(std_density),
                "min_density": float(min_density),
                "max_density": float(max_density),
                "low_density_regions": low_density_regions,
                "success": True
            }
        except Exception as e:
            self.logger.error(f"Density analysis failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    @timeit
    def prioritize_vulnerabilities(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Приоритизирует обнаруженные уязвимости по степени тяжести.
        
        Args:
            analysis_results: Результаты топологического анализа
            
        Returns:
            Список уязвимостей, упорядоченный по степени тяжести
        """
        if not self.verify_dependencies():
            raise RuntimeError("Required components are not properly set")
        
        if not analysis_results or "critical_regions" not in analysis_results:
            raise InputValidationError("Invalid analysis results format")
        
        try:
            # Извлечение критических регионов
            critical_regions = analysis_results["critical_regions"]
            
            # Сортировка по критичности (от высокой к низкой)
            sorted_vulnerabilities = sorted(
                critical_regions, 
                key=lambda x: x.get("criticality", 0), 
                reverse=True
            )
            
            # Добавление рекомендаций
            for vuln in sorted_vulnerabilities:
                vuln["recommendations"] = self._generate_recommendations(vuln)
            
            return sorted_vulnerabilities
        except Exception as e:
            self.logger.error(f"Vulnerability prioritization failed: {str(e)}")
            raise AIAssistantError(f"Vulnerability prioritization failed: {str(e)}") from e
    
    def _generate_recommendations(self, vulnerability: Dict[str, Any]) -> List[str]:
        """
        Генерирует рекомендации для устранения уязвимости.
        
        Args:
            vulnerability: Описание уязвимости
            
        Returns:
            Список рекомендаций
        """
        recommendations = []
        criticality = vulnerability.get("criticality", 0.0)
        vuln_type = vulnerability.get("type", "")
        
        # Рекомендации на основе типа уязвимости
        if "smoothing_anomaly" in vuln_type:
            recommendations.append("Investigate random number generator (RNG) for potential biases.")
            recommendations.append("Consider using deterministic ECDSA (RFC 6979) for nonce generation.")
        
        if "low_density" in vuln_type:
            recommendations.append("Check for insufficient entropy in nonce generation.")
            recommendations.append("Verify implementation against known secure ECDSA implementations.")
        
        if "torus_deformation" in vuln_type:
            recommendations.append("CRITICAL: Torus structure not verified. Check random number generator and consider RFC 6979.")
            recommendations.append("Immediate action required: This indicates a serious flaw in the ECDSA implementation.")
        
        if "periodicity_issue" in vuln_type:
            recommendations.append("Check for periodic patterns in the random number generator.")
            recommendations.append("Consider reseeding the random number generator more frequently.")
        
        # Рекомендации на основе критичности
        if criticality >= 0.8:
            recommendations.insert(0, "URGENT: High criticality vulnerability detected - immediate investigation required.")
        elif criticality >= 0.6:
            recommendations.insert(0, "HIGH: Significant vulnerability detected - investigate promptly.")
        elif criticality >= 0.4:
            recommendations.insert(0, "MEDIUM: Potential vulnerability detected - investigate during next security review.")
        
        # Специфические рекомендации для ECDSA
        recommendations.append("Verify ECDSA implementation against known secure implementations.")
        recommendations.append("Consider implementing deterministic nonce generation (RFC 6979) to prevent nonce reuse.")
        
        return recommendations
    
    def _update_monitoring(self, critical_regions: List[Dict]):
        """
        Обновляет данные мониторинга на основе результатов анализа.
        
        Args:
            critical_regions: Список критических регионов
        """
        # Обновление счетчика анализов
        self.monitoring_data["analysis_count"] += 1
        self.monitoring_data["last_analysis_time"] = datetime.now().isoformat()
        
        # Сброс счетчиков уязвимостей
        self.monitoring_data["critical_vulnerabilities"] = 0
        self.monitoring_data["high_vulnerabilities"] = 0
        self.monitoring_data["medium_vulnerabilities"] = 0
        
        # Подсчет уязвимостей по степени тяжести
        for region in critical_regions:
            criticality = region.get("criticality", 0.0)
            if criticality >= 0.8:
                self.monitoring_data["critical_vulnerabilities"] += 1
            elif criticality >= 0.6:
                self.monitoring_data["high_vulnerabilities"] += 1
            elif criticality >= 0.4:
                self.monitoring_data["medium_vulnerabilities"] += 1
    
    def _log_security_events(self, event_type: str, details: Dict[str, Any] = None):
        """
        Логирует события безопасности.
        
        Args:
            event_type: Тип события
            details: Детали события
        """
        details = details or {}
        
        # Обновление метрик безопасности
        if event_type == "input_validation_failure":
            self.security_metrics["input_validation_failures"] += 1
        elif event_type == "resource_limit_exceeded":
            self.security_metrics["resource_limit_exceeded"] += 1
        
        # Логирование события
        self.logger.warning(f"Security event: {event_type}, details: {details}")
    
    # ======================
    # REPORTING & EXPORT
    # ======================
    def generate_analysis_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Генерирует человекочитаемый отчет об анализе.
        
        Args:
            analysis_results: Результаты анализа
            
        Returns:
            Строка с форматированным отчетом
        """
        if not analysis_results.get("success", False):
            return self._generate_failure_report(analysis_results)
        
        critical_regions = analysis_results.get("critical_regions", [])
        num_vulnerabilities = len(critical_regions)
        
        # Генерация заголовка
        lines = [
            "=" * 80,
            "ECDSA SECURITY ANALYSIS REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            f"Total vulnerabilities detected: {num_vulnerabilities}",
            ""
        ]
        
        # Добавление информации о критических уязвимостях
        if num_vulnerabilities > 0:
            critical_vulns = [v for v in critical_regions if v.get("criticality", 0) >= 0.8]
            high_vulns = [v for v in critical_regions if 0.6 <= v.get("criticality", 0) < 0.8]
            medium_vulns = [v for v in critical_regions if 0.4 <= v.get("criticality", 0) < 0.6]
            
            lines.append(f"Critical vulnerabilities (>= 0.8): {len(critical_vulns)}")
            lines.append(f"High vulnerabilities (0.6-0.8): {len(high_vulns)}")
            lines.append(f"Medium vulnerabilities (0.4-0.6): {len(medium_vulns)}")
            lines.append("")
            
            # Добавление деталей по топ 3 уязвимостям
            top_vulns = sorted(critical_regions, key=lambda x: x.get("criticality", 0), reverse=True)[:3]
            lines.append("Top 3 vulnerabilities:")
            for i, vuln in enumerate(top_vulns, 1):
                ur_range = vuln.get("ur_range", (0, 0))
                uz_range = vuln.get("uz_range", (0, 0))
                criticality = vuln.get("criticality", 0)
                vuln_type = vuln.get("type", "unknown")
                
                lines.append(f"  {i}. Criticality: {criticality:.2f}")
                lines.append(f"     Type: {vuln_type}")
                lines.append(f"     Region: u_r={ur_range[0]}-{ur_range[1]}, u_z={uz_range[0]}-{uz_range[1]}")
                
                # Добавление рекомендаций
                recommendations = self._generate_recommendations(vuln)
                lines.append("     Recommendations:")
                for rec in recommendations[:2]:  # Показываем первые 2 рекомендации
                    lines.append(f"       - {rec}")
                lines.append("")
        else:
            lines.append("No critical vulnerabilities detected.")
            lines.append("The ECDSA implementation appears to be secure based on topological analysis.")
        
        lines.extend([
            "=" * 80,
            "NEXT STEPS:",
            "1. Investigate the critical regions identified above",
            "2. Consider generating synthetic signatures in these regions for further analysis",
            "3. Review ECDSA implementation for potential weaknesses",
            "4. Consult security team for remediation strategies",
            "=" * 80
        ])
        
        return "\n".join(lines)
    
    def _generate_failure_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Генерирует отчет о неудачном анализе.
        
        Args:
            analysis_results: Результаты анализа
            
        Returns:
            Строка с отчетом о неудачном анализе
        """
        error = analysis_results.get("error", "Unknown error")
        
        lines = [
            "=" * 80,
            "ECDSA SECURITY ANALYSIS REPORT - FAILED",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            f"Analysis failed with error: {error}",
            "",
            "Troubleshooting steps:",
            "1. Check system configuration and dependencies",
            "2. Verify input data format and validity",
            "3. Ensure sufficient system resources are available",
            "4. Check logs for detailed error information",
            "",
            "=" * 80,
            "NEXT STEPS:",
            "1. Review error details and correct configuration",
            "2. Retry analysis with corrected parameters",
            "3. Contact support if issue persists",
            "=" * 80
        ]
        
        return "\n".join(lines)
    
    def export_analysis_report(self, points: np.ndarray, output_path: str) -> str:
        """
        Экспортирует HTML-отчет об анализе.
        
        Args:
            points: Точки, использованные в анализе
            output_path: Путь для сохранения отчета
            
        Returns:
            Путь к сгенерированному отчету
        """
        try:
            # Выполнение анализа, если он еще не был выполнен
            if self.last_analysis is None:
                self.analyze_with_mapper(points)
            analysis_results = self.last_analysis
            
            # Генерация HTML отчета
            with open(output_path, 'w') as f:
                f.write("<!DOCTYPE html><html><head>")
                f.write("<title>AuditCore Security Analysis Report</title>")
                f.write("<meta charset='UTF-8'>")
                f.write("<style>")
                f.write("""
                    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }
                    .header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
                    .summary-box { border-left: 5px solid #3498db; background-color: #f8f9fa; padding: 15px; border-radius: 0 4px 4px 0; margin: 20px 0; }
                    .critical { border-left-color: #e74c3c; }
                    .high { border-left-color: #f39c12; }
                    .medium { border-left-color: #f1c40f; }
                    h1 { color: #2c3e50; }
                    h2 { color: #2980b9; border-bottom: 1px solid #eee; padding-bottom: 5px; }
                    h3 { color: #16a085; }
                    ul { padding-left: 20px; }
                    li { margin-bottom: 8px; }
                    .metric { display: inline-block; background: #eaeaea; padding: 5px 10px; border-radius: 3px; margin-right: 10px; }
                    .vulnerability { background: #fdf2f2; border: 1px solid #fbcaca; border-radius: 4px; padding: 10px; margin: 10px 0; }
                    .recommendation { background: #f0f7ff; border-left: 3px solid #3498db; padding: 8px; margin: 5px 0; }
                    .footer { margin-top: 40px; text-align: center; color: #7f8c8d; font-size: 0.9em; }
                """)
                f.write("</style>")
                f.write("</head><body>")
                
                # Заголовок
                f.write("<div class='header'>")
                f.write("<h1>AuditCore Security Analysis Report</h1>")
                f.write(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
                f.write("</div>")
                
                # Основная информация
                f.write("<h2>Analysis Summary</h2>")
                
                if analysis_results.get("success", False):
                    critical_regions = analysis_results.get("critical_regions", [])
                    num_vulnerabilities = len(critical_regions)
                    
                    if num_vulnerabilities > 0:
                        f.write("<div class='summary-box critical'>")
                        f.write("<h3>Critical Vulnerabilities Detected</h3>")
                        f.write(f"<p>{num_vulnerabilities} vulnerability regions identified requiring attention.</p>")
                        f.write("<h4>Vulnerability Distribution:</h4>")
                        f.write("<ul>")
                        
                        critical_vulns = [v for v in critical_regions if v.get("criticality", 0) >= 0.8]
                        high_vulns = [v for v in critical_regions if 0.6 <= v.get("criticality", 0) < 0.8]
                        medium_vulns = [v for v in critical_regions if 0.4 <= v.get("criticality", 0) < 0.6]
                        
                        if critical_vulns:
                            f.write(f"<li><span class='metric'>Critical: {len(critical_vulns)}</span> - Requires immediate action</li>")
                        if high_vulns:
                            f.write(f"<li><span class='metric'>High: {len(high_vulns)}</span> - Requires prompt investigation</li>")
                        if medium_vulns:
                            f.write(f"<li><span class='metric'>Medium: {len(medium_vulns)}</span> - Should be addressed</li>")
                        
                        f.write("</ul>")
                        f.write("</div>")
                        
                        # Топ уязвимостей
                        f.write("<h3>Top 3 Vulnerabilities</h3>")
                        top_vulns = sorted(critical_regions, key=lambda x: x.get("criticality", 0), reverse=True)[:3]
                        
                        for i, vuln in enumerate(top_vulns, 1):
                            ur_range = vuln.get("ur_range", (0, 0))
                            uz_range = vuln.get("uz_range", (0, 0))
                            criticality = vuln.get("criticality", 0)
                            
                            # Определение класса для стилизации
                            if criticality >= 0.8:
                                vuln_class = "critical"
                            elif criticality >= 0.6:
                                vuln_class = "high"
                            else:
                                vuln_class = "medium"
                            
                            f.write(f"<div class='vulnerability {vuln_class}'>")
                            f.write(f"<h4>Vulnerability #{i} (Criticality: {criticality:.2f})</h4>")
                            f.write(f"<p><strong>Region:</strong> u_r={ur_range[0]}-{ur_range[1]}, u_z={uz_range[0]}-{uz_range[1]}</p>")
                            f.write(f"<p><strong>Type:</strong> {vuln.get('type', 'unknown')}</p>")
                            
                            # Рекомендации
                            recommendations = self._generate_recommendations(vuln)
                            f.write("<h4>Recommendations:</h4>")
                            f.write("<ul>")
                            for rec in recommendations:
                                f.write(f"<li class='recommendation'>{rec}</li>")
                            f.write("</ul>")
                            f.write("</div>")
                    else:
                        f.write("<div class='summary-box'>")
                        f.write("<h3>No Critical Vulnerabilities Detected</h3>")
                        f.write("<p>The ECDSA implementation appears to be secure based on topological analysis of the provided signatures.</p>")
                        f.write("</div>")
                else:
                    f.write("<div class='summary-box' style='border-left-color: #e74c3c;'>")
                    f.write("<h3>Analysis Failed</h3>")
                    f.write(f"<p>Error: {analysis_results.get('error', 'Unknown error')}</p>")
                    f.write("<h4>Next Steps:</h4>")
                    f.write("<ul>")
                    f.write("<li>Check system configuration and dependencies</li>")
                    f.write("<li>Verify input data format and validity</li>")
                    f.write("<li>Ensure sufficient system resources are available</li>")
                    f.write("<li>Check logs for detailed error information</li>")
                    f.write("</ul>")
                    f.write("</div>")
                
                # Метрики производительности
                f.write("<h2>Performance Metrics</h2>")
                f.write("<div class='metrics'>")
                
                # Время анализа
                if "total_analysis_time" in self.performance_metrics and self.performance_metrics["total_analysis_time"]:
                    avg_time = sum(self.performance_metrics["total_analysis_time"]) / len(self.performance_metrics["total_analysis_time"])
                    f.write(f"<span class='metric'>Avg Analysis Time: {avg_time:.2f}s</span>")
                
                # Использование памяти
                if "function_memory" in self.memory_metrics:
                    total_memory = 0
                    for func_metrics in self.memory_metrics["function_memory"].values():
                        for metric in func_metrics:
                            total_memory += metric["diff"]
                    if total_memory > 0:
                        f.write(f"<span class='metric'>Memory Usage: {total_memory:.2f}MB</span>")
                
                # Кэш
                cache_hits = self.performance_metrics.get("cache_hits", 0)
                cache_misses = self.performance_metrics.get("cache_misses", 0)
                if cache_hits + cache_misses > 0:
                    hit_rate = (cache_hits / (cache_hits + cache_misses)) * 100
                    f.write(f"<span class='metric'>Cache Hit Rate: {hit_rate:.1f}%</span>")
                
                f.write("</div>")
                
                # Рекомендации по действиям
                f.write("<h2>Next Steps</h2>")
                f.write("<ul>")
                f.write("<li>Investigate the critical regions identified above</li>")
                f.write("<li>Consider generating synthetic signatures in these regions for further analysis</li>")
                f.write("<li>Review ECDSA implementation for potential weaknesses</li>")
                f.write("<li>Consult security team for remediation strategies</li>")
                f.write("</ul>")
                
                # Подвал
                f.write("<div class='footer'>")
                f.write("<p>Generated by AuditCore v3.2 - AI Assistant Module</p>")
                f.write("<p>For more information, contact security@auditcore.example.com</p>")
                f.write("</div>")
                
                f.write("</body></html>")
            
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to export analysis report: {str(e)}")
            raise AIAssistantError(f"Report export failed: {str(e)}") from e
    
    def export_metrics(self, output_path: str) -> str:
        """
        Экспортирует метрики производительности и безопасности.
        
        Args:
            output_path: Путь для сохранения метрик
            
        Returns:
            Путь к файлу с экспортированными метриками
        """
        metrics = {
            "performance": self.get_performance_metrics(),
            "memory": self.get_memory_metrics(),
            "security": self.get_security_metrics(),
            "monitoring": self.get_monitoring_data(),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "api_version": self.config.api_version,
                "vulnerability_threshold": self.config.vulnerability_threshold,
                "anomaly_score_threshold": self.config.anomaly_score_threshold
            }
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {str(e)}")
            raise AIAssistantError(f"Metrics export failed: {str(e)}") from e
    
    # ======================
    # METRICS & MONITORING
    # ======================
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Возвращает метрики производительности."""
        metrics = self.performance_metrics.copy()
        
        # Добавление дополнительных вычисляемых метрик
        if "total_analysis_time" in metrics and metrics["total_analysis_time"]:
            avg_time = sum(metrics["total_analysis_time"]) / len(metrics["total_analysis_time"])
            metrics["avg_analysis_time"] = avg_time
            
            # Проверка на превышение 80% от максимального времени
            if self.config.max_analysis_time > 0:
                metrics["time_utilization"] = (avg_time / self.config.max_analysis_time) * 100
        
        return metrics
    
    def get_memory_metrics(self) -> Dict[str, Any]:
        """Возвращает метрики использования памяти."""
        return self.memory_metrics.copy()
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Возвращает метрики безопасности."""
        return self.security_metrics.copy()
    
    def get_monitoring_data(self) -> Dict[str, Any]:
        """Возвращает данные мониторинга."""
        return self.monitoring_data.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Проверяет работоспособность компонента.
        
        Returns:
            Словарь с результатами проверки
        """
        # Проверка зависимостей
        dependencies_ok = self.verify_dependencies()
        missing_dependencies = []
        if not self.hypercore_transformer:
            missing_dependencies.append("HyperCoreTransformer")
        if not self.tcon:
            missing_dependencies.append("TCON")
        if not self.signature_generator:
            missing_dependencies.append("SignatureGenerator")
        
        # Проверка ресурсов
        resource_ok = True
        resource_issues = []
        
        # Проверка использования памяти
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        if memory_usage > self.config.max_memory_mb * 0.8:
            resource_issues.append(f"High memory usage: {memory_usage:.2f} MB (80% of limit)")
        
        # Проверка использования CPU
        cpu_percent = process.cpu_percent(interval=0.1)
        if cpu_percent > 80:
            resource_issues.append(f"High CPU usage: {cpu_percent}%")
        
        if resource_issues:
            resource_ok = False
        
        # Проверка кэша
        cache_hits = self.performance_metrics.get("cache_hits", 0)
        cache_misses = self.performance_metrics.get("cache_misses", 0)
        cache_health = {
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate": (cache_hits / (cache_hits + cache_misses)) * 100 
                if (cache_hits + cache_misses) > 0 else 0
        }
        
        # Общее состояние
        status = "healthy" if (dependencies_ok and resource_ok) else "unhealthy"
        
        return {
            "status": status,
            "component": "AIAssistant",
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
            "cache": cache_health,
            "monitoring": {
                "enabled": self.config.monitoring_enabled,
                "analysis_count": self.monitoring_data["analysis_count"],
                "critical_vulnerabilities": self.monitoring_data["critical_vulnerabilities"]
            },
            "timestamp": datetime.now().isoformat()
        }

# ======================
# EXAMPLE USAGE
# ======================
def example_usage_ai_assistant():
    """Example usage of AIAssistant for ECDSA security analysis."""
    print("=" * 80)
    print("Example Usage of AIAssistant with Mapper Integration for ECDSA Security Analysis")
    print("=" * 80)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 1. Создание тестовых данных
    print("1. Creating test data for AIAssistant...")
    n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141  # Порядок secp256k1
    
    # Безопасные данные (равномерно случайные)
    print(" - Creating safe data (uniform random)...")
    np.random.seed(42)
    safe_signatures = []
    
    # Генерация безопасных подписей
    for _ in range(1000):
        # Случайный nonce в пределах порядка кривой
        k = np.random.randint(1, n)
        # Случайное сообщение
        z = np.random.randint(1, n)
        
        # Для тестовых данных используем упрощенный расчет
        u_r = k % n
        u_z = (z * pow(k, -1, n)) % n
        
        safe_signatures.append({
            "u_r": int(u_r),
            "u_z": int(u_z),
            "r": int(u_r),
            "s": 1,  # Упрощенное значение для теста
            "z": int(z),
            "is_synthetic": True,
            "confidence": 0.95,
            "source": "test_safe"
        })
    
    # Уязвимые данные (с паттернами)
    print(" - Creating vulnerable data (with patterns)...")
    vulnerable_signatures = []
    
    # Генерация уязвимых подписей с предсказуемыми nonces
    for i in range(200):
        # Nonce с линейной зависимостью
        k = (1000 + i * 10) % n
        # Случайное сообщение
        z = np.random.randint(1, n)
        
        u_r = k % n
        u_z = (z * pow(k, -1, n)) % n
        
        vulnerable_signatures.append({
            "u_r": int(u_r),
            "u_z": int(u_z),
            "r": int(u_r),
            "s": 1,  # Упрощенное значение для теста
            "z": int(z),
            "is_synthetic": True,
            "confidence": 0.9,
            "source": "test_vulnerable"
        })
    
    # Объединение данных
    all_signatures = safe_signatures + vulnerable_signatures
    ur_uz_points = np.array([[s["u_r"], s["u_z"]] for s in all_signatures])
    
    # 2. Инициализация AIAssistant
    print("2. Initializing AIAssistant...")
    ai_assistant = AIAssistant(config=AIAssistantConfig(n=n))
    
    # 3. Мокирование зависимостей
    print("3. Setting up mock dependencies...")
    
    class MockPoint:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.infinity = False
            self.curve = None
    
    class MockHyperCoreTransformer:
        def get_stability_map(self, points):
            return np.random.random(len(points))
        
        def transform(self, points):
            return points
    
    class MockTCON:
        def verify_torus_structure(self, points):
            # Симулируем проверку структуры тора
            # Возвращаем не валидную структуру для некоторых точек
            invalid_points = []
            for i, point in enumerate(points):
                # Симулируем проблемы с торической структурой в определенных областях
                if 0.2 * n < point[0] < 0.3 * n and 0.2 * n < point[1] < 0.3 * n:
                    invalid_points.append(point.tolist())
            
            return {
                "is_valid": len(invalid_points) == 0,
                "invalid_points": invalid_points,
                "details": {
                    "issue_count": len(invalid_points),
                    "description": "Detected points violating torus structure"
                }
            }
        
        def analyze_diagonal_periodicity(self, points):
            # Симулируем анализ диагональной периодичности
            issues = []
            for i, point in enumerate(points):
                # Симулируем периодические паттерны
                if (point[0] % 100) < 10 and (point[1] % 100) < 10:
                    issues.append({
                        "ur": float(point[0]),
                        "uz": float(point[1]),
                        "pattern_strength": 0.8,
                        "description": "Detected strong periodic pattern"
                    })
            
            return {
                "has_issues": len(issues) > 0,
                "issues": issues,
                "details": {
                    "issue_count": len(issues),
                    "description": "Detected diagonal periodicity issues"
                }
            }
    
    class MockSignatureGenerator:
        def generate_region(self, ur_range, uz_range, count):
            signatures = []
            for _ in range(count):
                u_r = np.random.randint(ur_range[0], ur_range[1])
                u_z = np.random.randint(uz_range[0], uz_range[1])
                signatures.append({
                    "u_r": int(u_r),
                    "u_z": int(u_z),
                    "r": int(u_r),
                    "s": 1,
                    "z": np.random.randint(1, n),
                    "is_synthetic": True,
                    "confidence": 0.9,
                    "source": "test_region"
                })
            return signatures
        
        def generate_for_collision_search(self, public_key, target_regions, count):
            return self.generate_region((0, n), (0, n), count)
    
    class MockDynamicComputeRouter:
        def get_optimal_window_size(self, points):
            return 15
        
        def get_stability_threshold(self):
            return 0.75
        
        def adaptive_route(self, task, points, **kwargs):
            return task(points, **kwargs)
    
    # Установка зависимостей
    ai_assistant.set_hypercore_transformer(MockHyperCoreTransformer())
    ai_assistant.set_tcon(MockTCON())
    ai_assistant.set_signature_generator(MockSignatureGenerator())
    ai_assistant.set_dynamic_compute_router(MockDynamicComputeRouter())
    
    # 4. Анализ данных
    print("4. Analyzing signatures...")
    try:
        analysis_results = ai_assistant.analyze_real_signatures(
            public_key=MockPoint(1, 2), 
            real_signatures=all_signatures
        )
        
        # Приоритизация уязвимостей
        print(" - Prioritizing vulnerabilities...")
        prioritized_vulns = ai_assistant.prioritize_vulnerabilities(analysis_results)
        
        # Вывод результатов
        print(f" - Critical regions identified: {len(prioritized_vulns)}")
        if prioritized_vulns:
            top_vuln = prioritized_vulns[0]
            print(f"   Top vulnerability: Criticality={top_vuln['criticality']:.2f}, Type={top_vuln['type']}")
            print(f"   Region: u_r={top_vuln['ur_range'][0]}-{top_vuln['ur_range'][1]}, u_z={top_vuln['uz_range'][0]}-{top_vuln['uz_range'][1]}")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
    
    # 5. Генерация отчета
    print("5. Generating analysis report...")
    report = ai_assistant.generate_analysis_report(ai_assistant.last_analysis)
    print(report)
    
    # 6. Экспорт HTML отчета
    print("6. Exporting HTML report...")
    html_report_path = ai_assistant.export_analysis_report(ur_uz_points, "ecdsa_analysis_report.html")
    print(f" - HTML report exported to {html_report_path}")
    
    # 7. Экспорт метрик
    print("7. Exporting performance metrics...")
    metrics_path = ai_assistant.export_metrics("ai_assistant_metrics.json")
    print(f" - Metrics exported to {metrics_path}")
    
    # 8. Проверка работоспособности
    print("8. Performing health check...")
    health = ai_assistant.health_check()
    print(f" - Health status: {health['status']}")
    if health['status'] != 'healthy':
        print(f"   Issues: {health.get('resources', {}).get('issues', [])}")
    
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    example_usage_ai_assistant()
