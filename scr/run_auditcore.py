#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный скрипт запуска AuditCore v3.2
Инжектирует зависимости и запускает полный анализ.
"""

import numpy as np
import logging
from nerve_theorem import NerveTheorem
from datetime import datetime
from signature_generator import SignatureGenerator  # Предполагаем, что он есть
from hypercore_transformer import HyperCoreTransformer
from betti_analyzer import BettiAnalyzer
from topological_analyzer import TopologicalAnalyzer
from dynamic_compute_router import DynamicComputeRouter
from gradient_analysis import GradientAnalysis
from collision_engine import CollisionEngine
from nerve_theorem import NerveTheorem  # Из файла ai_assistant3.py
from mapper import Mapper  # Из файла ai_assistant3.py
from smoothing import Smoothing  # Из файла ai_assistant3.py

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AuditCoreRunner")

# --- 1. Конфигурация ---
N = 115792089237316195423570985008687907852837564279074904382605163141518161494337  # secp256k1.n
NUM_SIGNATURES = 1000  # Количество подписей для анализа

# --- 2. Инициализация компонентов ---
logger.info("1. Инициализация компонентов...")

# Генератор подписей (реальный или мок)
class MockSignatureGenerator:
    def generate_region(self, public_key, ur_range, uz_range, num_points=100, step=None):
        # Генерируем случайные u_r, u_z
        u_r = np.random.randint(1, N, num_points)
        u_z = np.random.randint(0, N, num_points)
        # Симуляция r = (u_r * d + u_z) % n (для теста с уязвимостью)
        d = 123456789  # Приватный ключ (для демонстрации)
        r = (u_r * d + u_z) % N
        signatures = []
        for i in range(num_points):
            sig = {
                "r": int(r[i]),
                "s": int(u_r[i]),  # Симуляция s
                "z": int(u_z[i]),  # Симуляция z
                "u_r": int(u_r[i]),
                "u_z": int(u_z[i]),
                "is_synthetic": True,
                "confidence": 1.0,
                "source": "mock"
            }
            signatures.append(sig)
        return signatures

    def generate_for_collision_search(self, public_key, base_u_r, base_u_z, search_radius):
        # Реализуйте аналогично generate_region, но в радиусе
        pass  # Для примера можно не реализовывать

signature_gen = MockSignatureGenerator()

# HyperCoreTransformer
hypercore = HyperCoreTransformer(n=N)

# BettiAnalyzer
betti_analyzer = BettiAnalyzer(curve_n=N)

# TopologicalAnalyzer
topo_analyzer = TopologicalAnalyzer(n=N)

# DynamicComputeRouter
router = DynamicComputeRouter()

# GradientAnalysis
grad_analyzer = GradientAnalysis(curve_n=N)

# CollisionEngine
collision_engine = CollisionEngine(curve_n=N)

# NerveTheorem (из ai_assistant3.py)
nerve_theorem = NerveTheorem(config=router.config)  # Передаем конфиг из router

# Mapper (из ai_assistant3.py)
mapper = Mapper()  # Нужно реализовать, если используется

# Smoothing (из ai_assistant3.py)
smoothing = Smoothing()  # Нужно реализовать, если используется

# --- 3. Инъекция зависимостей (КРИТИЧНО!) ---
logger.info("2. Инъекция зависимостей...")

# TopologicalAnalyzer зависит от всех!
topo_analyzer.set_dependencies(
    nerve_theorem=nerve_theorem,
    mapper=mapper,
    smoothing=smoothing,
    betti_analyzer=betti_analyzer,
    hypercore_transformer=hypercore,
    dynamic_compute_router=router
)

# GradientAnalysis зависит от:
grad_analyzer.set_signature_generator(signature_gen)
grad_analyzer.set_hypercore_transformer(hypercore)

# CollisionEngine зависит от:
collision_engine.set_signature_generator(signature_gen)
collision_engine.set_topological_analyzer(topo_analyzer)
collision_engine.set_gradient_analysis(grad_analyzer)
collision_engine.set_dynamic_compute_router(router)

# BettiAnalyzer зависит от:
betti_analyzer.set_nerve_theorem(nerve_theorem)
betti_analyzer.set_smoothing(smoothing)
betti_analyzer.set_dynamic_compute_router(router)

logger.info("✅ Все зависимости инжектированы.")

# --- 4. Генерация данных ---
logger.info("3. Генерация тестовых подписей...")
np.random.seed(42)
# Генерируем точки (u_r, u_z)
u_r_samples = np.random.randint(1, N, NUM_SIGNATURES)
u_z_samples = np.random.randint(0, N, NUM_SIGNATURES)
points = np.column_stack((u_r_samples, u_z_samples))

# --- 5. Запуск полного анализа ---
logger.info("4. Запуск TopologicalAnalyzer...")
try:
    result = topo_analyzer.analyze(points)
    
    print("\n" + "="*80)
    print("📊 РЕЗУЛЬТАТ АНАЛИЗА")
    print("="*80)
    print(f"Статус: {result.status.value.upper()}")
    print(f"Anomaly Score: {result.anomaly_score:.4f}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Betti Numbers: β₀={result.betti_numbers.beta_0}, β₁={result.betti_numbers.beta_1}, β₂={result.betti_numbers.beta_2}")
    print(f"Torus Structure: {'ДА' if result.is_torus_structure else 'НЕТ'}")
    print(f"Время выполнения: {result.execution_time:.2f} сек")
    print(f"Память: {result.resource_usage.get('memory_mb', 0):.2f} MB")
    
    print(f"\n⚠️ Обнаружено уязвимостей: {len(result.vulnerabilities)}")
    for i, vuln in enumerate(result.vulnerabilities[:5], 1):  # Только первые 5
        print(f"  {i}. {vuln['type']} | Локация: ({vuln['location'][0]:.0f}, {vuln['location'][1]:.0f}) | Критичность: {vuln['criticality']:.3f}")

    # Экспорт в JSON
    json_report = topo_analyzer.export_results(result, "json")
    with open("auditcore_result.json", "w") as f:
        f.write(json_report)
    logger.info("📄 Отчёт экспортирован в auditcore_result.json")

except Exception as e:
    logger.error(f"❌ Анализ завершился с ошибкой: {e}", exc_info=True)

# --- 6. Дополнительно: попробуем Gradient Analysis ---
logger.info("5. Запуск Gradient Analysis (только для проверки)...")
try:
    # В реальности нужно передать (u_r, u_z, r) — получим их через HyperCoreTransformer
    # Но в нашем моке мы можем создать данные вручную
    r_vals = [(ur * 123456789 + uz) % N for ur, uz in points]  # Симуляция r
    ur_uz_r = np.column_stack((u_r_samples, u_z_samples, r_vals))
    grad_result = grad_analyzer.analyze_gradient(ur_uz_r)
    print(f"\n📈 Gradient Analysis: d_est={grad_result.estimated_d_heuristic}, conf={grad_result.heuristic_confidence:.4f}")
    print("❗ Это хеуристическая оценка с низкой уверенностью (0.1)! Не доверяйте ей как основному методу.")
except Exception as e:
    logger.warning(f"Gradient Analysis не сработал: {e}")


logger.info("🎉 AuditCore v3.2 успешно запущен!")
