# -*- coding: utf-8 -*-
"""Mapper Implementation for AuditCore v3.2
Implementation based on Multiscale Mapper algorithm from TDA literature.
This is a simplified but functional implementation compatible with modern NumPy.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

# Configure logger
logger = logging.getLogger("AuditCore.Mapper")
logger.addHandler(logging.NullHandler())

@dataclass
class MapperResult:
    """Result of Mapper analysis."""
    graph: Dict[int, List[int]]  # Adjacency list: node_id -> connected nodes
    clusters: List[List[int]]    # List of point indices per cluster
    cover: List[Dict[str, Any]]  # Cover used for clustering
    persistence_diagram: Optional[List[Tuple[float, float]]] = None
    stability_score: float = 0.0
    execution_time: float = 0.0

class Mapper:
    """Simplified Mapper algorithm for ECDSA signature space analysis."""

    def __init__(self, n: int = 115792089237316195423570985008687907852837564279074904382605163141518161494337):
        self.n = n
        self.logger = logging.getLogger("AuditCore.Mapper")

    def get_critical_regions(self, points: np.ndarray, num_intervals: int = 10, overlap_percent: int = 70) -> List[Dict[str, Any]]:
        """
        Identifies critical regions in (u_r, u_z) space using interval covering.
        This is a simplified version of the Mapper algorithm.
        
        Args:
            points: Array of shape (N, 2) with (u_r, u_z) coordinates.
            num_intervals: Number of intervals along each dimension.
            overlap_percent: Overlap between adjacent intervals (%).

        Returns:
            List of cover cells (regions) with metadata.
        """
        if len(points) == 0:
            return []

        # Extract coordinates
        u_r_vals = points[:, 0]
        u_z_vals = points[:, 1]

        # Define ranges and step sizes
        r_min, r_max = int(np.min(u_r_vals)), int(np.max(u_r_vals))
        z_min, z_max = int(np.min(u_z_vals)), int(np.max(u_z_vals))

        # Avoid division by zero
        if r_max <= r_min:
            r_max = r_min + 1
        if z_max <= z_min:
            z_max = z_min + 1

        r_range = r_max - r_min
        z_range = z_max - z_min

        # Calculate interval size and overlap
        r_step = max(1, r_range // num_intervals)
        z_step = max(1, z_range // num_intervals)

        overlap_r = int(r_step * overlap_percent / 100)
        overlap_z = int(z_step * overlap_percent / 100)

        # Generate sliding windows
        cover = []
        for start_r in range(r_min, r_max, r_step - overlap_r):
            end_r = min(start_r + r_step, r_max)
            for start_z in range(z_min, z_max, z_step - overlap_z):
                end_z = min(start_z + z_step, z_max)

                # Find points in this cell
                in_cell = (
                    (u_r_vals >= start_r) &
                    (u_r_vals < end_r) &
                    (u_z_vals >= start_z) &
                    (u_z_vals < end_z)
                )
                indices = np.where(in_cell)[0].tolist()

                if len(indices) > 0:
                    cover.append({
                        "x1": start_r,
                        "y1": start_z,
                        "x2": end_r,
                        "y2": end_z,
                        "points": indices,
                        "diameter": max(end_r - start_r, end_z - start_z),
                        "size": len(indices)
                    })

        return cover

    def build_graph(self, cover: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        """
        Builds the Mapper graph by connecting overlapping clusters.
        Each cluster becomes a node; edges exist if clusters share points.
        """
        n_clusters = len(cover)
        graph = {i: [] for i in range(n_clusters)}

        # Check pairwise overlaps
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                # If two clusters share any point, connect them
                set_i = set(cover[i]["points"])
                set_j = set(cover[j]["points"])
                if set_i & set_j:  # Intersection exists
                    graph[i].append(j)
                    graph[j].append(i)

        return graph

    def analyze(self, points: np.ndarray, **kwargs) -> MapperResult:
        """
        Performs full Mapper analysis on the given points.

        Args:
            points: Array of shape (N, 2) with (u_r, u_z) coordinates.

        Returns:
            MapperResult object with graph, clusters, cover, etc.
        """
        start_time = time.time()

        # Step 1: Generate cover
        cover = self.get_critical_regions(
            points,
            num_intervals=kwargs.get('num_intervals', 10),
            overlap_percent=kwargs.get('overlap_percent', 70)
        )

        # Step 2: Build graph
        graph = self.build_graph(cover)

        # Step 3: Extract clusters (each cover cell is a cluster)
        clusters = [cell["points"] for cell in cover]

        # Step 4: Compute stability score (simplified)
        total_points = len(points)
        avg_cluster_size = np.mean([len(c) for c in clusters]) if clusters else 0
        stability_score = min(1.0, avg_cluster_size / (total_points * 0.1)) if total_points > 0 else 0.0

        # Step 5: Return result
        result = MapperResult(
            graph=graph,
            clusters=clusters,
            cover=cover,
            stability_score=stability_score,
            execution_time=time.time() - start_time
        )

        self.logger.info(f"[Mapper] Analysis completed: {len(clusters)} clusters, {result.stability_score:.3f} stability")
        return result

    def get_critical_regions(self, points: np.ndarray, num_intervals: int = 10, overlap_percent: int = 70) -> List[Dict[str, Any]]:
        """
        Identifies critical regions in (u_r, u_z) space using interval covering.
        This is a simplified version of the Mapper algorithm.
        
        Args:
            points: Array of shape (N, 2) with (u_r, u_z) coordinates.
            num_intervals: Number of intervals along each dimension.
            overlap_percent: Overlap between adjacent intervals (%).

        Returns:
            List of cover cells (regions) with metadata.
        """
        if len(points) == 0:
            return []

        # Extract coordinates
        u_r_vals = points[:, 0]
        u_z_vals = points[:, 1]

        # Define ranges and step sizes
        r_min, r_max = int(np.min(u_r_vals)), int(np.max(u_r_vals))
        z_min, z_max = int(np.min(u_z_vals)), int(np.max(u_z_vals))

        # Avoid division by zero
        if r_max <= r_min:
            r_max = r_min + 1
        if z_max <= z_min:
            z_max = z_min + 1

        r_range = r_max - r_min
        z_range = z_max - z_min

        # Calculate interval size and overlap
        r_step = max(1, r_range // num_intervals)
        z_step = max(1, z_range // num_intervals)

        overlap_r = int(r_step * overlap_percent / 100)
        overlap_z = int(z_step * overlap_percent / 100)

        # Generate sliding windows
        cover = []
        for start_r in range(r_min, r_max, r_step - overlap_r):
            end_r = min(start_r + r_step, r_max)
            for start_z in range(z_min, z_max, z_step - overlap_z):
                end_z = min(start_z + z_step, z_max)

                # Find points in this cell
                in_cell = (
                    (u_r_vals >= start_r) &
                    (u_r_vals < end_r) &
                    (u_z_vals >= start_z) &
                    (u_z_vals < end_z)
                )
                indices = np.where(in_cell)[0].tolist()

                if len(indices) > 0:
                    cover.append({
                        "x1": start_r,
                        "y1": start_z,
                        "x2": end_r,
                        "y2": end_z,
                        "points": indices,
                        "diameter": max(end_r - start_r, end_z - start_z),
                        "size": len(indices)
                    })

        return cover

    def build_graph(self, cover: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        """
        Builds the Mapper graph by connecting overlapping clusters.
        Each cluster becomes a node; edges exist if clusters share points.
        """
        n_clusters = len(cover)
        graph = {i: [] for i in range(n_clusters)}

        # Check pairwise overlaps
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                # If two clusters share any point, connect them
                set_i = set(cover[i]["points"])
                set_j = set(cover[j]["points"])
                if set_i & set_j:  # Intersection exists
                    graph[i].append(j)
                    graph[j].append(i)

        return graph

    def analyze(self, points: np.ndarray, **kwargs) -> MapperResult:
        """
        Performs full Mapper analysis on the given points.

        Args:
            points: Array of shape (N, 2) with (u_r, u_z) coordinates.

        Returns:
            MapperResult object with graph, clusters, cover, etc.
        """
        start_time = time.time()

        # Step 1: Generate cover
        cover = self.get_critical_regions(
            points,
            num_intervals=kwargs.get('num_intervals', 10),
            overlap_percent=kwargs.get('overlap_percent', 70)
        )

        # Step 2: Build graph
        graph = self.build_graph(cover)

        # Step 3: Extract clusters (each cover cell is a cluster)
        clusters = [cell["points"] for cell in cover]

        # Step 4: Compute stability score (simplified)
        total_points = len(points)
        avg_cluster_size = np.mean([len(c) for c in clusters]) if clusters else 0
        stability_score = min(1.0, avg_cluster_size / (total_points * 0.1)) if total_points > 0 else 0.0

        # Step 5: Return result
        result = MapperResult(
            graph=graph,
            clusters=clusters,
            cover=cover,
            stability_score=stability_score,
            execution_time=time.time() - start_time
        )

        self.logger.info(f"[Mapper] Analysis completed: {len(clusters)} clusters, {result.stability_score:.3f} stability")
        return result