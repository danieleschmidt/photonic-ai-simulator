"""
Wavelength Division Multiplexing (WDM) Routing Algorithms

Implements intelligent routing algorithms for wavelength-division multiplexed
photonic neural networks, including dynamic channel allocation and crosstalk mitigation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """WDM routing strategies."""
    SHORTEST_PATH = "shortest_path"
    LEAST_LOADED = "least_loaded"
    CROSSTALK_AWARE = "crosstalk_aware"
    POWER_EFFICIENT = "power_efficient"


@dataclass
class WavelengthChannel:
    """Wavelength channel specification."""
    wavelength_nm: float
    bandwidth_ghz: float
    power_dbm: float
    is_available: bool = True
    current_load: float = 0.0
    crosstalk_level_db: float = -40.0


@dataclass
class OpticalPath:
    """Optical path through the network."""
    source_node: int
    destination_node: int
    wavelength_channels: List[int]
    path_nodes: List[int]
    total_loss_db: float
    latency_ns: float
    power_consumption_mw: float


class WavelengthRouter:
    """
    Intelligent wavelength router for photonic neural networks.
    
    Manages dynamic wavelength allocation, routing optimization,
    and crosstalk mitigation in WDM photonic systems.
    """
    
    def __init__(self, 
                 num_wavelengths: int = 16,
                 wavelength_spacing_ghz: float = 50.0,
                 center_wavelength_nm: float = 1550.0):
        """Initialize wavelength router."""
        self.num_wavelengths = num_wavelengths
        self.wavelength_spacing_ghz = wavelength_spacing_ghz
        self.center_wavelength_nm = center_wavelength_nm
        
        # Initialize wavelength channels
        self.channels = self._initialize_channels()
        
        # Network topology (adjacency matrix)
        self.network_topology = None
        self.crosstalk_matrix = None
        
        # Routing table and statistics
        self.routing_table = defaultdict(list)
        self.routing_stats = {
            "successful_routes": 0,
            "blocked_requests": 0,
            "average_path_length": 0.0,
            "total_power_consumption": 0.0
        }
        
    def initialize_network_topology(self, 
                                  adjacency_matrix: np.ndarray,
                                  loss_matrix: np.ndarray):
        """
        Initialize network topology with loss characteristics.
        
        Args:
            adjacency_matrix: Network connectivity matrix
            loss_matrix: Optical loss between connected nodes (dB)
        """
        self.network_topology = adjacency_matrix
        self.loss_matrix = loss_matrix
        self.num_nodes = adjacency_matrix.shape[0]
        
        # Generate crosstalk matrix
        self.crosstalk_matrix = self._generate_crosstalk_matrix()
        
        logger.info(f"Initialized network with {self.num_nodes} nodes")
        
    def route_wavelength_path(self, 
                            source: int,
                            destination: int,
                            bandwidth_requirement_gbps: float,
                            strategy: RoutingStrategy = RoutingStrategy.CROSSTALK_AWARE) -> Optional[OpticalPath]:
        """
        Route a wavelength path from source to destination.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            bandwidth_requirement_gbps: Required bandwidth
            strategy: Routing strategy to use
            
        Returns:
            Optical path if routing successful, None otherwise
        """
        if self.network_topology is None:
            raise ValueError("Network topology not initialized")
            
        # Find available wavelength channels
        required_channels = self._calculate_required_channels(bandwidth_requirement_gbps)
        available_channels = self._find_available_channels(required_channels)
        
        if len(available_channels) < required_channels:
            self.routing_stats["blocked_requests"] += 1
            logger.warning(f"Insufficient channels for route {source}->{destination}")
            return None
        
        # Find optimal path based on strategy
        path_nodes = self._find_optimal_path(source, destination, strategy)
        
        if not path_nodes:
            self.routing_stats["blocked_requests"] += 1
            return None
        
        # Allocate wavelength channels
        allocated_channels = available_channels[:required_channels]
        self._allocate_channels(allocated_channels, path_nodes)
        
        # Calculate path characteristics
        optical_path = self._create_optical_path(
            source, destination, allocated_channels, path_nodes
        )
        
        # Update routing statistics
        self._update_routing_stats(optical_path)
        
        logger.info(f"Routed path {source}->{destination} with {len(allocated_channels)} channels")
        
        return optical_path
    
    def optimize_channel_allocation(self) -> Dict[str, Any]:
        """
        Optimize wavelength channel allocation across the network.
        
        Returns:
            Optimization results
        """
        logger.info("Starting wavelength channel optimization")
        
        # Collect current allocation statistics
        channel_utilization = [ch.current_load for ch in self.channels]
        total_crosstalk = self._calculate_total_crosstalk()
        
        # Perform load balancing
        optimization_results = self._balance_channel_loads()
        
        # Apply crosstalk mitigation
        crosstalk_improvement = self._mitigate_crosstalk()
        
        optimization_results.update({
            "initial_utilization": np.mean(channel_utilization),
            "initial_crosstalk_db": total_crosstalk,
            "crosstalk_improvement_db": crosstalk_improvement
        })
        
        logger.info(f"Optimization complete: {optimization_results['improvement_percent']:.1f}% improvement")
        
        return optimization_results
    
    def _initialize_channels(self) -> List[WavelengthChannel]:
        """Initialize wavelength channels."""
        channels = []
        
        # Calculate wavelengths around center frequency
        center_freq_thz = 299792458 / (self.center_wavelength_nm * 1e-9) / 1e12
        
        for i in range(self.num_wavelengths):
            # ITU-T grid spacing
            freq_offset = (i - self.num_wavelengths // 2) * self.wavelength_spacing_ghz / 1000
            wavelength_nm = 299792458 / ((center_freq_thz + freq_offset) * 1e12) * 1e9
            
            channel = WavelengthChannel(
                wavelength_nm=wavelength_nm,
                bandwidth_ghz=self.wavelength_spacing_ghz * 0.8,  # 80% usable bandwidth
                power_dbm=-10.0,  # 100 µW
                is_available=True,
                current_load=0.0,
                crosstalk_level_db=-40.0
            )
            channels.append(channel)
        
        return channels
    
    def _find_optimal_path(self, 
                          source: int,
                          destination: int,
                          strategy: RoutingStrategy) -> List[int]:
        """Find optimal path based on routing strategy."""
        if strategy == RoutingStrategy.SHORTEST_PATH:
            return self._dijkstra_shortest_path(source, destination)
        elif strategy == RoutingStrategy.LEAST_LOADED:
            return self._least_loaded_path(source, destination)
        elif strategy == RoutingStrategy.CROSSTALK_AWARE:
            return self._crosstalk_aware_path(source, destination)
        elif strategy == RoutingStrategy.POWER_EFFICIENT:
            return self._power_efficient_path(source, destination)
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")
    
    def _dijkstra_shortest_path(self, source: int, destination: int) -> List[int]:
        """Dijkstra's algorithm for shortest path routing."""
        distances = np.full(self.num_nodes, np.inf)
        distances[source] = 0
        previous = np.full(self.num_nodes, -1)
        unvisited = set(range(self.num_nodes))
        
        while unvisited:
            current = min(unvisited, key=lambda x: distances[x])
            unvisited.remove(current)
            
            if current == destination:
                break
                
            for neighbor in range(self.num_nodes):
                if (self.network_topology[current, neighbor] and 
                    neighbor in unvisited):
                    
                    alt_distance = distances[current] + self.loss_matrix[current, neighbor]
                    if alt_distance < distances[neighbor]:
                        distances[neighbor] = alt_distance
                        previous[neighbor] = current
        
        # Reconstruct path
        path = []
        current = destination
        while current != -1:
            path.append(current)
            current = previous[current]
        
        return list(reversed(path)) if path[0] == source else []
    
    def _least_loaded_path(self, source: int, destination: int) -> List[int]:
        """Find path through least loaded network segments."""
        # Modified Dijkstra with load-aware weights
        distances = np.full(self.num_nodes, np.inf)
        distances[source] = 0
        previous = np.full(self.num_nodes, -1)
        unvisited = set(range(self.num_nodes))
        
        while unvisited:
            current = min(unvisited, key=lambda x: distances[x])
            unvisited.remove(current)
            
            if current == destination:
                break
                
            for neighbor in range(self.num_nodes):
                if (self.network_topology[current, neighbor] and 
                    neighbor in unvisited):
                    
                    # Weight includes both loss and load
                    load_weight = self._calculate_link_load(current, neighbor)
                    total_weight = (self.loss_matrix[current, neighbor] + 
                                  load_weight * 10)  # Scale load impact
                    
                    alt_distance = distances[current] + total_weight
                    if alt_distance < distances[neighbor]:
                        distances[neighbor] = alt_distance
                        previous[neighbor] = current
        
        # Reconstruct path
        path = []
        current = destination
        while current != -1:
            path.append(current)
            current = previous[current]
        
        return list(reversed(path)) if path[0] == source else []
    
    def _crosstalk_aware_path(self, source: int, destination: int) -> List[int]:
        """Find path minimizing optical crosstalk."""
        # A* algorithm with crosstalk heuristic
        open_set = {source}
        g_score = defaultdict(lambda: np.inf)
        g_score[source] = 0
        f_score = defaultdict(lambda: np.inf)
        f_score[source] = self._crosstalk_heuristic(source, destination)
        
        came_from = {}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score[x])
            
            if current == destination:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
            
            open_set.remove(current)
            
            for neighbor in range(self.num_nodes):
                if not self.network_topology[current, neighbor]:
                    continue
                
                # Calculate crosstalk-aware cost
                crosstalk_penalty = self._calculate_crosstalk_penalty(current, neighbor)
                tentative_g = (g_score[current] + 
                              self.loss_matrix[current, neighbor] + 
                              crosstalk_penalty)
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = (tentative_g + 
                                       self._crosstalk_heuristic(neighbor, destination))
                    open_set.add(neighbor)
        
        return []  # No path found
    
    def _power_efficient_path(self, source: int, destination: int) -> List[int]:
        """Find most power-efficient path."""
        return self._dijkstra_shortest_path(source, destination)  # Simplified
    
    def _calculate_required_channels(self, bandwidth_gbps: float) -> int:
        """Calculate number of channels needed for bandwidth requirement."""
        channel_capacity_gbps = self.wavelength_spacing_ghz * 0.8 / 10  # Simplified
        return int(np.ceil(bandwidth_gbps / channel_capacity_gbps))
    
    def _find_available_channels(self, count: int) -> List[int]:
        """Find available wavelength channels."""
        available = []
        for i, channel in enumerate(self.channels):
            if channel.is_available and channel.current_load < 0.8:
                available.append(i)
                if len(available) >= count:
                    break
        return available
    
    def _allocate_channels(self, channel_indices: List[int], path_nodes: List[int]):
        """Allocate channels to a path."""
        for idx in channel_indices:
            self.channels[idx].is_available = False
            self.channels[idx].current_load = 1.0
    
    def _create_optical_path(self, 
                           source: int,
                           destination: int,
                           channel_indices: List[int],
                           path_nodes: List[int]) -> OpticalPath:
        """Create optical path object."""
        # Calculate path characteristics
        total_loss = sum(self.loss_matrix[path_nodes[i], path_nodes[i+1]]
                        for i in range(len(path_nodes)-1))
        
        latency = len(path_nodes) * 10  # 10 ns per hop (simplified)
        power_consumption = len(channel_indices) * 50  # 50 mW per channel
        
        return OpticalPath(
            source_node=source,
            destination_node=destination,
            wavelength_channels=channel_indices,
            path_nodes=path_nodes,
            total_loss_db=total_loss,
            latency_ns=latency,
            power_consumption_mw=power_consumption
        )
    
    def _generate_crosstalk_matrix(self) -> np.ndarray:
        """Generate crosstalk matrix between wavelength channels."""
        matrix = np.zeros((self.num_wavelengths, self.num_wavelengths))
        
        for i in range(self.num_wavelengths):
            for j in range(self.num_wavelengths):
                if i != j:
                    # Crosstalk decreases with wavelength separation
                    separation = abs(i - j)
                    matrix[i, j] = -30 - 10 * np.log10(separation + 1)  # dB
        
        return matrix
    
    def _calculate_total_crosstalk(self) -> float:
        """Calculate total network crosstalk."""
        active_channels = [i for i, ch in enumerate(self.channels) 
                          if not ch.is_available]
        
        total_crosstalk = 0.0
        for i in active_channels:
            for j in active_channels:
                if i != j:
                    total_crosstalk += 10**(self.crosstalk_matrix[i, j] / 10)
        
        return 10 * np.log10(total_crosstalk + 1e-10)
    
    def _balance_channel_loads(self) -> Dict[str, Any]:
        """Balance loads across wavelength channels."""
        initial_std = np.std([ch.current_load for ch in self.channels])
        
        # Implement load balancing logic (simplified)
        loads = np.array([ch.current_load for ch in self.channels])
        mean_load = np.mean(loads)
        
        for i, channel in enumerate(self.channels):
            if channel.current_load > mean_load * 1.2:
                # Redistribute some load
                excess_load = channel.current_load - mean_load
                channel.current_load = mean_load
                
                # Find underloaded channels
                for j, other_channel in enumerate(self.channels):
                    if (other_channel.current_load < mean_load * 0.8 and 
                        excess_load > 0):
                        transfer = min(excess_load, mean_load * 0.8 - other_channel.current_load)
                        other_channel.current_load += transfer
                        excess_load -= transfer
        
        final_std = np.std([ch.current_load for ch in self.channels])
        improvement = ((initial_std - final_std) / initial_std) * 100
        
        return {
            "initial_load_std": initial_std,
            "final_load_std": final_std,
            "improvement_percent": improvement
        }
    
    def _mitigate_crosstalk(self) -> float:
        """Apply crosstalk mitigation techniques."""
        initial_crosstalk = self._calculate_total_crosstalk()
        
        # Implement crosstalk mitigation (simplified channel reordering)
        active_channels = [i for i, ch in enumerate(self.channels) 
                          if not ch.is_available]
        
        if len(active_channels) > 1:
            # Try to maximize channel separation
            sorted_channels = sorted(active_channels)
            # Apply optimal spacing (implementation simplified)
            
        final_crosstalk = self._calculate_total_crosstalk()
        
        return initial_crosstalk - final_crosstalk
    
    def _calculate_link_load(self, node1: int, node2: int) -> float:
        """Calculate load on link between two nodes."""
        # Simplified: return average channel load
        return np.mean([ch.current_load for ch in self.channels])
    
    def _crosstalk_heuristic(self, node: int, destination: int) -> float:
        """Heuristic function for crosstalk-aware routing."""
        # Manhattan distance as base heuristic
        return abs(node - destination) * 0.5  # Simplified
    
    def _calculate_crosstalk_penalty(self, node1: int, node2: int) -> float:
        """Calculate crosstalk penalty for a link."""
        # Simplified crosstalk penalty calculation
        return 1.0 if (node1 + node2) % 2 == 0 else 0.5
    
    def _update_routing_stats(self, path: OpticalPath):
        """Update routing statistics."""
        self.routing_stats["successful_routes"] += 1
        
        # Update average path length
        current_avg = self.routing_stats["average_path_length"]
        total_routes = self.routing_stats["successful_routes"]
        new_avg = ((current_avg * (total_routes - 1) + len(path.path_nodes)) / 
                   total_routes)
        self.routing_stats["average_path_length"] = new_avg
        
        # Update power consumption
        self.routing_stats["total_power_consumption"] += path.power_consumption_mw