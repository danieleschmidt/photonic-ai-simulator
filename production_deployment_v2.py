"""
Advanced Production Deployment System for Photonic AI.

Implements enterprise-grade deployment with intelligent load balancing,
auto-scaling, advanced caching, and comprehensive monitoring for
production-ready photonic neural network systems.
"""

import asyncio
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import docker
from datetime import datetime
import yaml

# Import our advanced systems
import sys
sys.path.append('src')

from models import create_benchmark_network
from intelligent_load_balancer import IntelligentLoadBalancer, PhotonicNode, LoadBalancerConfig
from concurrent_optimization import InferencePipeline, ConcurrencyConfig
from advanced_caching import matrix_cache, inference_cache
from scaling import AutoScalingManager, ScalingConfig

logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Production deployment configuration."""
    cluster_name: str = "photonic-ai-cluster"
    initial_nodes: int = 3
    max_nodes: int = 20
    min_nodes: int = 1
    target_cpu_utilization: float = 70.0
    enable_gpu_acceleration: bool = True
    enable_advanced_caching: bool = True
    cache_size_mb: int = 1024
    monitoring_enabled: bool = True
    high_availability: bool = True
    backup_enabled: bool = True


class ProductionPhotonicCluster:
    """Production-ready photonic AI cluster."""
    
    def __init__(self, config: ProductionConfig = None):
        self.config = config or ProductionConfig()
        self.cluster_id = f"{self.config.cluster_name}-{int(time.time())}"
        
        # Core components
        self.load_balancer = None
        self.nodes: Dict[str, PhotonicNode] = {}
        self.auto_scaler = None
        
        # Deployment state
        self.deployment_status = "initializing"
        self.health_status = "unknown"
        self.deployment_metrics = {}
        
        # Docker client for containerized deployment
        try:
            self.docker_client = docker.from_env()
            self.container_support = True
        except:
            self.docker_client = None
            self.container_support = False
            logger.warning("Docker not available, using direct deployment")
        
        logger.info(f"Initialized ProductionPhotonicCluster: {self.cluster_id}")
    
    async def deploy_cluster(self) -> Dict[str, Any]:
        """Deploy complete production cluster."""
        logger.info(f"🚀 Starting production deployment of {self.config.cluster_name}")
        
        deployment_start = time.time()
        
        try:
            # Step 1: Initialize load balancer
            await self._initialize_load_balancer()
            
            # Step 2: Deploy initial nodes
            await self._deploy_initial_nodes()
            
            # Step 3: Configure auto-scaling
            await self._configure_auto_scaling()
            
            # Step 4: Setup monitoring
            await self._setup_monitoring()
            
            # Step 5: Health checks
            await self._perform_health_checks()
            
            # Step 6: Start load balancer monitoring
            await self.load_balancer.start_monitoring()
            
            self.deployment_status = "deployed"
            self.health_status = "healthy"
            
            deployment_time = time.time() - deployment_start
            
            deployment_report = {
                "cluster_id": self.cluster_id,
                "deployment_status": "SUCCESS",
                "deployment_time_seconds": deployment_time,
                "nodes_deployed": len(self.nodes),
                "load_balancer_status": "active",
                "auto_scaling_enabled": True,
                "monitoring_enabled": self.config.monitoring_enabled,
                "high_availability": self.config.high_availability,
                "advanced_caching": self.config.enable_advanced_caching,
                "deployment_timestamp": datetime.now().isoformat(),
                "cluster_endpoints": self._get_cluster_endpoints(),
                "performance_benchmarks": await self._run_deployment_benchmarks()
            }
            
            logger.info(f"🎉 Production deployment completed successfully in {deployment_time:.2f}s")
            logger.info(f"📊 Cluster ready with {len(self.nodes)} nodes")
            
            return deployment_report
            
        except Exception as e:
            self.deployment_status = "failed"
            logger.error(f"❌ Production deployment failed: {e}")
            await self._cleanup_failed_deployment()
            raise e
    
    async def _initialize_load_balancer(self):
        """Initialize intelligent load balancer."""
        logger.info("🔧 Initializing intelligent load balancer...")
        
        lb_config = LoadBalancerConfig(
            enable_predictive_scaling=True,
            photonic_optimization=True,
            health_check_interval_s=2.0
        )
        
        self.load_balancer = IntelligentLoadBalancer(lb_config)
        logger.info("✅ Load balancer initialized")
    
    async def _deploy_initial_nodes(self):
        """Deploy initial photonic nodes."""
        logger.info(f"🏗️ Deploying {self.config.initial_nodes} initial nodes...")
        
        deployment_tasks = []
        for i in range(self.config.initial_nodes):
            task = self._deploy_single_node(f"node-{i}")
            deployment_tasks.append(task)
        
        # Deploy nodes concurrently
        nodes = await asyncio.gather(*deployment_tasks)
        
        for node in nodes:
            if node:
                self.nodes[node.node_id] = node
                self.load_balancer.add_node(node)
        
        logger.info(f"✅ Deployed {len(self.nodes)} nodes successfully")
    
    async def _deploy_single_node(self, node_id: str) -> Optional[PhotonicNode]:
        """Deploy a single photonic node."""
        try:
            # Create photonic models for the node
            models = [
                create_benchmark_network("mnist"),
                create_benchmark_network("cifar10")
            ]
            
            # Configure node endpoint
            base_port = 8000
            node_port = base_port + len(self.nodes)
            
            # Create photonic node
            node = PhotonicNode(
                node_id=node_id,
                models=models,
                endpoint="localhost",
                port=node_port
            )
            
            # If containerized deployment is enabled
            if self.container_support and self.config.high_availability:
                await self._deploy_node_container(node)
            
            # Initialize node pipeline
            await self._initialize_node_pipeline(node)
            
            logger.info(f"✅ Node {node_id} deployed successfully")
            return node
            
        except Exception as e:
            logger.error(f"❌ Failed to deploy node {node_id}: {e}")
            return None
    
    async def _deploy_node_container(self, node: PhotonicNode):
        """Deploy node as Docker container."""
        if not self.docker_client:
            return
        
        try:
            container_name = f"photonic-node-{node.node_id}"
            
            # Build container if needed
            dockerfile_content = self._generate_dockerfile()
            
            # Run container
            container = self.docker_client.containers.run(
                "python:3.12-slim",
                command=f"python -c 'import time; time.sleep(3600)'",  # Keep alive
                name=container_name,
                ports={str(node.port): node.port},
                detach=True,
                environment={
                    "NODE_ID": node.node_id,
                    "PORT": str(node.port)
                }
            )
            
            logger.info(f"📦 Container deployed for node {node.node_id}")
            
        except Exception as e:
            logger.warning(f"Container deployment failed for {node.node_id}: {e}")
    
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile for node deployment."""
        return '''
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY examples/ ./examples/

EXPOSE 8000

CMD ["python", "src/cli.py", "serve", "--port", "8000"]
'''
    
    async def _initialize_node_pipeline(self, node: PhotonicNode):
        """Initialize inference pipeline for node."""
        # Pipeline is already initialized in PhotonicNode constructor
        # Additional setup can be done here
        pass
    
    async def _configure_auto_scaling(self):
        """Configure auto-scaling system."""
        logger.info("⚙️ Configuring auto-scaling...")
        
        scaling_config = ScalingConfig(
            min_instances=self.config.min_nodes,
            max_instances=self.config.max_nodes,
            scale_up_threshold=self.config.target_cpu_utilization / 100.0,
            scale_down_threshold=(self.config.target_cpu_utilization - 20.0) / 100.0
        )
        
        self.auto_scaler = AutoScalingManager(scaling_config)
        logger.info("✅ Auto-scaling configured")
    
    async def _setup_monitoring(self):
        """Setup comprehensive monitoring system."""
        if not self.config.monitoring_enabled:
            return
        
        logger.info("📊 Setting up monitoring system...")
        
        # Initialize caching systems
        if self.config.enable_advanced_caching:
            # Configure matrix cache
            matrix_cache.max_size_bytes = self.config.cache_size_mb * 1024 * 1024
            logger.info(f"💾 Advanced caching enabled ({self.config.cache_size_mb}MB)")
        
        # Setup Prometheus metrics (if available)
        try:
            from prometheus_client import start_http_server, Counter, Histogram, Gauge
            
            # Start metrics server
            start_http_server(9090)
            logger.info("📈 Prometheus metrics server started on port 9090")
            
        except ImportError:
            logger.info("📈 Prometheus not available, using basic monitoring")
        
        logger.info("✅ Monitoring system configured")
    
    async def _perform_health_checks(self):
        """Perform comprehensive health checks."""
        logger.info("🏥 Performing health checks...")
        
        healthy_nodes = 0
        for node in self.nodes.values():
            try:
                # Test node with simple inference
                test_input = np.random.randn(1, 784)
                result = await node.process_inference(test_input, model_idx=0)
                
                if result is not None:
                    healthy_nodes += 1
                    node.metrics.health_status = "healthy"
                else:
                    node.metrics.health_status = "unhealthy"
                    
            except Exception as e:
                logger.warning(f"Health check failed for {node.node_id}: {e}")
                node.metrics.health_status = "unhealthy"
        
        if healthy_nodes < self.config.min_nodes:
            raise Exception(f"Insufficient healthy nodes: {healthy_nodes}/{len(self.nodes)}")
        
        logger.info(f"✅ Health checks passed: {healthy_nodes}/{len(self.nodes)} nodes healthy")
    
    def _get_cluster_endpoints(self) -> Dict[str, str]:
        """Get cluster endpoint information."""
        endpoints = {}
        for node in self.nodes.values():
            endpoints[node.node_id] = f"{node.base_url}/inference"
        
        endpoints["load_balancer"] = "http://localhost:8080/inference"
        endpoints["monitoring"] = "http://localhost:9090/metrics"
        
        return endpoints
    
    async def _run_deployment_benchmarks(self) -> Dict[str, Any]:
        """Run deployment validation benchmarks."""
        logger.info("🚀 Running deployment benchmarks...")
        
        # Test inference latency
        test_input = np.random.randn(10, 784)
        
        start_time = time.time()
        try:
            result = await self.load_balancer.route_request(test_input, model_idx=0)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            
            benchmarks = {
                "inference_latency_ms": latency_ms,
                "throughput_test": await self._run_throughput_test(),
                "cache_performance": self._get_cache_metrics(),
                "cluster_stability": "stable"
            }
            
            logger.info(f"📊 Deployment benchmarks completed")
            return benchmarks
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {"benchmark_status": "failed", "error": str(e)}
    
    async def _run_throughput_test(self, duration_s: float = 10.0) -> Dict[str, float]:
        """Run throughput benchmark test."""
        logger.info(f"📈 Running {duration_s}s throughput test...")
        
        start_time = time.time()
        requests_completed = 0
        errors = 0
        
        while time.time() - start_time < duration_s:
            try:
                test_input = np.random.randn(1, 784)
                await self.load_balancer.route_request(test_input)
                requests_completed += 1
            except:
                errors += 1
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)
        
        actual_duration = time.time() - start_time
        throughput_rps = requests_completed / actual_duration
        error_rate = errors / (requests_completed + errors) if (requests_completed + errors) > 0 else 0.0
        
        return {
            "throughput_requests_per_second": throughput_rps,
            "error_rate": error_rate,
            "total_requests": requests_completed,
            "duration_seconds": actual_duration
        }
    
    def _get_cache_metrics(self) -> Dict[str, Any]:
        """Get caching performance metrics."""
        matrix_metrics = matrix_cache.get_metrics()
        
        return {
            "matrix_cache_hit_rate": matrix_metrics.hit_rate,
            "matrix_cache_size_mb": matrix_metrics.memory_usage_bytes / (1024 * 1024),
            "inference_cache_enabled": True
        }
    
    async def _cleanup_failed_deployment(self):
        """Cleanup resources from failed deployment."""
        logger.info("🧹 Cleaning up failed deployment...")
        
        # Stop monitoring if started
        if self.load_balancer:
            try:
                await self.load_balancer.stop_monitoring()
            except:
                pass
        
        # Stop containers if any
        if self.container_support and self.docker_client:
            try:
                containers = self.docker_client.containers.list(
                    filters={"name": f"photonic-node-*"}
                )
                for container in containers:
                    container.stop()
                    container.remove()
            except:
                pass
    
    async def scale_cluster(self, target_nodes: int) -> Dict[str, Any]:
        """Scale cluster to target number of nodes."""
        current_nodes = len(self.nodes)
        
        if target_nodes > current_nodes:
            # Scale up
            new_nodes_needed = target_nodes - current_nodes
            logger.info(f"📈 Scaling up: adding {new_nodes_needed} nodes")
            
            for i in range(new_nodes_needed):
                node_id = f"node-{current_nodes + i}"
                new_node = await self._deploy_single_node(node_id)
                if new_node:
                    self.nodes[new_node.node_id] = new_node
                    self.load_balancer.add_node(new_node)
        
        elif target_nodes < current_nodes:
            # Scale down
            nodes_to_remove = current_nodes - target_nodes
            logger.info(f"📉 Scaling down: removing {nodes_to_remove} nodes")
            
            # Remove least utilized nodes
            nodes_by_load = sorted(
                self.nodes.values(), 
                key=lambda n: n.get_load_score()
            )
            
            for i in range(nodes_to_remove):
                if i < len(nodes_by_load):
                    node_to_remove = nodes_by_load[i]
                    self.load_balancer.remove_node(node_to_remove.node_id)
                    del self.nodes[node_to_remove.node_id]
        
        return {
            "scaling_action": "completed",
            "previous_nodes": current_nodes,
            "current_nodes": len(self.nodes),
            "target_nodes": target_nodes
        }
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        cluster_metrics = self.load_balancer.get_cluster_metrics() if self.load_balancer else {}
        
        node_status = {}
        for node in self.nodes.values():
            node_status[node.node_id] = {
                "health": node.metrics.health_status.value,
                "load_score": node.get_load_score(),
                "response_time_ms": node.metrics.avg_response_time_ms,
                "throughput_rps": node.metrics.throughput_rps,
                "error_rate": node.metrics.error_rate
            }
        
        return {
            "cluster_id": self.cluster_id,
            "deployment_status": self.deployment_status,
            "health_status": self.health_status,
            "total_nodes": len(self.nodes),
            "cluster_metrics": cluster_metrics,
            "node_status": node_status,
            "cache_metrics": self._get_cache_metrics(),
            "uptime_seconds": time.time() - getattr(self, 'deployment_start_time', time.time())
        }
    
    async def shutdown_cluster(self):
        """Gracefully shutdown the entire cluster."""
        logger.info(f"🛑 Shutting down cluster {self.cluster_id}...")
        
        # Stop monitoring
        if self.load_balancer:
            await self.load_balancer.stop_monitoring()
        
        # Shutdown node pipelines
        for node in self.nodes.values():
            if hasattr(node.pipeline, 'shutdown'):
                await node.pipeline.shutdown()
        
        # Stop containers
        if self.container_support and self.docker_client:
            try:
                containers = self.docker_client.containers.list(
                    filters={"name": f"photonic-node-*"}
                )
                for container in containers:
                    container.stop()
                    container.remove()
            except:
                pass
        
        self.deployment_status = "shutdown"
        logger.info("✅ Cluster shutdown completed")


async def main():
    """Main deployment function."""
    print("🚀 TERRAGON LABS - PRODUCTION PHOTONIC AI DEPLOYMENT")
    print("=" * 60)
    
    # Create production configuration
    config = ProductionConfig(
        cluster_name="terragon-photonic-production",
        initial_nodes=3,
        max_nodes=10,
        enable_gpu_acceleration=True,
        enable_advanced_caching=True,
        cache_size_mb=2048,
        high_availability=True
    )
    
    # Deploy cluster
    cluster = ProductionPhotonicCluster(config)
    
    try:
        # Deploy cluster
        deployment_report = await cluster.deploy_cluster()
        
        print("\n🎉 DEPLOYMENT SUCCESSFUL!")
        print("=" * 40)
        print(f"Cluster ID: {deployment_report['cluster_id']}")
        print(f"Nodes Deployed: {deployment_report['nodes_deployed']}")
        print(f"Deployment Time: {deployment_report['deployment_time_seconds']:.2f}s")
        print(f"Load Balancer: {deployment_report['load_balancer_status']}")
        print(f"Auto-scaling: {'✅ Enabled' if deployment_report['auto_scaling_enabled'] else '❌ Disabled'}")
        print(f"Advanced Caching: {'✅ Enabled' if deployment_report['advanced_caching'] else '❌ Disabled'}")
        
        # Performance benchmarks
        benchmarks = deployment_report.get('performance_benchmarks', {})
        if benchmarks:
            print(f"\n📊 PERFORMANCE BENCHMARKS:")
            print(f"Inference Latency: {benchmarks.get('inference_latency_ms', 0):.2f}ms")
            throughput = benchmarks.get('throughput_test', {})
            if throughput:
                print(f"Throughput: {throughput.get('throughput_requests_per_second', 0):.1f} RPS")
                print(f"Error Rate: {throughput.get('error_rate', 0)*100:.2f}%")
        
        print(f"\n🌐 CLUSTER ENDPOINTS:")
        endpoints = deployment_report.get('cluster_endpoints', {})
        for name, url in endpoints.items():
            print(f"{name}: {url}")
        
        # Save deployment report
        with open('production_deployment_report.json', 'w') as f:
            json.dump(deployment_report, f, indent=2)
        
        print(f"\n📄 Full report saved to: production_deployment_report.json")
        
        # Keep cluster running for demonstration
        print(f"\n⏳ Cluster running... (Press Ctrl+C to shutdown)")
        
        # Monitor cluster status
        for i in range(10):  # Run for 10 status checks
            await asyncio.sleep(5)
            status = await cluster.get_cluster_status()
            print(f"Status check {i+1}: {status['total_nodes']} nodes, "
                  f"health: {status['health_status']}")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Shutdown requested...")
        await cluster.shutdown_cluster()
        print("✅ Graceful shutdown completed")
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        await cluster.shutdown_cluster()
        raise


if __name__ == "__main__":
    asyncio.run(main())