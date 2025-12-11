"""
Thread Management System for Autonomous Agent

Provides centralized thread monitoring, management, and optimization for the
autonomous agent system. Tracks all threads, monitors performance, and provides
thread pool management with safety features.
"""

import threading
import time
import psutil
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ThreadState(Enum):
    """Thread state enumeration"""
    CREATED = "created"
    RUNNING = "running"
    WAITING = "waiting"
    BLOCKED = "blocked"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


@dataclass
class ThreadInfo:
    """Information about a managed thread"""
    thread_id: int
    name: str
    thread_type: str  # 'session', 'monitoring', 'dashboard', 'hardware', 'tool', 'custom'
    state: ThreadState
    created_time: datetime
    cpu_time: float = 0.0
    memory_usage: int = 0  # bytes
    is_daemon: bool = True
    is_alive: bool = True
    target_function: Optional[str] = None
    parent_component: Optional[str] = None
    last_activity: Optional[datetime] = None
    error_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    _thread_object: Optional[object] = field(default=None, repr=False)


class ThreadManager:
    """
    Centralized thread management system with monitoring and optimization.
    
    Features:
    - Thread registration and tracking
    - Performance monitoring
    - Resource usage analysis
    - Thread pool management
    - Safety controls and limits
    - Automatic cleanup
    """
    
    def __init__(self, max_threads: int = 50):
        self.max_threads = max_threads
        self.threads: Dict[int, ThreadInfo] = {}
        self.thread_pools: Dict[str, Dict[str, Any]] = {}  # Track ThreadPoolExecutors
        
        # Monitoring
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.monitoring_interval = 1.0  # seconds
        
        # Statistics
        self.stats = {
            'total_created': 0,
            'total_terminated': 0,
            'peak_concurrent': 0,
            'average_lifetime': 0.0,
            'total_cpu_time': 0.0,
            'total_memory_peak': 0,
            'error_count': 0
        }
        
        # Callbacks for monitoring
        self.thread_created_callbacks: List[Callable[[ThreadInfo], None]] = []
        self.thread_terminated_callbacks: List[Callable[[ThreadInfo], None]] = []
        self.performance_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Safety limits
        self.cpu_limit_percent = 80.0
        self.memory_limit_mb = 1024
        self.thread_timeout_seconds = 300  # 5 minutes default
        
        # Start monitoring
        self.start_monitoring()
    
    def register_thread(self, thread: threading.Thread, thread_type: str = "custom", 
                       parent_component: Optional[str] = None) -> ThreadInfo:
        """
        Register a thread for monitoring and management.
        
        Args:
            thread: Thread to register
            thread_type: Type of thread (session, monitoring, dashboard, etc.)
            parent_component: Component that created the thread
            
        Returns:
            ThreadInfo object for the registered thread
        """
        # Use thread object id as key since ident might be None before start
        thread_key = id(thread)
        
        thread_info = ThreadInfo(
            thread_id=thread_key,
            name=thread.name,
            thread_type=thread_type,
            state=ThreadState.CREATED,
            created_time=datetime.now(),
            is_daemon=thread.daemon,
            target_function=getattr(thread._target, '__name__', None) if hasattr(thread, '_target') else None,
            parent_component=parent_component
        )
        
        # Store the actual thread object for later reference
        thread_info._thread_object = thread
        
        self.threads[thread_key] = thread_info
        self.stats['total_created'] += 1
        
        # Update peak concurrent
        active_count = len([t for t in self.threads.values() if t.is_alive])
        if active_count > self.stats['peak_concurrent']:
            self.stats['peak_concurrent'] = active_count
        
        # Notify callbacks
        for callback in self.thread_created_callbacks:
            try:
                callback(thread_info)
            except Exception as e:
                logger.error(f"Thread created callback error: {e}")
        
        logger.info(f"Registered thread: {thread_info.name} ({thread_type})")
        return thread_info
    
    def unregister_thread(self, thread_id: int) -> bool:
        """
        Unregister a thread when it terminates.
        
        Args:
            thread_id: ID of thread to unregister
            
        Returns:
            True if thread was found and unregistered
        """
        if thread_id in self.threads:
            thread_info = self.threads[thread_id]
            thread_info.state = ThreadState.TERMINATED
            thread_info.is_alive = False
            
            # Calculate lifetime
            lifetime = (datetime.now() - thread_info.created_time).total_seconds()
            
            # Update statistics
            self.stats['total_terminated'] += 1
            total_lifetimes = self.stats['average_lifetime'] * (self.stats['total_terminated'] - 1)
            self.stats['average_lifetime'] = (total_lifetimes + lifetime) / self.stats['total_terminated']
            
            # Notify callbacks
            for callback in self.thread_terminated_callbacks:
                try:
                    callback(thread_info)
                except Exception as e:
                    logger.error(f"Thread terminated callback error: {e}")
            
            logger.info(f"Unregistered thread: {thread_info.name} (lifetime: {lifetime:.2f}s)")
            return True
        
        return False
    
    def get_thread_info(self, thread_id: int) -> Optional[ThreadInfo]:
        """Get information about a specific thread"""
        return self.threads.get(thread_id)
    
    def get_all_threads(self) -> List[ThreadInfo]:
        """Get information about all registered threads"""
        return list(self.threads.values())
    
    def get_active_threads(self) -> List[ThreadInfo]:
        """Get information about currently active threads"""
        return [t for t in self.threads.values() if t.is_alive]
    
    def get_threads_by_type(self, thread_type: str) -> List[ThreadInfo]:
        """Get threads of a specific type"""
        return [t for t in self.threads.values() if t.thread_type == thread_type]
    
    def start_monitoring(self):
        """Start the thread monitoring system"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.name = "ThreadManager-Monitor"
            
            # Register the monitoring thread itself
            monitor_info = ThreadInfo(
                thread_id=id(self.monitor_thread),
                name=self.monitor_thread.name,
                thread_type="monitoring",
                state=ThreadState.CREATED,
                created_time=datetime.now(),
                is_daemon=True,
                target_function="_monitoring_loop",
                parent_component="ThreadManager",
                _thread_object=self.monitor_thread
            )
            self.threads[id(self.monitor_thread)] = monitor_info
            self.stats['total_created'] += 1
            
            self.monitor_thread.start()
            logger.info("Thread monitoring started")
    
    def stop_monitoring(self):
        """Stop the thread monitoring system"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("Thread monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._update_thread_states()
                self._update_thread_pool_stats()
                self._check_resource_usage()
                self._cleanup_terminated_threads()
                self._update_performance_metrics()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Thread monitoring error: {e}")
                time.sleep(1)
    
    def _update_thread_states(self):
        """Update the state of all registered threads"""
        for thread_id, thread_info in self.threads.items():
            if hasattr(thread_info, '_thread_object'):
                thread = thread_info._thread_object
                thread_info.is_alive = thread.is_alive()
                
                # Update state based on thread status
                if not thread_info.is_alive:
                    thread_info.state = ThreadState.TERMINATED
                elif thread_info.state == ThreadState.CREATED and thread.is_alive():
                    thread_info.state = ThreadState.RUNNING
                
                thread_info.last_activity = datetime.now()
            else:
                # No thread object reference, assume terminated
                thread_info.is_alive = False
                thread_info.state = ThreadState.TERMINATED
    
    def _check_resource_usage(self):
        """Check resource usage of threads and enforce limits"""
        try:
            process = psutil.Process()
            
            # Get per-thread CPU times (if available)
            try:
                thread_times = process.threads()
                thread_cpu_map = {t.id: t.user_time + t.system_time for t in thread_times}
                
                for thread_id, thread_info in self.threads.items():
                    if thread_id in thread_cpu_map:
                        thread_info.cpu_time = thread_cpu_map[thread_id]
            except (AttributeError, psutil.AccessDenied):
                # Per-thread CPU times not available on this platform
                pass
            
            # Check overall process limits
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if cpu_percent > self.cpu_limit_percent:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}% (limit: {self.cpu_limit_percent}%)")
                self._handle_high_resource_usage("cpu", cpu_percent)
            
            if memory_mb > self.memory_limit_mb:
                logger.warning(f"High memory usage: {memory_mb:.1f}MB (limit: {self.memory_limit_mb}MB)")
                self._handle_high_resource_usage("memory", memory_mb)
            
        except Exception as e:
            logger.error(f"Resource usage check error: {e}")
    
    def _handle_high_resource_usage(self, resource_type: str, current_value: float):
        """Handle high resource usage situations"""
        # Log the situation
        active_threads = self.get_active_threads()
        logger.warning(f"High {resource_type} usage detected with {len(active_threads)} active threads")
        
        # Notify performance callbacks
        for callback in self.performance_callbacks:
            try:
                callback({
                    'type': 'high_resource_usage',
                    'resource': resource_type,
                    'value': current_value,
                    'active_threads': len(active_threads),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Performance callback error: {e}")
    
    def _cleanup_terminated_threads(self):
        """Remove terminated threads from tracking"""
        terminated_ids = [
            thread_id for thread_id, thread_info in self.threads.items()
            if thread_info.state == ThreadState.TERMINATED and 
            (datetime.now() - thread_info.created_time).total_seconds() > 300  # Keep for 5 minutes
        ]
        
        for thread_id in terminated_ids:
            del self.threads[thread_id]
    
    def _update_performance_metrics(self):
        """Update overall performance metrics"""
        active_threads = self.get_active_threads()
        
        # Update statistics
        self.stats['total_cpu_time'] = sum(t.cpu_time for t in self.threads.values())
        
        # Notify performance callbacks
        metrics = {
            'active_threads': len(active_threads),
            'total_threads': len(self.threads),
            'threads_by_type': {
                thread_type: len(self.get_threads_by_type(thread_type))
                for thread_type in set(t.thread_type for t in self.threads.values())
            },
            'average_lifetime': self.stats['average_lifetime'],
            'peak_concurrent': self.stats['peak_concurrent'],
            'timestamp': datetime.now().isoformat()
        }
        
        for callback in self.performance_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Performance metrics callback error: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        active_threads = self.get_active_threads()
        thread_pool_summary = self.get_thread_pool_summary()
        
        return {
            'thread_counts': {
                'active': len(active_threads),
                'total_created': self.stats['total_created'],
                'total_terminated': self.stats['total_terminated'],
                'peak_concurrent': self.stats['peak_concurrent']
            },
            'threads_by_type': {
                thread_type: len(self.get_threads_by_type(thread_type))
                for thread_type in set(t.thread_type for t in self.threads.values())
            },
            'thread_pools': thread_pool_summary,
            'performance': {
                'average_lifetime': self.stats['average_lifetime'],
                'total_cpu_time': self.stats['total_cpu_time'],
                'error_count': self.stats['error_count']
            },
            'resource_limits': {
                'max_threads': self.max_threads,
                'cpu_limit_percent': self.cpu_limit_percent,
                'memory_limit_mb': self.memory_limit_mb,
                'thread_timeout_seconds': self.thread_timeout_seconds
            },
            'current_threads': [
                {
                    'id': t.thread_id,
                    'name': t.name,
                    'type': t.thread_type,
                    'state': t.state.value,
                    'lifetime': (datetime.now() - t.created_time).total_seconds(),
                    'cpu_time': t.cpu_time,
                    'parent': t.parent_component
                }
                for t in active_threads
            ]
        }
    
    def emergency_terminate_all(self, exclude_types: Optional[List[str]] = None):
        """Emergency termination of all threads (except excluded types)"""
        exclude_types = exclude_types or ['monitoring']
        
        logger.warning("Emergency thread termination initiated")
        
        for thread_info in self.get_active_threads():
            if thread_info.thread_type not in exclude_types:
                try:
                    # Find the actual thread object
                    for thread in threading.enumerate():
                        if thread.ident == thread_info.thread_id:
                            logger.warning(f"Force terminating thread: {thread_info.name}")
                            # Note: Python doesn't have thread.terminate(), so we rely on
                            # the threads checking their stop events or other mechanisms
                            break
                except Exception as e:
                    logger.error(f"Failed to terminate thread {thread_info.name}: {e}")
    
    def add_thread_created_callback(self, callback: Callable[[ThreadInfo], None]):
        """Add callback for thread creation events"""
        self.thread_created_callbacks.append(callback)
    
    def add_thread_terminated_callback(self, callback: Callable[[ThreadInfo], None]):
        """Add callback for thread termination events"""
        self.thread_terminated_callbacks.append(callback)
    
    def add_performance_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for performance metrics updates"""
        self.performance_callbacks.append(callback)
    
    def register_thread_pool(self, pool: ThreadPoolExecutor, name: str, 
                           component: str, pool_type: str = "tool") -> None:
        """
        Register a ThreadPoolExecutor for monitoring.
        
        Args:
            pool: ThreadPoolExecutor to monitor
            name: Name identifier for the pool
            component: Component that owns the pool
            pool_type: Type of pool (tool, general, etc.)
        """
        pool_info = {
            'executor': pool,
            'name': name,
            'component': component,
            'pool_type': pool_type,
            'registered_time': datetime.now(),
            'max_workers': pool._max_workers,
            'active_count': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_submitted': 0
        }
        
        self.thread_pools[name] = pool_info
        logger.info(f"Registered thread pool: {name} ({pool_type}) with {pool._max_workers} max workers")
    
    def get_thread_pool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered thread pool"""
        return self.thread_pools.get(name)
    
    def get_all_thread_pools(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered thread pools"""
        return self.thread_pools.copy()
    
    def _update_thread_pool_stats(self):
        """Update statistics for all registered thread pools"""
        for pool_name, pool_info in self.thread_pools.items():
            try:
                executor = pool_info['executor']
                
                # Get current thread count from the executor's internal state
                if hasattr(executor, '_threads'):
                    pool_info['active_count'] = len(executor._threads)
                else:
                    # Fallback: count threads with the pool name pattern
                    pool_threads = [t for t in threading.enumerate() 
                                  if pool_name.lower() in t.name.lower() or 'ThreadPoolExecutor' in t.name]
                    pool_info['active_count'] = len(pool_threads)
                
                # Update last seen time
                pool_info['last_update'] = datetime.now()
                
            except Exception as e:
                logger.error(f"Error updating thread pool stats for {pool_name}: {e}")
    
    def get_thread_pool_summary(self) -> Dict[str, Any]:
        """Get summary of all thread pool activity"""
        total_pools = len(self.thread_pools)
        total_max_workers = sum(pool['max_workers'] for pool in self.thread_pools.values())
        total_active = sum(pool.get('active_count', 0) for pool in self.thread_pools.values())
        
        pools_by_type = {}
        for pool_info in self.thread_pools.values():
            pool_type = pool_info['pool_type']
            if pool_type not in pools_by_type:
                pools_by_type[pool_type] = {'count': 0, 'active_workers': 0, 'max_workers': 0}
            
            pools_by_type[pool_type]['count'] += 1
            pools_by_type[pool_type]['active_workers'] += pool_info.get('active_count', 0)
            pools_by_type[pool_type]['max_workers'] += pool_info['max_workers']
        
        return {
            'total_pools': total_pools,
            'total_max_workers': total_max_workers,
            'total_active_workers': total_active,
            'pools_by_type': pools_by_type,
            'individual_pools': {
                name: {
                    'name': info['name'],
                    'component': info['component'],
                    'pool_type': info['pool_type'],
                    'max_workers': info['max_workers'],
                    'active_count': info.get('active_count', 0),
                    'utilization': (info.get('active_count', 0) / info['max_workers']) * 100 if info['max_workers'] > 0 else 0
                }
                for name, info in self.thread_pools.items()
            }
        }


# Global thread manager instance
_thread_manager = None

def get_thread_manager() -> ThreadManager:
    """Get the global thread manager instance"""
    global _thread_manager
    if _thread_manager is None:
        _thread_manager = ThreadManager()
    return _thread_manager


def register_thread(thread: threading.Thread, thread_type: str = "custom", 
                   parent_component: Optional[str] = None) -> ThreadInfo:
    """Convenience function to register a thread with the global manager"""
    return get_thread_manager().register_thread(thread, thread_type, parent_component)


def unregister_thread(thread_id: int) -> bool:
    """Convenience function to unregister a thread from the global manager"""
    return get_thread_manager().unregister_thread(thread_id)


def register_thread_pool(pool: ThreadPoolExecutor, name: str, 
                        component: str, pool_type: str = "tool") -> None:
    """Convenience function to register a thread pool with the global manager"""
    return get_thread_manager().register_thread_pool(pool, name, component, pool_type)