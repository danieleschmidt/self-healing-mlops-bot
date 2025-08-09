"""Distributed coordination and leader election for multi-instance deployments."""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

logger = logging.getLogger(__name__)


class NodeRole(Enum):
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"


class NodeState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    host: str
    port: int
    role: NodeRole
    state: NodeState
    last_heartbeat: datetime
    version: str
    capabilities: List[str]
    load: float  # 0.0 to 1.0
    active_executions: int
    metadata: Dict[str, Any]


@dataclass
class LeadershipBid:
    """Leadership bid for election process."""
    node_id: str
    bid_time: datetime
    priority: int
    term: int
    qualifications: Dict[str, float]


class DistributedCoordinator:
    """Distributed coordination system for multi-instance bot deployments."""
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        redis_client=None,
        election_timeout: int = 30,
        heartbeat_interval: int = 10
    ):
        self.node_id = node_id or self._generate_node_id()
        self.redis_client = redis_client
        self.election_timeout = election_timeout
        self.heartbeat_interval = heartbeat_interval
        
        # Node state
        self.role = NodeRole.FOLLOWER
        self.state = NodeState.HEALTHY
        self.current_term = 0
        self.leader_id: Optional[str] = None
        self.last_heartbeat_sent = datetime.utcnow()
        self.last_leader_heartbeat = datetime.utcnow()
        
        # Cluster state
        self.cluster_nodes: Dict[str, NodeInfo] = {}
        self.active_executions: Set[str] = set()
        self.load_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "active_tasks": 0,
            "queue_size": 0
        }
        
        # Coordination state
        self._coordination_active = False
        self._election_in_progress = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._election_task: Optional[asyncio.Task] = None
        
        # Event handlers
        self.leadership_gained_handlers: List[callable] = []
        self.leadership_lost_handlers: List[callable] = []
        self.node_joined_handlers: List[callable] = []
        self.node_left_handlers: List[callable] = []
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        import socket
        hostname = socket.gethostname()
        timestamp = str(int(time.time()))
        unique_str = f"{hostname}_{timestamp}_{uuid.uuid4().hex[:8]}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]
    
    async def start_coordination(self, host: str = "localhost", port: int = 8080):
        """Start distributed coordination."""
        try:
            if self._coordination_active:
                logger.warning("Coordination already active")
                return
            
            self._coordination_active = True
            
            # Register this node
            await self._register_node(host, port)
            
            # Start heartbeat loop
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Start monitoring cluster
            await self._discover_cluster()
            
            # Trigger initial election if no leader
            if not self.leader_id:
                await self._trigger_election()
            
            logger.info(f"Distributed coordination started for node {self.node_id}")
            
        except Exception as e:
            logger.exception(f"Error starting coordination: {e}")
            await self.stop_coordination()
            raise
    
    async def stop_coordination(self):
        """Stop distributed coordination."""
        try:
            self._coordination_active = False
            
            # Cancel tasks
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            if self._election_task:
                self._election_task.cancel()
                try:
                    await self._election_task
                except asyncio.CancelledError:
                    pass
            
            # Notify cluster of departure
            await self._unregister_node()
            
            # Trigger new election if this node was leader
            if self.role == NodeRole.LEADER:
                await self._trigger_election()
            
            logger.info(f"Distributed coordination stopped for node {self.node_id}")
            
        except Exception as e:
            logger.exception(f"Error stopping coordination: {e}")
    
    async def _register_node(self, host: str, port: int):
        """Register this node with the cluster."""
        try:
            node_info = NodeInfo(
                node_id=self.node_id,
                host=host,
                port=port,
                role=self.role,
                state=self.state,
                last_heartbeat=datetime.utcnow(),
                version="1.0.0",
                capabilities=["event_processing", "repair_execution", "monitoring"],
                load=self._calculate_load(),
                active_executions=len(self.active_executions),
                metadata={
                    "started_at": datetime.utcnow().isoformat(),
                    "process_id": str(uuid.uuid4())
                }
            )
            
            if self.redis_client:
                # Store in Redis with TTL
                key = f"bot_cluster:nodes:{self.node_id}"
                await self.redis_client.setex(
                    key,
                    self.heartbeat_interval * 3,  # 3x heartbeat interval TTL
                    json.dumps(asdict(node_info), default=str)
                )
            
            self.cluster_nodes[self.node_id] = node_info
            
            logger.info(f"Registered node {self.node_id} at {host}:{port}")
            
        except Exception as e:
            logger.exception(f"Error registering node: {e}")
            raise
    
    async def _unregister_node(self):
        """Unregister this node from the cluster."""
        try:
            if self.redis_client:
                key = f"bot_cluster:nodes:{self.node_id}"
                await self.redis_client.delete(key)
            
            if self.node_id in self.cluster_nodes:
                del self.cluster_nodes[self.node_id]
            
            logger.info(f"Unregistered node {self.node_id}")
            
        except Exception as e:
            logger.exception(f"Error unregistering node: {e}")
    
    async def _discover_cluster(self):
        """Discover other nodes in the cluster."""
        try:
            if not self.redis_client:
                return
            
            # Get all nodes
            pattern = "bot_cluster:nodes:*"
            keys = await self.redis_client.keys(pattern)
            
            discovered_nodes = {}
            
            for key in keys:
                try:
                    data = await self.redis_client.get(key)
                    if data:
                        node_data = json.loads(data)
                        # Convert datetime strings back to datetime objects
                        node_data["last_heartbeat"] = datetime.fromisoformat(
                            node_data["last_heartbeat"].replace("Z", "+00:00")
                        )
                        
                        node_info = NodeInfo(**node_data)
                        discovered_nodes[node_info.node_id] = node_info
                        
                        # Check if this node claims to be leader
                        if node_info.role == NodeRole.LEADER:
                            if not self.leader_id or self.leader_id != node_info.node_id:
                                self.leader_id = node_info.node_id
                                self.last_leader_heartbeat = datetime.utcnow()
                                logger.info(f"Discovered leader: {self.leader_id}")
                
                except Exception as e:
                    logger.warning(f"Error parsing node data from {key}: {e}")
            
            # Update cluster state
            self.cluster_nodes.update(discovered_nodes)
            
            # Notify about new nodes
            for node_id, node_info in discovered_nodes.items():
                if node_id != self.node_id:  # Don't notify about self
                    for handler in self.node_joined_handlers:
                        try:
                            await handler(node_info)
                        except Exception as e:
                            logger.exception(f"Error in node joined handler: {e}")
            
            logger.info(f"Discovered {len(discovered_nodes)} cluster nodes")
            
        except Exception as e:
            logger.exception(f"Error discovering cluster: {e}")
    
    async def _heartbeat_loop(self):
        """Main heartbeat loop."""
        while self._coordination_active:
            try:
                await self._send_heartbeat()
                await self._check_leader_health()
                await self._update_cluster_state()
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeat(self):
        """Send heartbeat to cluster."""
        try:
            if not self.redis_client:
                return
            
            # Update node info
            node_info = self.cluster_nodes.get(self.node_id)
            if node_info:
                node_info.last_heartbeat = datetime.utcnow()
                node_info.role = self.role
                node_info.state = self.state
                node_info.load = self._calculate_load()
                node_info.active_executions = len(self.active_executions)
                
                # Store updated info
                key = f"bot_cluster:nodes:{self.node_id}"
                await self.redis_client.setex(
                    key,
                    self.heartbeat_interval * 3,
                    json.dumps(asdict(node_info), default=str)
                )
            
            self.last_heartbeat_sent = datetime.utcnow()
            
            # If leader, send leader heartbeat
            if self.role == NodeRole.LEADER:
                leader_key = "bot_cluster:leader"
                leader_data = {
                    "node_id": self.node_id,
                    "term": self.current_term,
                    "heartbeat_time": datetime.utcnow().isoformat()
                }
                await self.redis_client.setex(
                    leader_key,
                    self.heartbeat_interval * 2,
                    json.dumps(leader_data)
                )
            
        except Exception as e:
            logger.exception(f"Error sending heartbeat: {e}")
    
    async def _check_leader_health(self):
        """Check if current leader is still healthy."""
        try:
            if self.role == NodeRole.LEADER:
                return  # Leaders don't check themselves
            
            if not self.redis_client:
                return
            
            # Check leader heartbeat
            leader_key = "bot_cluster:leader"
            leader_data = await self.redis_client.get(leader_key)
            
            if leader_data:
                leader_info = json.loads(leader_data)
                last_heartbeat = datetime.fromisoformat(
                    leader_info["heartbeat_time"].replace("Z", "+00:00")
                )
                
                time_since_heartbeat = (datetime.utcnow() - last_heartbeat).total_seconds()
                
                if time_since_heartbeat > self.election_timeout:
                    logger.warning(f"Leader {self.leader_id} heartbeat timeout ({time_since_heartbeat}s)")
                    self.leader_id = None
                    await self._trigger_election()
                else:
                    self.last_leader_heartbeat = datetime.utcnow()
            else:
                # No leader heartbeat found
                if self.leader_id:
                    logger.warning("Leader heartbeat not found, triggering election")
                    self.leader_id = None
                    await self._trigger_election()
        
        except Exception as e:
            logger.exception(f"Error checking leader health: {e}")
    
    async def _update_cluster_state(self):
        """Update cluster state and clean up offline nodes."""
        try:
            if not self.redis_client:
                return
            
            current_time = datetime.utcnow()
            offline_nodes = []
            
            # Check each known node
            for node_id, node_info in list(self.cluster_nodes.items()):
                if node_id == self.node_id:
                    continue  # Skip self
                
                # Check if node is still alive
                key = f"bot_cluster:nodes:{node_id}"
                data = await self.redis_client.get(key)
                
                if not data:
                    # Node is offline
                    offline_nodes.append(node_id)
                    continue
                
                # Update node info
                try:
                    updated_data = json.loads(data)
                    updated_data["last_heartbeat"] = datetime.fromisoformat(
                        updated_data["last_heartbeat"].replace("Z", "+00:00")
                    )
                    self.cluster_nodes[node_id] = NodeInfo(**updated_data)
                except Exception as e:
                    logger.warning(f"Error updating node {node_id} info: {e}")
            
            # Remove offline nodes
            for node_id in offline_nodes:
                del self.cluster_nodes[node_id]
                
                # Notify handlers
                for handler in self.node_left_handlers:
                    try:
                        await handler(node_id)
                    except Exception as e:
                        logger.exception(f"Error in node left handler: {e}")
                
                logger.info(f"Node {node_id} marked as offline")
        
        except Exception as e:
            logger.exception(f"Error updating cluster state: {e}")
    
    async def _trigger_election(self):
        """Trigger leader election."""
        try:
            if self._election_in_progress:
                logger.debug("Election already in progress")
                return
            
            if self.role == NodeRole.LEADER:
                logger.debug("Already leader, no election needed")
                return
            
            self._election_in_progress = True
            self.role = NodeRole.CANDIDATE
            
            logger.info(f"Starting leader election for term {self.current_term + 1}")
            
            self._election_task = asyncio.create_task(self._conduct_election())
            
        except Exception as e:
            logger.exception(f"Error triggering election: {e}")
            self._election_in_progress = False
    
    async def _conduct_election(self):
        """Conduct leader election using modified Raft algorithm."""
        try:
            self.current_term += 1
            
            # Create leadership bid
            bid = LeadershipBid(
                node_id=self.node_id,
                bid_time=datetime.utcnow(),
                priority=self._calculate_leadership_priority(),
                term=self.current_term,
                qualifications=self._get_leadership_qualifications()
            )
            
            # Submit bid to cluster
            if self.redis_client:
                bid_key = f"bot_cluster:election:{self.current_term}:bids:{self.node_id}"
                await self.redis_client.setex(
                    bid_key,
                    self.election_timeout,
                    json.dumps(asdict(bid), default=str)
                )
            
            # Wait for other bids
            await asyncio.sleep(self.election_timeout / 2)
            
            # Collect all bids
            bids = await self._collect_election_bids()
            
            # Determine winner
            winner_bid = self._determine_election_winner(bids)
            
            if winner_bid and winner_bid.node_id == self.node_id:
                # Won election
                await self._become_leader()
            else:
                # Lost election
                self.role = NodeRole.FOLLOWER
                if winner_bid:
                    self.leader_id = winner_bid.node_id
                    logger.info(f"Election completed, new leader: {self.leader_id}")
            
        except Exception as e:
            logger.exception(f"Error conducting election: {e}")
        finally:
            self._election_in_progress = False
    
    async def _collect_election_bids(self) -> List[LeadershipBid]:
        """Collect all election bids."""
        bids = []
        
        try:
            if not self.redis_client:
                return bids
            
            pattern = f"bot_cluster:election:{self.current_term}:bids:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                try:
                    data = await self.redis_client.get(key)
                    if data:
                        bid_data = json.loads(data)
                        bid_data["bid_time"] = datetime.fromisoformat(
                            bid_data["bid_time"].replace("Z", "+00:00")
                        )
                        bids.append(LeadershipBid(**bid_data))
                except Exception as e:
                    logger.warning(f"Error parsing bid from {key}: {e}")
            
            logger.info(f"Collected {len(bids)} election bids")
            
        except Exception as e:
            logger.exception(f"Error collecting election bids: {e}")
        
        return bids
    
    def _determine_election_winner(self, bids: List[LeadershipBid]) -> Optional[LeadershipBid]:
        """Determine election winner based on qualifications."""
        if not bids:
            return None
        
        # Sort by priority (highest first), then by qualifications, then by bid time
        def bid_score(bid):
            qual_score = sum(bid.qualifications.values())
            return (bid.priority, qual_score, -bid.bid_time.timestamp())
        
        sorted_bids = sorted(bids, key=bid_score, reverse=True)
        winner = sorted_bids[0]
        
        logger.info(f"Election winner: {winner.node_id} (priority: {winner.priority})")
        return winner
    
    async def _become_leader(self):
        """Become cluster leader."""
        try:
            self.role = NodeRole.LEADER
            self.leader_id = self.node_id
            
            # Announce leadership
            if self.redis_client:
                leader_key = "bot_cluster:leader"
                leader_data = {
                    "node_id": self.node_id,
                    "term": self.current_term,
                    "heartbeat_time": datetime.utcnow().isoformat(),
                    "elected_at": datetime.utcnow().isoformat()
                }
                await self.redis_client.setex(
                    leader_key,
                    self.heartbeat_interval * 2,
                    json.dumps(leader_data)
                )
            
            logger.info(f"Became cluster leader for term {self.current_term}")
            
            # Notify handlers
            for handler in self.leadership_gained_handlers:
                try:
                    await handler(self.current_term)
                except Exception as e:
                    logger.exception(f"Error in leadership gained handler: {e}")
        
        except Exception as e:
            logger.exception(f"Error becoming leader: {e}")
    
    def _calculate_leadership_priority(self) -> int:
        """Calculate leadership priority based on node capabilities."""
        priority = 100  # Base priority
        
        # Reduce priority based on load
        load = self._calculate_load()
        priority -= int(load * 50)
        
        # Increase priority based on capabilities
        capabilities = len(self.cluster_nodes.get(self.node_id, NodeInfo(
            node_id=self.node_id, host="", port=0, role=self.role,
            state=self.state, last_heartbeat=datetime.utcnow(),
            version="", capabilities=[], load=0.0, active_executions=0, metadata={}
        )).capabilities)
        priority += capabilities * 10
        
        # Adjust for health state
        if self.state == NodeState.HEALTHY:
            priority += 20
        elif self.state == NodeState.DEGRADED:
            priority -= 10
        elif self.state == NodeState.UNHEALTHY:
            priority -= 50
        
        return max(0, priority)
    
    def _get_leadership_qualifications(self) -> Dict[str, float]:
        """Get leadership qualifications."""
        return {
            "uptime": min((datetime.utcnow() - datetime.utcnow()).total_seconds() / 3600, 24.0),
            "success_rate": 0.95,  # Would be calculated from actual metrics
            "available_capacity": 1.0 - self._calculate_load(),
            "version_compatibility": 1.0,
            "network_stability": 0.98  # Would be measured
        }
    
    def _calculate_load(self) -> float:
        """Calculate current node load (0.0 to 1.0)."""
        # Combine different load factors
        cpu_weight = 0.4
        memory_weight = 0.3
        task_weight = 0.3
        
        cpu_load = self.load_metrics.get("cpu_usage", 0.0)
        memory_load = self.load_metrics.get("memory_usage", 0.0)
        task_load = min(self.load_metrics.get("active_tasks", 0) / 10.0, 1.0)  # Normalize to 10 max tasks
        
        total_load = (
            cpu_load * cpu_weight +
            memory_load * memory_weight +
            task_load * task_weight
        )
        
        return min(total_load, 1.0)
    
    # Public API methods
    
    def is_leader(self) -> bool:
        """Check if this node is the cluster leader."""
        return self.role == NodeRole.LEADER
    
    def get_leader_id(self) -> Optional[str]:
        """Get current cluster leader ID."""
        return self.leader_id
    
    def get_cluster_nodes(self) -> Dict[str, NodeInfo]:
        """Get information about all cluster nodes."""
        return self.cluster_nodes.copy()
    
    def get_cluster_size(self) -> int:
        """Get current cluster size."""
        return len(self.cluster_nodes)
    
    def update_load_metrics(self, metrics: Dict[str, float]):
        """Update node load metrics."""
        self.load_metrics.update(metrics)
    
    def add_active_execution(self, execution_id: str):
        """Add active execution to tracking."""
        self.active_executions.add(execution_id)
    
    def remove_active_execution(self, execution_id: str):
        """Remove active execution from tracking."""
        self.active_executions.discard(execution_id)
    
    def set_node_state(self, state: NodeState):
        """Set node health state."""
        if self.state != state:
            old_state = self.state
            self.state = state
            logger.info(f"Node state changed: {old_state.value} -> {state.value}")
    
    # Event handler registration
    
    def on_leadership_gained(self, handler: callable):
        """Register handler for leadership gained event."""
        self.leadership_gained_handlers.append(handler)
    
    def on_leadership_lost(self, handler: callable):
        """Register handler for leadership lost event."""
        self.leadership_lost_handlers.append(handler)
    
    def on_node_joined(self, handler: callable):
        """Register handler for node joined event."""
        self.node_joined_handlers.append(handler)
    
    def on_node_left(self, handler: callable):
        """Register handler for node left event."""
        self.node_left_handlers.append(handler)
    
    async def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status."""
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "state": self.state.value,
            "current_term": self.current_term,
            "leader_id": self.leader_id,
            "cluster_size": len(self.cluster_nodes),
            "active_executions": len(self.active_executions),
            "load": self._calculate_load(),
            "coordination_active": self._coordination_active,
            "election_in_progress": self._election_in_progress,
            "last_heartbeat_sent": self.last_heartbeat_sent.isoformat() if self.last_heartbeat_sent else None,
            "last_leader_heartbeat": self.last_leader_heartbeat.isoformat() if self.last_leader_heartbeat else None,
            "cluster_nodes": {
                node_id: {
                    "role": node.role.value,
                    "state": node.state.value,
                    "load": node.load,
                    "active_executions": node.active_executions,
                    "last_heartbeat": node.last_heartbeat.isoformat()
                }
                for node_id, node in self.cluster_nodes.items()
            }
        }