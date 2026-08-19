#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENESIS CORE: The Unthought-of Algorithms
Հեղինակ: AI Architect
Նկարագրություն: Այս համակարգը պարունակում է 10 եզակի, նախկինում չտեսնված ալգորիթմական մոտեցումներ,
որոնք միավորում են քվանտային տրամաբանությունը, խաոսի տեսությունը, նեյրոպլաստիկությունը և էվոլյուցիոն կիբեռնետիկան։
Ոչ մի արտաքին գրադարկա (No external dependencies): Pure Python Logic.
"""

import math
import time
import random
import hashlib
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# ==============================================================================
# 1. QUANTUM ENTROPY SIMULATION (Քվանտային Էնտրոպիայի Իմիտացիա)
# ==============================================================================
class QuantumState:
    """
    Ներկայացնում է տվյալը որպես հավանային սուպերպոզիցիա, ոչ թե ֆիքսված արժեք։
    Չափումը (measure) փլուզում է վիճակը՝ հիմնվելով էնտրոպիայի և քվանտային փուլի վրա։
    """
    def __init__(self, data: Any, phase: float = 0.0):
        self.data = data
        self.probability_amplitude = 1.0
        self.phase = phase  # radians
        self.entropy = 0.0
        self._collapsed = False
        self.entangled_nodes: List['QuantumState'] = []

    def entangle(self, other: 'QuantumState') -> None:
        """Կապում է երկու քվանտային վիճակներ, փոխելով էնտրոպիան և փուլերը։"""
        combined_entropy = (self.entropy + other.entropy) / 2 + random.gauss(0, 0.05)
        self.entropy = max(0.01, min(1.0, combined_entropy))
        other.entropy = self.entropy
        avg_phase = (self.phase + other.phase) / 2
        self.phase = avg_phase
        other.phase = avg_phase
        if other not in self.entangled_nodes:
            self.entangled_nodes.append(other)
        if self not in other.entangled_nodes:
            other.entangled_nodes.append(self)

    def measure(self) -> Any:
        """Փլուզեցնում է վիճակը՝ վերադարձնելով կոնկրետ արժեք։"""
        if self._collapsed:
            return self.data
        
        # Էնտրոպիան և փուլը որոշում են աղավաղման աստիճանը
        noise_factor = math.sin(self.entropy * math.pi + self.phase)
        if isinstance(self.data, (int, float)):
            result = self.data * (1 + noise_factor * 0.1)
        else:
            result = self.data
            
        self._collapsed = True
        self.probability_amplitude = 0.0
        for node in self.entangled_nodes:
            if not node._collapsed:
                node.entropy = min(1.0, node.entropy + 0.05)
        return result

    def evolve(self) -> None:
        """Թարմացնում է էնտրոպիան և փուլը ժամանակի ընթացքում։"""
        self.entropy += random.uniform(-0.05, 0.05)
        self.entropy = max(0.0, min(1.0, self.entropy))
        self.phase = (self.phase + 0.1) % (2 * math.pi)


class QuantumFourierTransformSim:
    """
    1D Discrete Quantum Fourier Transform (QFT) simulation on state vectors.
    Computes exact phase transformations and superposition state distributions.
    """
    def __init__(self, num_qubits: int = 4):
        self.num_qubits = num_qubits
        self.dim = 1 << num_qubits

    def transform(self, state_vector: List[complex]) -> List[complex]:
        """Performs QFT on a complex state vector of length N=2^n."""
        N = len(state_vector)
        out = [0.0 + 0.0j] * N
        inv_sqrt_n = 1.0 / math.sqrt(N)
        for k in range(N):
            s = 0.0 + 0.0j
            for j in range(N):
                angle = 2 * math.pi * j * k / N
                s += state_vector[j] * complex(math.cos(angle), math.sin(angle))
            out[k] = s * inv_sqrt_n
        return out

    def inverse_transform(self, state_vector: List[complex]) -> List[complex]:
        """Performs Inverse QFT on a complex state vector."""
        N = len(state_vector)
        out = [0.0 + 0.0j] * N
        inv_sqrt_n = 1.0 / math.sqrt(N)
        for k in range(N):
            s = 0.0 + 0.0j
            for j in range(N):
                angle = -2 * math.pi * j * k / N
                s += state_vector[j] * complex(math.cos(angle), math.sin(angle))
            out[k] = s * inv_sqrt_n
        return out

# ==============================================================================
# 2. NEURO-PLASTIC GRAPH (Նեյրո-Պլաստիկ Գրաֆ)
# ==============================================================================
class NeuroNode:
    def __init__(self, node_id: str):
        self.id = node_id
        self.connections: Dict[str, float] = {} # neighbor_id -> weight
        self.activation_level = 0.0
        self.usage_count = 0

    def strengthen_path(self, neighbor_id: str) -> None:
        """Hebbian Learning: 'Cells that fire together, wire together'."""
        if neighbor_id in self.connections:
            self.connections[neighbor_id] = min(10.0, self.connections[neighbor_id] * 1.1)
        self.usage_count += 1
        self.activation_level = min(1.0, self.activation_level + 0.1)

    def weaken_unused(self) -> None:
        """Նվազեցնում է չօգտագործվող կապերի կշիռը։"""
        for neighbor in list(self.connections.keys()):
            self.connections[neighbor] *= 0.95
            if self.connections[neighbor] < 0.01:
                del self.connections[neighbor]

class NeuroPlasticGraph:
    def __init__(self):
        self.nodes: Dict[str, NeuroNode] = {}

    def add_node(self, node_id: str):
        if node_id not in self.nodes:
            self.nodes[node_id] = NeuroNode(node_id)

    def connect(self, u: str, v: str, initial_weight: float = 1.0):
        self.add_node(u)
        self.add_node(v)
        self.nodes[u].connections[v] = initial_weight
        self.nodes[v].connections[u] = initial_weight

    def traverse_adaptive(self, start: str, steps: int) -> List[str]:
        """Ճանապարհ, որը փոխվում է յուրաքանչյուր քայլից հետո։"""
        path = [start]
        current = start
        for _ in range(steps):
            if current not in self.nodes or not self.nodes[current].connections:
                break
            
            # Ընտրություն՝ հիմնված կշիռների և ակտիվացման վրա
            neighbors = self.nodes[current].connections
            best_next = max(neighbors, key=lambda k: neighbors[k] * (1 + self.nodes[k].activation_level))
            
            self.nodes[current].strengthen_path(best_next)
            self.nodes[best_next].strengthen_path(current)
            
            # Մաքրում (Pruning)
            self.nodes[current].weaken_unused()
            
            current = best_next
            path.append(current)
        return path

    def prune_synapses(self, threshold: float = 0.05) -> int:
        """Global synaptic pruning: removes connections below weight threshold."""
        pruned_count = 0
        for node in self.nodes.values():
            for neighbor, weight in list(node.connections.items()):
                if weight < threshold:
                    del node.connections[neighbor]
                    pruned_count += 1
        return pruned_count

    def find_optimal_synaptic_path(self, start: str, target: str) -> List[str]:
        """Finds path with maximum accumulated Hebbian weight using Dijkstra's algorithm."""
        if start not in self.nodes or target not in self.nodes:
            return []

        distances = {node_id: float('inf') for node_id in self.nodes}
        previous = {node_id: None for node_id in self.nodes}
        distances[start] = 0.0
        unvisited = set(self.nodes.keys())

        while unvisited:
            current = min(unvisited, key=lambda n: distances[n])
            if current == target or distances[current] == float('inf'):
                break
            unvisited.remove(current)

            for neighbor, weight in self.nodes[current].connections.items():
                if neighbor in unvisited:
                    # Inverse weight as distance metric
                    cost = 1.0 / max(0.001, weight)
                    alt = distances[current] + cost
                    if alt < distances[neighbor]:
                        distances[neighbor] = alt
                        previous[neighbor] = current

        path = []
        curr = target
        while curr:
            path.append(curr)
            curr = previous[curr]
        path.reverse()
        return path if path[0] == start else []

# ==============================================================================
# 3. CHAOS THEORY ENCODING (Խաոսի Տեսության Կոդավորում)
# ==============================================================================
class ChaosEncoder:
    """
    Օգտագործում է Լորենցի և Ռյոսլերի գրավչության հավասարումները տվյալների քաոսային քարտեզագրման համար։
    """
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01, system_type="lorenz"):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.system_type = system_type

    def step(self, x, y, z):
        if self.system_type == "rossler":
            a, b, c = 0.2, 0.2, 5.7
            dx = (-y - z) * self.dt
            dy = (x + a * y) * self.dt
            dz = (b + z * (x - c)) * self.dt
            return x + dx, y + dy, z + dz
        else:
            dx = self.sigma * (y - x) * self.dt
            dy = (x * (self.rho - z) - y) * self.dt
            dz = (x * y - self.beta * z) * self.dt
            return x + dx, y + dy, z + dz

    def lorenz_step(self, x, y, z):
        return self.step(x, y, z)

    def encode(self, message: str) -> List[Tuple[float, float, float]]:
        """Վերածում է տեքստը 3D խաոսային հետագծի։"""
        trajectory = []
        x, y, z = 0.1, 0.0, 0.0
        
        # Սկզբնական վիճակը կախված է հաշից
        seed = int(hashlib.md5(message.encode()).hexdigest()[:8], 16) % 1000
        x += seed * 0.0001

        for char in message:
            val = ord(char)
            # Փոփոխում ենք պարամետրերը՝ հիմնվելով սիմվոլի վրա
            self.rho = 28.0 + (val % 10) * 0.1
            
            # Կատարում ենք իտերացիաներ
            for _ in range(5):
                x, y, z = self.step(x, y, z)
            trajectory.append((x, y, z))
        
        return trajectory

    def decode_approx(self, trajectory: List[Tuple[float, float, float]]) -> str:
        """Փորձում է վերականգնել տեքստը (մոտավոր)։"""
        decoded = ""
        prev_x, prev_y, prev_z = 0.1, 0.0, 0.0
        for x, y, z in trajectory:
            dist = math.sqrt((x-prev_x)**2 + (y-prev_y)**2 + (z-prev_z)**2)
            char_code = int(dist * 100) % 128
            if 32 <= char_code <= 126:
                decoded += chr(char_code)
            else:
                decoded += '?'
            prev_x, prev_y, prev_z = x, y, z
        return decoded


# ==============================================================================
# 11. HYPER-DIMENSIONAL VECTOR ENGINE (Vector Symbolic Architecture - VSA)
# ==============================================================================
class HyperDimensionalVectorEngine:
    """
    10,000-D Bipolar Vector Symbolic Architecture (VSA).
    Implements symbolic binding (*), bundling (+), unbinding, and associative memory.
    """
    def __init__(self, dim: int = 10000, seed: int = 42):
        self.dim = dim
        self.rng = random.Random(seed)
        self.memory: Dict[str, List[int]] = {}

    def create_vector(self, name: str) -> List[int]:
        """Generates a random bipolar hypervector {-1, +1}."""
        vec = [1 if self.rng.random() > 0.5 else -1 for _ in range(self.dim)]
        self.memory[name] = vec
        return vec

    def bind(self, v1: List[int], v2: List[int]) -> List[int]:
        """Binding operation via Hadamard elementwise multiplication."""
        return [a * b for a, b in zip(v1, v2)]

    def bundle(self, vectors: List[List[int]]) -> List[int]:
        """Bundling operation via elementwise addition + majority thresholding."""
        summed = [sum(col) for col in zip(*vectors)]
        return [1 if s >= 0 else -1 for s in summed]

    def similarity(self, v1: List[int], v2: List[int]) -> float:
        """Cosine similarity between two bipolar hypervectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        return dot / float(self.dim)

    def query_memory(self, query_vec: List[int]) -> Tuple[str, float]:
        """Queries associative memory for the closest concept hypervector."""
        best_name = ""
        best_sim = -1.0
        for name, vec in self.memory.items():
            sim = self.similarity(query_vec, vec)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        return best_name, best_sim

# ==============================================================================
# 4. PSYCHOLOGICAL FEEDBACK LOOP (Հոգեբանական Հետադարձ Կապ)
# ==============================================================================
class EmotionalAnalyzer:
    """
    Վերլուծում է տեքստի «հուզական լարվածությունը» և փոխում ալգորիթմի վարքագիծը։
    """
    def __init__(self):
        self.mood_state = 0.5 # 0.0 (տխուր/դանդաղ) -> 1.0 (ուրախ/արագ)
        self.history = deque(maxlen=10)

    def analyze_sentiment(self, text: str) -> float:
        """Պարզ բառային հիմքով հուզական վերլուծություն։"""
        positive_words = {'good', 'great', 'love', 'fast', 'strong', 'light', 'yes', 'success'}
        negative_words = {'bad', 'slow', 'hate', 'weak', 'dark', 'no', 'fail', 'error'}
        
        tokens = text.lower().split()
        score = 0.5
        count = 0
        for t in tokens:
            if t in positive_words:
                score += 0.1
                count += 1
            elif t in negative_words:
                score -= 0.1
                count += 1
        
        if count == 0: return 0.5
        return max(0.0, min(1.0, score))

    def process_with_mood(self, data: Any, context: str) -> Any:
        """Մշակում է տվյալները՝ հաշվի առնելով ընթացիկ տրամադրությունը։"""
        sentiment = self.analyze_sentiment(context)
        self.history.append(sentiment)
        self.mood_state = sum(self.history) / len(self.history)

        if isinstance(data, list):
            if self.mood_state > 0.7:
                return sorted(data) # Դրական՝ կարգավորված
            elif self.mood_state < 0.3:
                return sorted(data, reverse=True) # Բացասական՝ հակառակ
            else:
                random.shuffle(data) # Չեզոք՝ խառը
        return data

# ==============================================================================
# 5. TEMPORAL CONTEXT WINDOW (Ժամանակային Կոնտեքստի Պատուհան)
# ==============================================================================
class TemporalBuffer:
    """
    Պահպանում է տվյալները ոչ թե FIFO, այլ «Կարևորության Խտության» հիման վրա։
    """
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.buffer: List[Dict] = []

    def add(self, item: Any, importance: float, timestamp: float):
        entry = {
            'item': item,
            'importance': importance,
            'timestamp': timestamp,
            'decay': importance # Ժամանակի հետ նվազում է
        }
        self.buffer.append(entry)
        self._prune()

    def _prune(self):
        # Հեռացնում ենք ամենացածր «կենսունակ» արժեքով տվյալները
        if len(self.buffer) > self.capacity:
            self.buffer.sort(key=lambda x: x['decay'], reverse=True)
            self.buffer = self.buffer[:self.capacity]

    def get_context(self) -> List[Any]:
        # Վերադարձնում է ըստ ներկայիս կարևորության
        current_time = time.time()
        for entry in self.buffer:
            age = current_time - entry['timestamp']
            entry['decay'] = entry['importance'] / (1 + age) # Ժամանակային նվազում
        
        self.buffer.sort(key=lambda x: x['decay'], reverse=True)
        return [e['item'] for e in self.buffer]

# ==============================================================================
# 6. GENETIC POETRY ENGINE (Գենետիկ Բանաստեղծություն)
# ==============================================================================
class GeneticPoet:
    def __init__(self, vocabulary: List[str]):
        self.vocab = vocabulary
        self.population_size = 20
        self.mutation_rate = 0.1

    def generate_line(self, length: int = 5) -> List[str]:
        return random.choices(self.vocab, k=length)

    def fitness(self, line: List[str]) -> float:
        # «Ֆիտնես»՝ հիմնված ռիթմի և կրկնությունների վրա
        score = 0
        for i in range(len(line) - 1):
            if line[i] == line[i+1]: score -= 1 # Կրկնությունները վատ են
            else: score += 1
        if len(set(line)) == len(line): score += 5 # Յուրահատկություն
        return score

    def evolve_poem(self, generations: int = 50) -> List[str]:
        population = [self.generate_line() for _ in range(self.population_size)]
        
        for _ in range(generations):
            # Ընտրություն
            population.sort(key=self.fitness, reverse=True)
            parents = population[:5]
            
            next_gen = parents.copy()
            while len(next_gen) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                # Խաչասերում (Crossover)
                point = random.randint(1, len(p1)-1)
                child = p1[:point] + p2[point:]
                
                # Մուտացիա
                if random.random() < self.mutation_rate:
                    idx = random.randint(0, len(child)-1)
                    child[idx] = random.choice(self.vocab)
                
                next_gen.append(child)
            population = next_gen
        
        best_line = max(population, key=self.fitness)
        return best_line

# ==============================================================================
# 7. FRACTAL MEMORY STRUCTURE (Ֆրակտալ Հիշողություն)
# ==============================================================================
class FractalMemoryNode:
    def __init__(self, data: Any, depth: int = 0, max_depth: int = 5):
        self.data = data
        self.depth = depth
        self.children: List['FractalMemoryNode'] = []
        self.max_depth = max_depth

    def store(self, data: Any):
        if self.depth >= self.max_depth:
            return # Հասել ենք սահմանին
        
        # Ստեղծում ենք ենթակառուցվածք (ինքնանմանություն)
        if not self.children:
            for i in range(3): # 3 ճյուղ
                self.children.append(FractalMemoryNode(None, self.depth + 1, self.max_depth))
        
        # Բաշխում ենք տվյալը ճյուղերի միջև (հեշավորում)
        hash_val = hash(str(data))
        target = self.children[hash_val % len(self.children)]
        target.store(data)

    def retrieve_pattern(self, pattern_depth: int) -> List[Any]:
        """Վերադարձնում է տվյալները կոնկրետ խորության վրա։"""
        results = []
        if self.depth == pattern_depth and self.data is not None:
            results.append(self.data)
        
        for child in self.children:
            results.extend(child.retrieve_pattern(pattern_depth))
        return results

# ==============================================================================
# 8. STOCHASTIC DECONSTRUCTION (Ստոխաստիկ Դեկոնստրուկցիա)
# ==============================================================================
class StochasticSolver:
    """
    Լուծում է բարդ օպտիմիզացիոն խնդիրներ՝ օգտագործելով «սառեցման» (annealing) նոր մեթոդ։
    """
    def __init__(self, initial_temp: float = 1000.0, cooling_rate: float = 0.995):
        self.temp = initial_temp
        self.cooling_rate = cooling_rate

    def solve(self, initial_state: List[int], cost_func) -> Tuple[List[int], float]:
        current_state = initial_state[:]
        current_cost = cost_func(current_state)
        best_state = current_state[:]
        best_cost = current_cost

        while self.temp > 1.0:
            # Պատահական հարևան վիճակ
            neighbor = current_state[:]
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
            neighbor_cost = cost_func(neighbor)
            
            # Ընդունման հավանականություն
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / self.temp):
                current_state = neighbor
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_state = current_state[:]
                    best_cost = current_cost
            
            self.temp *= self.cooling_rate
        
        return best_state, best_cost

# ==============================================================================
# 9. HOLOGRAPHIC DATA SUMMARIZATION (Հոլոգրաֆիկ Ամփոփում)
# ==============================================================================
class HolographicCompressor:
    """
    Սեղմում է տվյալները այնպես, որ ցանկացած մաս պարունակի ամբողջության մասին տեղեկություն։
    """
    def __init__(self, block_size: int = 4):
        self.block_size = block_size

    def project(self, data: str) -> List[str]:
        """Ստեղծում է «հոլոգրաֆիկ» բլոկեր։"""
        blocks = []
        full_hash = hashlib.sha256(data.encode()).hexdigest()
        
        for i in range(0, len(data), self.block_size):
            chunk = data[i:i+self.block_size]
            # Յուրաքանչյուր բլոկ պարունակում է տեղային տվյալ + գլոբալ հեշի մաս
            hologram_part = full_hash[i % len(full_hash): (i % len(full_hash)) + 8]
            blocks.append(f"{chunk}::{hologram_part}")
        
        return blocks

    def reconstruct_hint(self, block: str) -> str:
        """Մեկ բլոկից փորձում է տալ ակնարկ ամբողջական տվյալի մասին։"""
        parts = block.split('::')
        if len(parts) != 2: return "Invalid"
        local_data, global_hint = parts
        return f"Local: {local_data} | Global Signature Hint: {global_hint}..."

# ==============================================================================
# 10. EVOLUTIONARY CYBERNETIC KEY (Էվոլյուցիոն Կիբեռնետիկ Բանալի)
# ==============================================================================
class CyberKey:
    """
    Բանալի, որը մուտացիայի է ենթարկվում ամեն օգտագործումից հետո։
    """
    def __init__(self, seed: str):
        self.key_material = list(seed)
        self.generation = 0
        self.access_log = []

    def attempt_access(self, input_key: str) -> bool:
        current_key_str = "".join(self.key_material)
        success = input_key == current_key_str
        self.access_log.append((time.time(), success))
        
        if success:
            self.mutate() # Հաջող մուտքից հետո փոխվում է
        return success

    def mutate(self):
        """Էվոլյուցիոն փոփոխություն։"""
        self.generation += 1
        idx = random.randint(0, len(self.key_material) - 1)
        # Փոխում ենք մեկ սիմվոլ
        new_char = chr(ord(self.key_material[idx]) + random.choice([-1, 1]))
        if 32 <= ord(new_char) <= 126:
            self.key_material[idx] = new_char
        
        # Երբեմն ավելացնում ենք աղմուկ
        if random.random() < 0.3:
            self.key_material.insert(random.randint(0, len(self.key_material)), '#')

    def get_status(self) -> Dict:
        return {
            "generation": self.generation,
            "current_length": len(self.key_material),
            "access_attempts": len(self.access_log)
        }

# ==============================================================================
# MAIN EXECUTION & DEMONSTRATION
# ==============================================================================
def run_genesis_demonstration():
    print("="*60)
    print("GENESIS CORE: INITIATING UNTHOUGHT ALGORITHMS")
    print("="*60)

    # 1. Quantum Demo
    print("\n[1] Quantum Entropy Simulation:")
    q1 = QuantumState(100)
    q2 = QuantumState(200)
    q1.entangle(q2)
    q1.evolve()
    print(f"   Collapsed Value: {q1.measure()} (Entropy was: {q1.entropy:.4f})")

    # 2. Neuro Graph Demo
    print("\n[2] Neuro-Plastic Graph Traversal:")
    graph = NeuroPlasticGraph()
    for i in range(5): graph.add_node(f"N{i}")
    for i in range(4): graph.connect(f"N{i}", f"N{i+1}")
    path = graph.traverse_adaptive("N0", 10)
    print(f"   Adaptive Path: {' -> '.join(path)}")

    # 3. Chaos Encoder Demo
    print("\n[3] Chaos Theory Encoding:")
    encoder = ChaosEncoder()
    trajectory = encoder.encode("Hello World")
    print(f"   Original: 'Hello World'")
    print(f"   Encoded Points: {len(trajectory)} (First point: {trajectory[0]})")

    # 4. Psychological Loop Demo
    print("\n[4] Psychological Feedback Loop:")
    analyzer = EmotionalAnalyzer()
    data_list = [1, 2, 3, 4, 5]
    res1 = analyzer.process_with_mood(data_list, "This is bad and slow")
    res2 = analyzer.process_with_mood(data_list, "This is great and fast")
    print(f"   Negative Mood Result: {res1}")
    print(f"   Positive Mood Result: {res2}")

    # 5. Temporal Buffer Demo
    print("\n[5] Temporal Context Window:")
    buffer = TemporalBuffer(capacity=3)
    now = time.time()
    buffer.add("Old News", 0.2, now - 100)
    buffer.add("Breaking News", 0.9, now - 5)
    buffer.add("Medium News", 0.5, now - 50)
    context = buffer.get_context()
    print(f"   Prioritized Context: {context}")

    # 6. Genetic Poet Demo
    print("\n[6] Genetic Poetry Engine:")
    vocab = ["code", "dream", "light", "dark", "flow", "byte", "soul", "mind"]
    poet = GeneticPoet(vocab)
    poem_line = poet.evolve_poem(generations=100)
    print(f"   Evolved Line: {' '.join(poem_line)}")

    # 7. Fractal Memory Demo
    print("\n[7] Fractal Memory Structure:")
    mem_root = FractalMemoryNode("Root", max_depth=3)
    for i in range(100): mem_root.store(f"Data_{i}")
    level_2_data = mem_root.retrieve_pattern(2)
    print(f"   Retrieved from Depth 2: {len(level_2_data)} items (Sample: {level_2_data[:3]})")

    # 8. Stochastic Solver Demo
    print("\n[8] Stochastic Deconstruction (Traveling Salesman Mini):")
    solver = StochasticSolver()
    cities = list(range(10))
    random.shuffle(cities)
    def cost(state): return sum(abs(state[i] - state[i+1]) for i in range(len(state)-1))
    best_route, best_score = solver.solve(cities, cost)
    print(f"   Optimized Route Cost: {best_score} (Route: {best_route})")

    # 9. Holographic Demo
    print("\n[9] Holographic Data Summarization:")
    holo = HolographicCompressor()
    blocks = holo.project("SecretMessage")
    hint = holo.reconstruct_hint(blocks[0])
    print(f"   Block 0 Hint: {hint}")

    # 10. Cyber Key Demo
    print("\n[10] Evolutionary Cybernetic Key:")
    key = CyberKey("StartKey")
    initial = "".join(key.key_material)
    print(f"   Initial Key: {initial}")
    key.attempt_access(initial) # Success -> Mutates
    print(f"   Key after successful access: {''.join(key.key_material)}")
    print(f"   Status: Gen {key.get_status()['generation']}")

    print("\n" + "="*60)
    print("GENESIS CORE EXECUTION COMPLETE.")
    print("="*60)

if __name__ == "__main__":
    try:
        run_genesis_demonstration()
    except Exception as e:
        print(f"Critical System Anomaly: {e}")
        sys.exit(1)
