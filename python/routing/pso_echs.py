from hashlib import new
import logging, sys
import config as cf

from python.network.network import *
from python.routing.routing_protocol import *
from python.utils.utils import calculate_vector
from tqdm import tqdm

class PSO_EACHS(RoutingProtocol):   
    
    def __init__(self) -> None:
        super().__init__()
        self.params = {
            'swarm_size': 30,
            'numiteres': 10,
            'C1': 2.0,
            'C2': 2.0,
            'W': 0.7,
            'vmax': 200,
            'alpha': 0.3,
            'D': 30
        }
        self.pop = None
    
    def random_clustering(self, network):
        centroids = []
        alive_nodes = network.get_alive_nodes()
        # logging.info('PSO EACHS: deciding which nodes are cluster centroids.')
        idx = 0
        prob_ch = float(cf.NB_CLUSTERS)/float(cf.NB_NODES)
        while len(centroids) != cf.NB_CLUSTERS:
            node = alive_nodes[idx]
            u_random = np.random.uniform(0, 1)
            # node will be a cluster centroid
            if u_random < prob_ch:
                node.next_hop = cf.BSID
                centroids.append(node)
            idx = idx+1 if idx < len(alive_nodes)-1 else 0
            
        clusters = [[] for i in range(cf.NB_CLUSTERS)]
        for node in alive_nodes:
            if node in centroids:
                continue
            min_distance = cf.INFINITY
            min_distance_id = -1
            clusters[0].append(node)
            # find the nearest cluster centroid
            for i, centroid in enumerate(centroids):
                distance = calculate_distance(node, centroid)
                if min_distance > distance:
                    min_distance = distance
                    min_distance_id = i
                    
            clusters[min_distance_id].append(node)
        
        for i, cluster in enumerate(clusters):
            for node in cluster:    
                node.next_hop = centroids[i].id
                
        return {"centroid": centroids, "cluster": clusters}
        
    def setup_phase(self, network, round_nb=None):
        logging.info('PSO EACHS: setup phase.')
        # decide which network are cluster centroids
        pop = [self.random_clustering(network) for i in range(self.params['swarm_size'])]
        centroids, clusters = self.optimize(network, pop)
        network.broadcast_next_hop()
        
    def get_fitness(self, network, centroids, clusters):
        term_1 = 0
        term_2 = 0
        
        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue
            total_distance = 0
            for node in cluster:
                total_distance += calculate_distance(node, centroids[i])
            term_1 += (total_distance + len(cluster) * calculate_distance(centroids[i], network.get_BS())) \
                / (len(cluster) * 2 * (cf.AREA_LENGTH + cf.AREA_WIDTH))

        transform = lambda x: x.energy_source.energy
        energies = [transform(x) for x in centroids]
        term_2 = 1 / np.sum(energies)
        return term_1 * self.params['alpha'] + term_2 * (1 - self.params['alpha'])
    
    def update_velocity(self, v, local_best, global_best, current):
        vx, vy = v
        r1, r2 = np.random.uniform(0, 1, 2)
        vlx, vly = calculate_vector(local_best, current)
        vgx, vgy = calculate_vector(global_best, current)
        vx = self.params['W'] * vx + self.params['C1'] * r1 *  \
            + self.params['C2'] * r2 * vlx + self.params['C2'] * r2 * vgx
        vy = self.params['W'] * vy + self.params['C1'] * r1 *  \
            + self.params['C2'] * r2 * vly + self.params['C2'] * r2 * vgy
        if abs(vx) > self.params['vmax']:
            vx = self.params['vmax'] * np.sign(vx)
        if abs(vy) > self.params['vmax']:
            vy = self.params['vmax'] * np.sign(vy)
        return (vx, vy)
    
    def clip_position(self, position):
        x, y = position
        if x < 0:
            x = 0
        elif x > cf.AREA_LENGTH:
            x = cf.AREA_LENGTH
        if y < 0:
            y = 0
        elif y > cf.AREA_WIDTH:
            y = cf.AREA_WIDTH
        return (x, y)
    
    def optimize(self, network, pop):
        fitness = [self.get_fitness(network, particle['centroid'], particle['cluster']) for particle in pop]
        global_best = pop[np.argmin(fitness)]
        candidate_nodes = network.get_alive_nodes()
        for i in range(len(pop)):
            pop[i]['fitness'] = fitness[i]
            pop[i]['local_best'] = pop[i]
            pop[i]['velocity'] = [(0, 0) for i in range(len(pop[i]['cluster']))]
            
        for i in tqdm(range(self.params['numiteres'])):
            for j in range(self.params['swarm_size']):
                # update velocity
                new_centroids = []
                for k in range(len(pop[j]['cluster'])):
                    # update velocity
                    v = pop[j]['velocity'][k]
                    v = self.update_velocity(v, pop[j]['local_best']['centroid'][k], 
                                             global_best['centroid'][k], pop[j]['centroid'][k])
                    pop[j]['velocity'][k] = v
                    # update position
                    new_x = pop[j]['centroid'][k].pos_x + pop[j]['velocity'][k][0]
                    new_y = pop[j]['centroid'][k].pos_y + pop[j]['velocity'][k][1]
                    new_x, new_y = self.clip_position((new_x, new_y))
                    temp_node = Node(0)
                    temp_node.pos_x = new_x
                    temp_node.pos_y = new_y
                    min_distance = calculate_distance(pop[j]['centroid'][k], temp_node)
                    for node in candidate_nodes:
                        distance = calculate_distance(pop[j]['centroid'][k], node)
                        if min_distance > distance:
                            min_distance = distance
                            pop[j]['centroid'][k] = node
                            
                    new_centroids.append(pop[j]['centroid'][k])
                pop[j]['centroid'] = new_centroids
                pop[j]['cluster'] = self.get_cluster(network, pop[j]['centroid'])
                pop[j]['fitness'] = self.get_fitness(network, pop[j]['centroid'], pop[j]['cluster'])
                if pop[j]['fitness'] < pop[j]['local_best']['fitness']:
                    pop[j]['local_best'] = pop[j]
                if pop[j]['fitness'] < global_best['fitness']:
                    global_best = pop[j]
        return global_best['centroid'], global_best['cluster']
    
    def get_cluster(self, network, centroids):
        clusters = [[] for i in range(cf.NB_CLUSTERS)]
        for node in network.get_alive_nodes():
            min_distance = cf.INFINITY
            min_distance_id = -1
            for i, centroid in enumerate(centroids):
                distance = calculate_distance(node, centroid)
                if min_distance > distance:
                    min_distance = distance
                    min_distance_id = i
            clusters[min_distance_id].append(node)
        return clusters