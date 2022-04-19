# Temporal Attention Module
import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer, get_neighbor_finder
from utils.data_processing import Data
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode
# Temporal Attention Module can be used for the Temporal Attention Mechanism but without Memory.

"""
Helper functions:
"""
device = 'cpu'
# To se Memory, TGN-compute_temporal_embedding can be adapted.


class TGAT(torch.nn.Module):
    # Why we init a network but need to input the data which are processed? They are utilized in Memory and should be
    #  input when the memory is initialized.
    def __init__(self, node_feat_dim, edge_feat_dim,#node_features, edge_features,
                 device, n_layers=2,  # node_features and edge features need to be input
                 n_heads=2, dropout=0.1, use_memory=False,
                 memory_update_at_start=True, message_dimension=100,
                 memory_dimension=500, embedding_module_type="graph_attention",
                 message_function="mlp",
                 mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                 std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
                 memory_updater_type="gru",
                 use_destination_embedding_in_message=False,
                 use_source_embedding_in_message=False,
                 dyrep=False):
        super(TGAT, self).__init__()

        self.n_layers = n_layers
        self.device = device
        self.logger = logging.getLogger(__name__)



        self.n_node_features = node_feat_dim
        self.n_nodes = None
        self.n_edge_features = edge_feat_dim # No edge feature needed.
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.embedding_module_type = embedding_module_type
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        self.dyrep = dyrep
        self.message_dimension = message_dimension

        self.use_memory = use_memory
        self.time_encoder = TimeEncode(dimension=self.n_node_features)
        self.memory = None
        self.node_raw_features = None
        self.edge_raw_features = None  # memory， node_raw_feature & edge_raw_feature only init when given an event.

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

        if self.use_memory:
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                                    self.time_encoder.dimension
            message_dimension = message_dimension if message_function != "identity" else raw_message_dimension

            # How to clear the memory.We can't create a new memory straightly, parameters are in it.
            # Seems We can, because memory has no gradients.
            self.memory = None

            self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                             device=device)
            self.message_function = get_message_function(module_type=message_function,
                                                         raw_message_dimension=raw_message_dimension,
                                                         message_dimension=message_dimension)
            self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                     message_dimension=message_dimension,
                                                     memory_dimension=self.memory_dimension,
                                                     device=device)

        self.embedding_module_type = embedding_module_type

        # 把 neighbor_finder 不在初始化的时候加入，而是在compute_embedding时候作为参数加入
        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     n_node_features=self.n_node_features,
                                                     n_edge_features=self.n_edge_features,
                                                     n_time_features=self.n_node_features,
                                                     embedding_dimension=self.embedding_dimension,
                                                     device=self.device,
                                                     n_heads=n_heads, dropout=dropout,
                                                     use_memory=use_memory,
                                                     n_neighbors=self.n_neighbors)

        # MLP to compute probability on an edge given two node embeddings
        self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                         self.n_node_features,
                                         1)

    # What is the usage of negative nodes?
    #  For self-supervised training, what's existing in graph, what's not, then get loss based on
    #  link prediction results, then back propagate and update weights.

    # However, we want these TGAT module run in a supervised training way, the loss is from the fake
    # news prediction results. So we should not need the negative nodes.
    def init_event(self, node_features, edge_features):
        # To create a new memory, we need to know the max-num of nodes in current temporal graph.

        # Get the dimension of memory as n_nodes
        self.n_nodes = len(node_features)
        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device) # astype?
        self.memory = Memory(n_nodes=self.n_nodes,
                             memory_dimension=self.memory_dimension,
                             input_dimension=self.message_dimension,
                             message_dimension=self.message_dimension,
                             device=self.device)
        # neighbor_finder需要data_processing中 get_neighbor_finder(Data) 来生成
        neighbor_finder = None
        return neighbor_finder

    def forward(self, memory, raw_node_features, raw_edge_features, neighbor_finder, source_nodes, destination_nodes,
                edge_times, edge_idxs, n_neighbors=20):

        self.memory = memory
        # Usually in forward function we process with a whole temporal graph,
        # so before every run we need to initialize memory first (self.init_memory), This should be down by training code.
        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times])

        memory = None
        time_diffs = None

        if self.use_memory:
            if self.memory_update_at_start:
                # Update memory for all nodes with messages stored in previous batches
                memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                              self.memory.messages)
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update

            ### Compute differences between the time the memory of a node was last updated,
            ### and the time for which we want to compute the embedding of a node
            source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
                source_nodes].long()
            source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
            destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
                destination_nodes].long()
            destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

            time_diffs = torch.cat([source_time_diffs, destination_time_diffs],
                                   dim=0)

        # Compute the embeddings using the embedding module
        # Here whe computing the node_embedding, we take all nodes including source, destination and negative into
        #   consideration. In fake news detection we can only take sources and destinations.
        node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                 node_features=raw_node_features,
                                                                 edge_features=raw_edge_features,
                                                                 neighbor_finder=neighbor_finder,
                                                                 source_nodes=nodes,
                                                                 timestamps=timestamps,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=time_diffs)

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]

        if self.use_memory:  # What is doing here?
            if self.memory_update_at_start:
                # Persist the updates to the memory only for sources and destinations (since now we have
                # new messages for them)
                self.update_memory(positives, self.memory.messages)

                assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
                    "Something wrong in how the memory was updated"

                # Remove messages for the positives since we have already updated the memory using them
                self.memory.clear_messages(positives)

            unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                          source_node_embedding,
                                                                          destination_nodes,
                                                                          destination_node_embedding,
                                                                          edge_times, edge_idxs)
            unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                                    destination_node_embedding,
                                                                                    source_nodes,
                                                                                    source_node_embedding,
                                                                                    edge_times, edge_idxs)
            if self.memory_update_at_start:
                self.memory.store_raw_messages(unique_sources, source_id_to_messages)
                self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
            else:
                self.update_memory(unique_sources, source_id_to_messages)
                self.update_memory(unique_destinations, destination_id_to_messages)

            if self.dyrep:
                source_node_embedding = memory[source_nodes]
                destination_node_embedding = memory[destination_nodes]
                #negative_node_embedding = memory[negative_nodes]

        return source_node_embedding, destination_node_embedding

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(self.memory, unique_nodes, unique_messages,
                                          timestamps=unique_timestamps)

    def get_updated_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(self.memory,
                                                                                     unique_nodes,
                                                                                     unique_messages,
                                                                                     timestamps=unique_timestamps)
        return updated_memory, updated_last_update

    def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                         destination_node_embedding, edge_times, edge_idxs):
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        edge_features = self.edge_raw_features[edge_idxs]

        source_memory = self.memory.get_memory(source_nodes) if not \
            self.use_source_embedding_in_message else source_node_embedding
        destination_memory = self.memory.get_memory(destination_nodes) if \
            not self.use_destination_embedding_in_message else destination_node_embedding

        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
            source_nodes), -1)

        source_message = torch.cat([source_memory, destination_memory, edge_features,
                                    source_time_delta_encoding],
                                    dim=1)
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], edge_times[i]))

        return unique_sources, messages

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder


if __name__ == '__main__':
    model_tgat = TGAT(node_feat_dim=500, edge_feat_dim=500, #node_features, edge_features,
                 device='cpu',
                 n_layers=2,  # node_features and edge features need to be input
                 n_heads=2, dropout=0.1, use_memory=True,
                 memory_update_at_start=True, message_dimension=100,
                 memory_dimension=500, embedding_module_type="graph_attention",
                 message_function="mlp",
                 mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                 std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
                 memory_updater_type="gru",
                 use_destination_embedding_in_message=False,
                 use_source_embedding_in_message=False,
                 dyrep=False)
    print("model_tgat created. ")

    raw_node_features = np.random.rand(6,500)

    raw_edge_features = np.random.rand(3,500)
    model_tgat.init_event(raw_node_features, raw_edge_features)

    full_data = Data([1, 1, 1], [2, 3, 4], np.random.rand(3), [0, 1, 2], [0, 0, 0])
    # We need full_data to init neighbor_finder
    neighbor_finder = get_neighbor_finder(data=full_data, uniform=False, max_node_idx=4)
    result = model_tgat(memory=model_tgat.memory,
                        raw_node_features=torch.from_numpy(raw_node_features).float(),
                        raw_edge_features=torch.from_numpy(raw_edge_features).float(),
                        neighbor_finder=neighbor_finder,
                        source_nodes=full_data.sources,
                        destination_nodes=full_data.destinations,
                        edge_times=full_data.timestamps,
                        edge_idxs=full_data.edge_idxs,
                        n_neighbors=20)
    print("model_tgat memory reset. ")
    '''
    train_batch = [1, 2, 3]
    for batch in train_batch:
        memory = model_tgat.init_memory(batch[0])
        res = model_tgat(memory=memory,
                         source_nodes=batch[0],
                         destination_nodes=batch[1],
                         negative_nodes=batch[2],
                         edge_times=batch[3],
                         edge_idxs=batch[4],
                         n_neighbors=20)
    '''