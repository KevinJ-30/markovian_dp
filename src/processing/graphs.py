"""Select training graphs without changing global node identifiers."""

def make_training_graph(test_graph):
    """Return the graph visible to training, separate from the test graph."""
    train_graph = test_graph.clone()
    if hasattr(test_graph, 'train_edge_index'):
        train_graph.edge_index = test_graph.train_edge_index
    else:
        is_train = test_graph.train_mask
        edge_index = test_graph.edge_index
        train_graph.edge_index = edge_index[
            :, is_train[edge_index[0]] & is_train[edge_index[1]]]
    return train_graph
