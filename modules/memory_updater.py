from torch import nn
import torch


class MemoryUpdater(nn.Module):
  def update_memory(self, memory, unique_node_ids, unique_messages, timestamps):
    pass


class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, message_dimension, memory_dimension, device):
    super(SequenceMemoryUpdater, self).__init__()
    #self.memory = memory # self.memory在两个函数， update_memory 和 get_updated_memory 中都有用到，但是私以为没必要出现在构造函数
                              # 直接在两个应用函数中作为参数出现即可，因为Memory实质上是两个不带梯度的张量。
    self.memory = None
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device

  def update_memory(self, memory, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return
    self.memory = memory
    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    memory = self.memory.get_memory(unique_node_ids)
    self.memory.last_update[unique_node_ids] = timestamps

    updated_memory = self.memory_updater(unique_messages, memory)

    self.memory.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory(self, memory, unique_node_ids, unique_messages, timestamps):

    self.memory = memory
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    updated_memory = self.memory.memory.data.clone()
    updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps

    return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self,  message_dimension, memory_dimension, device):
    super(GRUMemoryUpdater, self).__init__(message_dimension, memory_dimension, device)

    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, message_dimension, memory_dimension, device):
    super(RNNMemoryUpdater, self).__init__(message_dimension, memory_dimension, device)

    self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


def get_memory_updater(module_type, message_dimension, memory_dimension, device):
  if module_type == "gru":
    return GRUMemoryUpdater(message_dimension, memory_dimension, device)
  elif module_type == "rnn":
    return RNNMemoryUpdater(message_dimension, memory_dimension, device)
