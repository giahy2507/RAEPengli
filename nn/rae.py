#-*- coding: utf-8 -*-
'''
Recursive autoencoder

@author: lpeng
'''
from __future__ import division

import argparse
import logging
from sys import stderr
from numpy import arange, dot, zeros, zeros_like, tanh, concatenate
from numpy import linalg as LA
from functions import tanh_norm1_prime, sum_along_column
from vec.wordvector import WordVectors
from ioutil import unpickle, Reader, Writer
from nn.instance import Instance



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InternalNode(object):
  
  def __init__(self, index,
               left_child, right_child, 
               p, p_unnormalized, 
               y1_minus_c1, y2_minus_c2,
               y1_unnormalized, y2_unnormalized):
    self.index = index # for debugging purpose
    self.left_child = left_child
    self.right_child = right_child
    self.p = p
    self.p_unnormalized = p_unnormalized
    self.y1_minus_c1 = y1_minus_c1
    self.y2_minus_c2 = y2_minus_c2
    self.y1_unnormalized = y1_unnormalized
    self.y2_unnormalized = y2_unnormalized
  
  
class LeafNode(object):
  
  def __init__(self, index, embedding):
    self.index = index
    self.p = embedding


class RecursiveAutoencoder(object):
  
  def __init__(self, Wi1, Wi2, bi, Wo1, Wo2, bo1, bo2,
               f=tanh, f_norm1_prime=tanh_norm1_prime):
    '''Initialize the weight matrices and set the nonlinear function
    
    Note: bi, bo1 and bo2 must have the shape (embsize, 1) instead of (embsize,)
    
    Args:
      Wi1, Wi2, bi: weight matrices for encoder
      Wo1, Wo2, bo1, bo2: weight matrices for decoder
      f: nonlinear function
      f_norm1_prime: returns the Jacobi matrix of f(x)/sqrt(||f(x)||^2).
        Note that the input of f_norm1_prime is f(x).
    '''
    self.Wi1 = Wi1
    self.Wi2 = Wi2
    self.bi = bi
    
    self.Wo1 = Wo1
    self.Wo2 = Wo2
    self.bo1 = bo1
    self.bo2 = bo2
    
    self.f = f
    self.f_norm1_prime = f_norm1_prime
  
  def get_embsize(self):
    return self.bi.shape[0]
  
  @classmethod
  def build(cls, theta, embsize):
    '''Initailize a recursive autoencoder using a parameter vector
    
    Args:
      theta: parameter vector
      embsize: dimension of word embedding vector
    '''
    assert(theta.size == cls.compute_parameter_num(embsize))
    offset = 0
    sz = embsize * embsize
    Wi1 = theta[offset:offset+sz].reshape(embsize, embsize)
    offset += sz
    
    Wi2 = theta[offset:offset+sz].reshape(embsize, embsize)
    offset += sz
    
    bi = theta[offset:offset+embsize].reshape(embsize, 1)
    offset += embsize
    
    Wo1 = theta[offset:offset+sz].reshape(embsize, embsize)
    offset += sz
    
    Wo2 = theta[offset:offset+sz].reshape(embsize, embsize)
    offset += sz
    
    bo1 = theta[offset:offset+embsize].reshape(embsize, 1)
    offset += embsize
    
    bo2 = theta[offset:offset+embsize].reshape(embsize, 1)
    offset += embsize
    
    return RecursiveAutoencoder(Wi1, Wi2, bi, Wo1, Wo2, bo1, bo2)
  
  @classmethod
  def compute_parameter_num(cls, embsize):
    '''Compute the parameter number of a recursive autoencoder
    
    Args:
      embsize: dimension of word embedding vector
    
    Returns:
      number of parameters
    '''
    sz = embsize*embsize # Wi1
    sz += embsize*embsize # Wi2
    sz += embsize # bi
    sz += embsize*embsize # Wo1
    sz += embsize*embsize # Wo2
    sz += embsize # bo1
    sz += embsize # bo2
    return sz
      
  def get_weights_square(self):
    square = (self.Wi1**2).sum()
    square += (self.Wi2**2).sum()
    square += (self.Wo1**2).sum()
    square += (self.Wo2**2).sum()
    return square
  
  def get_bias_square(self):
    square = (self.bi**2).sum()
    square += (self.bo1**2).sum()
    square += (self.bo2**2).sum()
    return square
  
  def encode(self, words_embedded):
    root, _ = self.forward(words_embedded)
    return root.p
    
  def forward(self, words_embedded):
    ''' Forward pass of training recursive autoencoders using backpropagation
    through structures.  
    
    Args:
      words_embedded: word embedding vectors (column vectors)
      
    Returns:
      value1: root of the tree, an instance of InternalNode 
      value2: reconstruction_error
    '''
    
    words_num = words_embedded.shape[1]
    
    tree_nodes = [None]*(2*words_num - 1)
    tree_nodes[0:words_num] = [LeafNode(i, words_embedded[:, (i,)]) 
                                        for i in range(words_num)]
    
    reconstruction_error = 0
    
    # build a tree greedily
    # initialize reconstruction errors
    c1 = words_embedded[:, arange(words_num-1)]
    c2 = words_embedded[:, arange(1, words_num)]

    p_unnormalized = self.f(dot(self.Wi1, c1) + dot(self.Wi2, c2) + self.bi[:, zeros(words_num-1, dtype=int)])
    p = p_unnormalized / LA.norm(p_unnormalized, axis=0)
    
    y1_unnormalized = self.f(dot(self.Wo1, p) + self.bo1[:, zeros(words_num-1, dtype=int)])
    y1 = y1_unnormalized / LA.norm(y1_unnormalized, axis=0)
  
    y2_unnormalized = self.f(dot(self.Wo2, p) + self.bo2[:, zeros(words_num-1, dtype=int)])
    y2 = y2_unnormalized / LA.norm(y2_unnormalized, axis=0)
    
    y1c1 = y1 - c1
    y2c2 = y2 - c2
    
    J = 1/2 * (sum_along_column(y1c1**2) + sum_along_column(y2c2**2))
    
    # initialize candidate internal nodes
    candidate_nodes = []
    for i in range(words_num-1):
      left_child = tree_nodes[i]
      right_child = tree_nodes[i+1]
      node = InternalNode(-i-1, left_child, right_child,
                          p[:, (i,)], p_unnormalized[:, (i,)],
                          y1c1[:, (i,)], y2c2[:, (i,)],
                          y1_unnormalized[:, (i,)], 
                          y2_unnormalized[:, (i,)])
      candidate_nodes.append(node)
    debugging_cand_node_index = words_num
      
    
    for j in range(words_num-1):
      # find the smallest reconstruction error
      J_minpos = J.argmin()
      J_min = J[J_minpos]
      reconstruction_error += J_min
      
      node = candidate_nodes[J_minpos]
      node.index = words_num + j # for dubugging
      tree_nodes[words_num+j] = node
  
      # update reconstruction errors
      if J_minpos+1 < len(candidate_nodes):
        c1 = node
        c2 = candidate_nodes[J_minpos+1].right_child
        right_cand_node, right_J = self.__build_internal_node(c1, c2)
        
        right_cand_node.index = -debugging_cand_node_index
        debugging_cand_node_index += 1
        candidate_nodes[J_minpos+1] = right_cand_node
        
        J[J_minpos+1] = right_J
        
      if J_minpos-1 >= 0:
        c1 = candidate_nodes[J_minpos-1].left_child
        c2 = node
        left_cand_node, left_J = self.__build_internal_node(c1, c2)
        
        left_cand_node.index = -debugging_cand_node_index
        debugging_cand_node_index += 1
        candidate_nodes[J_minpos-1] = left_cand_node
        J[J_minpos-1] = left_J
      
      valid_indices = [i for i in range(words_num-1-j) if i != J_minpos]
      J = J[valid_indices]
      candidate_nodes = [candidate_nodes[k] for k in valid_indices]
     
    return tree_nodes[-1], reconstruction_error 
  
  def __build_internal_node(self, c1_node, c2_node):
    '''Build a new internal node which represents the representation of 
    c1_node.p and c2_node.p computed using autoencoder  
    
    Args:
      c1_node: left node
      c2_node: right node
      
    Returns:
      value1: a new internal node 
      value2: reconstruction error, a scalar
    
    ''' 
    c1 = c1_node.p
    c2 = c2_node.p
    p_unnormalized = self.f(dot(self.Wi1, c1) + dot(self.Wi2, c2) + self.bi)
    p = p_unnormalized / LA.norm(p_unnormalized, axis=0)
    
    y1_unnormalized = self.f(dot(self.Wo1, p) + self.bo1)
    y1 = y1_unnormalized / LA.norm(y1_unnormalized, axis=0)
  
    y2_unnormalized = self.f(dot(self.Wo2, p) + self.bo2)
    y2 = y2_unnormalized / LA.norm(y2_unnormalized, axis=0)
    
    y1c1 = y1 - c1
    y2c2 = y2 - c2
    
    node = InternalNode(-1, c1_node, c2_node,
                        p, p_unnormalized,
                        y1c1, y2c2,
                        y1_unnormalized, y2_unnormalized)

    reconstruction_error = sum_along_column(y1c1**2) + sum_along_column(y2c2**2)
    reconstruction_error = 0.5*reconstruction_error[0]
    
    return node, reconstruction_error 
  
  
  class Gradients(object):
    '''Class for storing gradients.
    '''
    
    def __init__(self, rae):
      '''
      Args:
        rae: an instance of RecursiveAutoencoder
      '''
      self.gradWi1 = zeros_like(rae.Wi1)
      self.gradWi2 = zeros_like(rae.Wi2)
      self.gradbi = zeros_like(rae.bi)
      
      self.gradWo1 = zeros_like(rae.Wo1)
      self.gradWo2 = zeros_like(rae.Wo2)
      self.gradbo1 = zeros_like(rae.bo1)
      self.gradbo2 = zeros_like(rae.bo2)
      
    def to_row_vector(self):
      '''Place all the gradients in a row vector
      '''
      vectors = []
      vectors.append(self.gradWi1.reshape(self.gradWi1.size, 1))
      vectors.append(self.gradWi2.reshape(self.gradWi2.size, 1))
      vectors.append(self.gradbi)
      vectors.append(self.gradWo1.reshape(self.gradWo1.size, 1))
      vectors.append(self.gradWo2.reshape(self.gradWo2.size, 1))
      vectors.append(self.gradbo1)
      vectors.append(self.gradbo2)
      return concatenate(vectors)[:, 0]
    
    def __mul__(self, other):
      self.gradWi1 *= other
      self.gradWi2 *= other
      self.gradbi *= other
      self.gradWo1 *= other
      self.gradWo2 *= other
      self.gradbo1 *= other
      self.gradbo2 *= other
      return self
      
  def get_zero_gradients(self):
    return self.Gradients(self)
  
  def backward(self, root_node, total_grad, delta_parent=None, freq=1):
    '''Backward pass of training recursive autoencoder using backpropagation
    through structures.
    
    Args:
      root_node: an instance of InternalNode returned by forward()
      total_grad: the local gradients will be added to it. 
        It should be initialized by get_zero_gradients()
      delta_parent: delta vector that propagates from upper layer, it must have
        the shape (embsize, 1) instead of (embsize,)
      freq: frequency of this instance
    
    Returns:
      None
    '''
    if delta_parent is None:
      delta_parent_out = zeros((self.bi.size, 1))
    else:
      delta_parent_out = delta_parent
      
    self.__backward(root_node, total_grad, delta_parent_out, freq)
  
  def __backward(self, node, total_grad, delta_parent_out, freq):
    '''Backward pass of training recursive autoencoder using backpropagation
    through structures.
    
    Args:
      node: an instance of InternalNode or LeafNode
      total_grad: the local gradients will be added to it. 
        It should be initialized by get_zero_gradients()
      delta_parent_out: delta vector that propagates from upper layer
      freq: frequency of this instance
    Returns:
      None
    '''
    if isinstance(node, InternalNode):
      # reconstruction layer
      jcob1 = self.f_norm1_prime(node.y1_unnormalized)
      delta_out1 = dot(jcob1, node.y1_minus_c1)

      jcob2 = self.f_norm1_prime(node.y2_unnormalized)
      delta_out2 = dot(jcob2, node.y2_minus_c2)

      total_grad.gradWo1 += dot(delta_out1, node.p.T) * freq
      total_grad.gradWo2 += dot(delta_out2, node.p.T) * freq
      total_grad.gradbo1 += delta_out1 * freq
      total_grad.gradbo2 += delta_out2 * freq
      
      # encoder layer
      delta_sum = dot(self.Wo1.T, delta_out1) + dot(self.Wo2.T, delta_out2) + delta_parent_out

      delta_parent = dot(self.f_norm1_prime(node.p_unnormalized), delta_sum)
      
      total_grad.gradWi1 += dot(delta_parent, node.left_child.p.T) * freq
      total_grad.gradWi2 += dot(delta_parent, node.right_child.p.T) * freq
      total_grad.gradbi += delta_parent * freq
      
      # recursive
      delta_parent_out_left = dot(self.Wi1.T, delta_parent) - node.y1_minus_c1
      self.__backward(node.left_child, total_grad, delta_parent_out_left, freq)
      
      delta_parent_out_right = dot(self.Wi2.T, delta_parent) - node.y2_minus_c2
      self.__backward(node.right_child, total_grad, delta_parent_out_right, freq)
    elif isinstance(node, LeafNode):
      return
    else:
      msg = 'node should be an instance of InternalNode or LeafNode';
      raise TypeError(msg)

from util import*
def init_theta(embsize):


  parameters = []

  # Wi1
  parameters.append(init_W(embsize, embsize))
  # Wi2
  parameters.append(init_W(embsize, embsize))
  # bi
  parameters.append(zeros(embsize))

  # Wo1
  parameters.append(init_W(embsize, embsize))
  # Wo2
  parameters.append(init_W(embsize, embsize))
  # bo1
  parameters.append(zeros(embsize))
  # bo2
  parameters.append(zeros(embsize))

  return concatenate(parameters)

if __name__ == '__main__':
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('phrases', help='input file, each line is a phrase')
  parser.add_argument('word_vector_file', help='word vector file')
  parser.add_argument('theta', help='RAE parameter file (pickled)')
  parser.add_argument('output', help='output file')
  options = parser.parse_args()
  
  phrases_file = options.phrases
  word_vector_file = options.word_vector_file
  theta_file = options.theta
  output_file = options.output
  """
  word_vector_file = '../file_raw_sentences.txt'
  phrases_file = '../sample-training-file.txt'

  output_file = '../outputfile.txt'

  print >> stderr, 'load word vectors...'
  word_vectors = WordVectors.load_vectors(word_vector_file)
  embsize = word_vectors.embsize()
  
  print >> stderr, 'load RAE parameters...'
  theta = init_theta(10)
  rae = RecursiveAutoencoder.build(theta, embsize)
    
  total_cost = 0
  total_instance_num = 0
  total_internal_node_num = 0
  
  print '='*63
  print '%20s %20s %20s' % ('all', 'avg/node', 'internal node')
  print '-'*63
  
  with Reader(phrases_file) as reader, Writer(output_file) as writer:
    for phrase in reader:
      instance = Instance.parse_from_str(phrase, word_vectors)
      words_embedded = word_vectors[instance.words]
      print(words_embedded.shape)
      root_node, cost = rae.forward(words_embedded)
      vec = root_node.p.T[0] # convert n*1 vector to common vector
      writer.write(' '.join([str(vec[i]) for i in range(vec.size)]))
      writer.write('\n')
      
      internal_node_num = len(instance.words)-1
      if internal_node_num > 0:
        print '%20.8f, %20.8f, %18d' % (cost, cost / internal_node_num, internal_node_num)
      else:
        print '%20.8f, %20.8f, %18d' % (cost, cost, 0)
      
      total_cost += cost
      total_instance_num += 1
      total_internal_node_num += internal_node_num
  
  print '-'*63    
  print 'average reconstruction error per instance: %20.8f' % (total_cost / total_instance_num)
  print 'average reconstruction error per node:     %20.8f' % (total_cost / total_internal_node_num)
  print '='*63