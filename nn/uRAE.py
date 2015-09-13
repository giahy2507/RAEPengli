from numpy import  arange, dot, zeros, zeros_like, tanh, concatenate
from numpy import linalg as LA
from functions import tanh_norm1_prime, sum_along_column



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

class InternalNodeDecode(object):
    def __init__(self, index, left_child, right_child, p, p_unnormalized,y_minus_p, delta_parent_out):
        self.index = index
        self.left_child = left_child
        self.right_child = right_child
        self.p = p
        self.p_unnormalized = p_unnormalized
        self.delta_parent_out = delta_parent_out
        self.y_minus_p = y_minus_p

class LeafNodeDecode(object):
    def __init__(self, index, p_unnormalized, y_minus_p):
        self.index = index
        self.p_unnormalized = p_unnormalized
        self.y_minus_p = y_minus_p
        self.delta_parent_out = y_minus_p

class UnfoldingRecursiveAutoencoder(object):

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

        return UnfoldingRecursiveAutoencoder(Wi1, Wi2, bi, Wo1, Wo2, bo1, bo2)

    def updateParameter(self,delta_theta, embsize, alpha = 0.1):
        sz = embsize * embsize
        offset = 0
        delta_Wi1 = delta_theta[offset:offset+sz].reshape(embsize, embsize)
        self.Wi1 -= alpha*delta_Wi1
        offset += sz

        delta_Wi2 = delta_theta[offset:offset+sz].reshape(embsize, embsize)
        self.Wi2 -= alpha*delta_Wi2
        offset += sz

        delta_bi = delta_theta[offset:offset+embsize].reshape(embsize, 1)
        self.bi -= alpha*delta_bi
        offset += embsize

        delta_Wo1 = delta_theta[offset:offset+sz].reshape(embsize, embsize)
        self.Wo1 -= alpha*delta_Wo1
        offset += sz

        delta_Wo2 = delta_theta[offset:offset+sz].reshape(embsize, embsize)
        self.Wo2 -= alpha*delta_Wo2
        offset += sz

        delta_bo1 = delta_theta[offset:offset+embsize].reshape(embsize, 1)
        self.bo1 -= alpha*delta_bo1
        offset += embsize

        delta_bo2 = delta_theta[offset:offset+embsize].reshape(embsize, 1)
        self.bo2 -= alpha*delta_bo2
        offset += embsize

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


    def forwardUnfolding(self, words_embedded):
        encode_tree, rec_error_encode = self.encode(words_embedded)
        decode_tree, rec_enror_decode = self.decode(encode_tree)
        return encode_tree, decode_tree, rec_enror_decode

    def decode(self, encode_tree):
        decode_tree = [None]*(len(encode_tree))
        count = len(encode_tree) - 1
        decode_tree[-1] = InternalNodeDecode(encode_tree[-1].index,encode_tree[-1].left_child,encode_tree[-1].right_child,encode_tree[-1].p,encode_tree[-1].p_unnormalized,None,None)
        while 1:
            if isinstance(encode_tree[count],LeafNode):
                break
            left_child_decode, right_child_decode = self.extract_Internal_Node(decode_tree[count],encode_tree[count])
            decode_tree[count].left_child = left_child_decode
            decode_tree[count].right_child = right_child_decode
            decode_tree[left_child_decode.index] = left_child_decode
            decode_tree[right_child_decode.index] = right_child_decode
            count-=1

        construction_error = 0
        for index in arange(0,len(encode_tree),1):
            if isinstance(encode_tree[index],InternalNode):
                break
            construction_error += LA.norm(decode_tree[index].p_unnormalized - encode_tree[index].p)

        return decode_tree, construction_error

    def extract_Internal_Node(self, node, node_baseon):
        p_left_unnormalized = self.f(dot(self.Wo1, node.p) + self.bo1)
        p_left = p_left_unnormalized / LA.norm(p_left_unnormalized, axis=0)

        p_right_unnormalized = self.f(dot(self.Wo2, node.p) + self.bo2)
        p_right = p_right_unnormalized / LA.norm(p_right_unnormalized, axis=0)

        if isinstance(node_baseon.left_child,LeafNode):
            left_child = LeafNodeDecode(node_baseon.left_child.index,p_left_unnormalized,p_left_unnormalized - node_baseon.left_child.p)
        else:
            left_child = InternalNodeDecode(node_baseon.left_child.index,None,None,p_left,p_left_unnormalized,
                                            p_left_unnormalized - node_baseon.left_child.p_unnormalized,None)

        if isinstance(node_baseon.right_child,LeafNode):
            right_child = LeafNodeDecode(node_baseon.right_child.index,p_right_unnormalized,p_right_unnormalized - node_baseon.right_child.p)
        else:
            right_child = InternalNodeDecode(node_baseon.right_child.index,None,None,p_right,p_right_unnormalized,
                                             p_right_unnormalized - node_baseon.right_child.p_unnormalized,None)

        return left_child,right_child

    def encode(self, words_embedded):
        '''
        Forward pass of training recursive autoencoders using backpropagation
        through structures.

        Args:
          words_embedded: word embedding vectors (column vectors)

        Returns:
          value1: root of the tree, an instance of InternalNode
          value2: reconstruction_error
        '''
        words_num = words_embedded.shape[1]

        tree_nodes = [None]*(2*words_num - 1)
        tree_nodes[0:words_num] = [LeafNode(i, words_embedded[:, (i,)]) for i in range(words_num)]

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

        return tree_nodes, reconstruction_error

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
            self.f_norm1_prime = rae.f_norm1_prime
            self.f = rae.f

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

    def getFirstInternalIndex(self,tree):
        for i in range(0,len(tree)):
            if isinstance(tree[i],InternalNode) or isinstance(tree[i],InternalNodeDecode):
                return i

    def backwardUnfolding(self, encode_tree, decode_tree, total_grad, freq = 1):
        index = self.getFirstInternalIndex(encode_tree)
        # reconstruct layer
        for i in arange(index,len(decode_tree),1):
            left_child = decode_tree[i].left_child
            right_child = decode_tree[i].right_child

            jcob_left_child = self.f_norm1_prime(left_child.p_unnormalized)
            delta_out1 = dot(jcob_left_child, left_child.delta_parent_out)

            jcob_right_child = self.f_norm1_prime(right_child.p_unnormalized)
            delta_out2 = dot(jcob_right_child, right_child.delta_parent_out)

            decode_tree[i].delta_parent_out = dot(self.Wo1.T, delta_out1) + dot(self.Wo2.T, delta_out2)

            total_grad.gradWo1 += dot(delta_out1, decode_tree[i].p_unnormalized.T) * freq
            total_grad.gradWo2 += dot(delta_out2, decode_tree[i].p_unnormalized.T) * freq

            total_grad.gradbo1 += delta_out1 * freq
            total_grad.gradbo2 += delta_out2 * freq

        # encoder layer
        self.__backwardForUnfolding(encode_tree[-1],total_grad,decode_tree[-1].delta_parent_out,freq)

    def __backwardForUnfolding(self, node, total_grad, delta_parent_out, freq):
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
            # encoder layer
            delta_sum = delta_parent_out

            delta_parent = dot(self.f_norm1_prime(node.p_unnormalized), delta_sum)

            total_grad.gradWi1 += dot(delta_parent, node.left_child.p.T) * freq
            total_grad.gradWi2 += dot(delta_parent, node.right_child.p.T) * freq
            total_grad.gradbi += delta_parent * freq

            # recursive
            delta_parent_out_left = dot(self.Wi1.T, delta_parent) - node.y1_minus_c1
            self.__backwardForUnfolding(node.left_child, total_grad, delta_parent_out_left, freq)

            delta_parent_out_right = dot(self.Wi2.T, delta_parent) - node.y2_minus_c2
            self.__backwardForUnfolding(node.right_child, total_grad, delta_parent_out_right, freq)
        elif isinstance(node, LeafNode):
            return
        else:
            msg = 'node should be an instance of InternalNode or LeafNode';
            raise TypeError(msg)

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

"""
def process_local_batch(uRAE, instances):
  gradients = uRAE.get_zero_gradients()
  total_rec_error = 0
  total_internal_node_num = 0;
  for instance in instances:
    if instance.words_embedded.shape[1] == 1:
        continue

    encode_tree, rec_error_encode = uRAE.encode(instance.words_embedded)
    decode_tree, rec_enror_decode = uRAE.decode(encode_tree)
    total_internal_node_num += 0.5*(len(encode_tree)-1)*instance.freq

    uRAE.backwardUnfolding(encode_tree, decode_tree, gradients, freq=instance.freq)

    total_rec_error += rec_enror_decode * instance.freq
  return total_rec_error, gradients.to_row_vector()/total_internal_node_num


def my_load_instances(file_train, file_model, embsize):
  '''Load training examples

  Args:
    instance_strs: each string is a training example
    word_vectors: an instance of vec.wordvector

  Return:
    instances: a list of Instance
  '''
  modelW2V = WordVector(filename='file_model',embsize=100)
  fi = open(file_train,'r')
  total_internal_node = 0
  instances = []
  for line in fi:
    words, freq = modelW2V.loadInstance(line)
    total_internal_node += len(words)
    instances.append(modelW2V.loadInstance(line))
  return instances, total_internal_node

"""

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
    '''
    embsize = 100
    theta = init_theta(embsize)
    uRAE = UnfoldingRecursiveAutoencoder.build(theta, embsize)

    fi = open('../sentences.txt','r')
    fo = open('../wordandvector.txt','w')
    modelW2V = WordVector(filename='../w2v',embsize=100)
    total_internal_node = 0;
    instances = []
    for line in fi:
        words, freq = modelW2V.loadInstance(line)
        total_internal_node += len(words)
        instances.append(modelW2V.loadInstance(line))

    '''

