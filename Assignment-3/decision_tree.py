import numpy as np
from typing import List
from classifier import Classifier


class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert (len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels) + 1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return

	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(
					node=child,
					name='  ' + name + '/' + str(idx_child),
					indent=indent + '  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent + '}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int],
				 num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label  # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None  # the dim of feature to be splitted

		self.feature_uniq_split = None  # the feature to be splitted

	def split(self):
		features = np.array(self.features)
		labels = np.array(self.labels)

		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			conditional_entropy_qty = 0.0
			# Get B x C array
			branches = np.array(branches).T
			total_nodes_tree = 0
			total_nodes_branch = []
			uncertainty_branch = []
			for i in range(len(branches)):
				total_nodes_branch.append(np.sum(branches[i]))
				uncertainty = 0
				for label in branches[i]:
					label_weight = label / total_nodes_branch[i]
					if label_weight > 0:
						uncertainty += (label_weight) * np.log(label_weight)
				uncertainty_branch.append(-uncertainty)
			total_nodes_tree = np.sum(np.array(total_nodes_branch))
			for i in range(len(branches)):
				conditional_entropy_qty += total_nodes_branch[i] / total_nodes_tree * uncertainty_branch[i]
			return conditional_entropy_qty

		C = np.unique(labels)
		min_entropy = 10000.0
		for idx_dim in range(len(features[0])):
			############################################################
			# TODO: compare each split using conditional entropy
			#       find the
			############################################################
			B = np.unique(features[:, idx_dim])
			branches = np.zeros((C.shape[0], B.shape[0]))
			for i in range(len(features)):
				ci, = np.where(C == labels[i])
				bi, = np.where(B == features[i][idx_dim])
				branches[ci[0]][bi[0]] += 1
			conditional_entropy_idx_dim = conditional_entropy(branches)
			if conditional_entropy_idx_dim < min_entropy:
				min_entropy = conditional_entropy_idx_dim
				self.dim_split = idx_dim

		############################################################
		# TODO: split the node, add child nodes
		############################################################
		#Split on min_entropy_index - create 1 child node for each branch
		self.feature_uniq_split = np.unique(features[:, self.dim_split]).tolist()
		for feature_uniq in self.feature_uniq_split:
			split_features = []
			split_labels = []
			split_cls_max = 100
			for i in range(len(features)):
				if features[i][self.dim_split] == feature_uniq:
					split_feature = np.delete(features[i], self.dim_split,
											  0).tolist()
					if split_feature:
						split_features.append(split_feature)
						split_labels.append(labels[i])
					else:
						self.splittable = False
						return
			label, count = np.unique(split_labels, return_counts=True)
			split_cls_max = label[np.argmax(count)]
			child = TreeNode(split_features, split_labels, split_cls_max)
			self.children.append(child)

		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			feature = feature[:self.dim_split] + feature[self.dim_split + 1:]
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max
