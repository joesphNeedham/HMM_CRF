# -*-coding:utf-8 -*-
import numpy as np
import pandas as pd


class MyHMM(object):
    """
    param estimate
    """
    def __init__(self, p, state_trans_matrix, emission_matrix, observed_sequence, method="forward"):
        self.p = p
        self.state_trans_matrix = state_trans_matrix
        self.emission_matrix = emission_matrix
        self.observed_sequence = observed_sequence
        self.method = method

    def forward(self):
        """
        1. 这里没有指定第一个观察值到底是哪个状态，
        因此初始状态取得是一个所有状态对应的概率向量；最后的结果就是一个关于各个状态的整体期望
        2. 在已知第一个观察值X的时候，需要找出P(X|Z),也就是emission_prob_matrix中该观测值
        对应的一列；这里也是由于没有确定出初始观测值对应的具体状态，
        因此取得是一个向量。
        3. 前向算法的实施，就是不断的重新赋值的过程。
        4. 状态转移概率，用状态转移矩阵；而发射概率，也是一个发射概率向量
        :return:
        """
        init_emission_prob = self.emission_matrix[self.observed_sequence[0]] # 由于初始的状态未知，因此发射概率是考虑了所有的状态后对应的一个向量
        alpha = self.p*init_emission_prob
        for i in range(1, self.observed_sequence.size):
            alpha = alpha.dot(self.state_trans_matrix)*self.emission_matrix[self.observed_sequence[i]]
        return alpha.sum()





