#
#
# import numpy as np
#
#
# dict_ = {}
#
# dict_['agent0'] = [[1,2,31], [2,52,33]]
# dict_['agent1'] = [[1,23,32], [2,5,354]]
# dict_['agent2'] = [[1,24,33], [2,54,3]]
# dict_['agent3'] = [[1,24,34], [25,5,3]]
#
# for i in dict_.values():
#     print(i)
#
# list = np.asarray([i for i in dict_.values()])
# print(list.shape)
#

# import tensorflow as tf
# na_sparse = tf.one_hot(10, 4, axis=-1)
# print(na_sparse)
#
# na_sparse = tf.reshape(na_sparse, [-1, 9*4])
# print(na_sparse)

import numpy as np
# a = [3.3, 1.1]
#
# b = np.tile(a,(10,1))
# print(b.shape)

# import tensorflow as tf
# num_agents = 2
# agents = [0, 1]
# one_hot_agents = tf.expand_dims(tf.one_hot(agents, num_agents, dtype=tf.float32), axis=1)
# print(one_hot_agents)
# tiled = tf.tile(one_hot_agents[0], (32,1))
# print(tiled)

# a = np.asarray([[[1,1],[2,2],[3,3],[4,4],[5,5]], [[10,10],[20, 20],[30,30],[40,40],[50,50]]])
# print(a)
# batch_size, n_step = 4, 2
# # a = np.random.normal(0.5, 2.0, (batch_size, n_step, 51, 40, 16))
# print(a.shape)
#
# s = a.shape
# b = a.reshape((s[0]*s[1], *a.shape[2:]))
# print(b)
# print(b.shape)
#
# b_flat = b.flatten()
# print(b_flat)
# print(b_flat.shape)
#
# c = b.reshape((a.shape[0], a.shape[1], *a.shape[2:]))
# # print(c.shape, c)
# import numpy as np
# fps_ = [[1, 2, 3],
#         [5, 6, 7],
#         [8, 9, 10],
#         [11, 12, 13]]
#
# agent_ids = [0, 1, 2, 3]
# fps = []
# for a in agent_ids:
#     fps.append(fps_[:a] + fps_[a+1:])
#
# # fps = np.delete(fps_, 1, axis=0)
# print(np.array(fps_))
# print(np.array(fps))

# rewards = np.array([[1, 2, 3],
#             [5, 6, 7],
#             [8, 9, 10],
#             [11, 12, 13]])

# dones = np.array([[0., 0., 0.],
#                 [0., 0., 0.],
#                 [0., 0., 0.],
#                 [1., 1., 1.]])
#
# print('rewards ', rewards)
# print('rewards[::-1] ', rewards[::-1])
# print('dones ', dones)
# print('dones[::-1] ', dones[::-1])
# gamma = 0.99
# discounted = []
# r = 0
# # print(f'rewards[::-1] is {rewards[::-1]}')
# # print(f'dones[::-1] is {dones[::-1]}')
# for reward, done in zip(rewards[::-1], dones[::-1]):
#     print(f'reward is {reward}')
#     print(f'done is {done}')
#     r = reward + gamma * r * (1. - done)  # fixed off by one bug
#     discounted.append(r)
#
# print('discounted reward ', discounted)
# print('discounted[::-1] reward ', discounted[::-1])

arr = np.array([[1, 2, 3],
            [5, 6, 7],
            [8, 9, 10],
            [11, 12, 13]])
print(arr)
s = arr.shape
print(s)
arr_ = arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
print(arr_)
