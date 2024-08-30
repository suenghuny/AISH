from Components.Modeler_Component_test import *
from Components.Adapter_Component import *
from Components.Policy import *
from collections import deque
import torch
from cfg import get_cfg
from GPO import Agent
import numpy as np
import json

from scipy.stats import randint

fix_l = 0
fix_u = 17
data_memory= []
graph_memory = []
visual_memory = []
duration_memory = []

def preprocessing(scenarios):
    scenario = scenarios
    if mode == 'txt':
        input_path = ["Data/Test/dataset{}/ship.txt".format(scenario),
                      "Data/Test/dataset{}/patrol_aircraft.txt".format(scenario),
                      "Data/Test/dataset{}/SAM.txt".format(scenario),
                      "Data/Test/dataset{}/SSM.txt".format(scenario),
                      "Data/Test/dataset{}/inception.txt".format(scenario)]
    else:
        input_path = "Data/Test/dataset{}/input_data.xlsx".format(scenario)

    data = Adapter(input_path=input_path,
                   mode=mode,
                   polar_chart=episode_polar_chart,
                   polar_chart_visualize=polar_chart_visualize)
    return data





visual_memory_dummy = [[0 for _ in range(cfg.discr_n)] for _ in range(100)]

def evaluation(agent, env, e):
    temp = random.uniform(fix_l, fix_u)
    agent_yellow = Policy(env, rule='rule2', temperatures=[temp, temp])
    done = False
    episode_reward = 0
    eval = False
    enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
    friendly_action_for_transition = [0] * len(env.friendlies_fixed_list)
    k = 0
    visual_memory_temp = deepcopy(visual_memory_dummy)
    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            k += 1
            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue')
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')

            actions_blue = list()
            for i in range(len(env.friendlies_fixed_list)):
                edge_index_ssm_to_ship = env.get_ssm_to_ship_edge_index(k = i)
                edge_index_ssm_to_ssm = env.get_ssm_to_ssm_edge_index(k = i)
                edge_index_sam_to_ssm = env.get_sam_to_ssm_edge_index(k = i)
                edge_index_ship_to_sam = env.get_ship_to_sam_edge_index(k = i)
                edge_index_ship_to_enemy = env.get_ship_to_enemy_edge_index(k = i)
                heterogeneous_edges = (edge_index_ssm_to_ship, edge_index_ssm_to_ssm, edge_index_sam_to_ssm, edge_index_ship_to_sam,edge_index_ship_to_enemy)
                ship_feature = env.get_ship_feature(k = i)
                missile_node_feature, node_cats = env.get_missile_node_feature(k = i)
                action_feature = env.get_action_feature(k = i)
                agent.eval_check(eval=True)
                if k >=2.0:
                    td_target = agent.get_td_target(ship_feature, missile_node_feature, heterogeneous_edges, avail_action_blue[i], action_feature, reward = reward, done = done)
                    #print(ship_feature)
                    # 7+cfg.discr_n+1:7+cfg.discr_n+4
                    # 7+cfg.discr_n+4:7+cfg.discr_n+6
                    # 7+cfg.discr_n+6:7+cfg.discr_n+8
                    #print(ship_feature[0][0:7])

                    #visual_memory_temp = np.zeros([80, cfg.discr_n])

                    try:
                        visual_memory_temp[int(k)] = ship_feature[0][7: 7+cfg.discr_n]
                    except IndexError as IE:pass
                    graph_memory.append([
                        graph_embedding, graph_feature, output,
                        td_target])





                    data_memory.append([
                        ship_feature[0][0:7],
                        ship_feature[0][7: 7+cfg.discr_n],
                        ship_feature[0][7+cfg.discr_n:7+cfg.discr_n+3],
                        ship_feature[0][7+cfg.discr_n+3:7+cfg.discr_n+5],
                        ship_feature[0][7+cfg.discr_n+5:7+cfg.discr_n+7],
                        ship_feature[0][7+cfg.discr_n+7:],
                        td_target])
                    #print(len(data_memory))
                    #if len(data_memory) - 1 in [2048, 2049, 4478, 2052, 2053, 2054, 2055, 2057, 2058, 4105, 4106, 4107, 4108, 4112, 4116, 4117, 4118, 4119, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2115, 2116, 2117, 4172, 4173, 79, 80, 81, 4175, 4176, 4181, 4182, 2166, 2167, 2168, 2169, 2172, 2173, 2177, 130, 2178, 2180, 133, 134, 137, 138, 4233, 4236, 4237, 4238, 4244, 4245, 2223, 2224, 2228, 2230, 2231, 184, 185, 186, 2232, 2233, 2234, 2235, 2236, 2237, 4289, 4290, 4293, 4294, 4295, 4296, 4297, 4298, 4299, 4300, 4301, 4302, 2275, 2276, 2278, 2282, 2283, 2284, 2285, 4340, 4341, 4342, 4345, 4346, 4347, 4348, 4349, 4350, 4351, 4352, 4353, 4354, 263, 264, 265, 266, 267, 268, 270, 271, 272, 276, 277, 2334, 2336, 2337, 2341, 2342, 2343, 2344, 4405, 4406, 4407, 4408, 4412, 328, 329, 330, 331, 334, 335, 342, 4463, 4464, 4465, 4466, 4467, 4468, 4470, 4471, 2424, 2429, 2430, 2431, 4477, 385, 386, 387, 2434, 2435, 390, 391, 392, 393, 394, 395, 396, 397, 398, 400, 401, 402, 2473, 2474, 4529, 4530, 4531, 4532, 4533, 4536, 4537, 4542, 4543, 2517, 2518, 2519, 2520, 2523, 477, 478, 2528, 2529, 2530, 483, 484, 485, 486, 487, 489, 4585, 4586, 4587, 493, 494, 4588, 4591, 4592, 4595, 4597, 4598, 2576, 2577, 2583, 2584, 2585, 538, 539, 540, 541, 2586, 2587, 2588, 545, 546, 547, 548, 549, 550, 551, 552, 553, 4649, 4650, 4651, 4652, 4653, 4658, 2626, 2627, 2628, 2631, 2632, 2633, 2637, 590, 591, 592, 593, 2638, 2639, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 4701, 4702, 4703, 4706, 4707, 4708, 4709, 4710, 4711, 4712, 657, 658, 659, 660, 661, 2712, 2713, 702, 703, 2750, 2756, 709, 710, 711, 712, 2757, 2758, 2759, 716, 717, 2760, 2761, 2762, 2763, 2589, 2795, 2796, 4856, 4857, 4858, 4859, 4860, 4868, 4869, 4870, 4871, 4872, 4873, 4874, 803, 804, 805, 806, 807, 808, 809, 815, 816, 4926, 4927, 4928, 4929, 4930, 4933, 4934, 4935, 4936, 4937, 4938, 4939, 4940, 4941, 4942, 4943, 4944, 4945, 2902, 2903, 857, 2905, 864, 865, 866, 867, 868, 869, 870, 2912, 2913, 2918, 2919, 876, 877, 2963, 2964, 2965, 920, 921, 922, 923, 924, 925, 926, 927, 928, 2968, 2970, 2972, 2977, 2978, 2979, 3023, 976, 3024, 3030, 983, 984, 985, 986, 3031, 3032, 3035, 3036, 3037, 3038, 3039, 1034, 1035, 3083, 3084, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 3090, 3091, 3092, 3093, 3095, 3096, 3097, 3098, 3099, 3100, 3134, 3135, 3136, 3141, 3142, 3143, 1096, 3144, 3145, 3146, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 3147, 3148, 3185, 3186, 3187, 3188, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 1152, 1153, 1154, 3200, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 3238, 3239, 3240, 3241, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 1242, 1243, 1244, 1247, 1248, 1249, 3298, 3299, 3300, 3301, 1254, 1255, 1256, 3302, 3303, 3340, 3341, 3342, 3347, 3348, 3349, 3350, 3351, 3352, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 3354, 3355, 1318, 1319, 3400, 3407, 3408, 3413, 3414, 3415, 1370, 1371, 1372, 1373, 1374, 1375, 1379, 1380, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3468, 3469, 1425, 1426, 1427, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 3514, 3515, 3522, 3523, 3524, 3525, 3526, 3527, 1481, 1482, 1483, 1486, 1487, 1491, 1492, 1493, 1494, 3580, 3581, 3582, 3583, 3584, 3585, 3586, 1540, 1541, 1542, 1543, 3590, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 3637, 3638, 3639, 3646, 3647, 1603, 1604, 1605, 1606, 1607, 3651, 3652, 1610, 1611, 3653, 3654, 1615, 1616, 1617, 1618, 3697, 3698, 3699, 3700, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 1667, 1668, 1669, 1670, 1671, 1679, 3591, 3592, 3593, 3760, 3761, 3762, 3763, 3764, 3765, 3767, 3768, 1722, 3773, 3774, 3775, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 3829, 1782, 1783, 1784, 3830, 1786, 1787, 3831, 3832, 1790, 1791, 1792, 1793, 3833, 3834, 3876, 3877, 3878, 3879, 3880, 3881, 3882, 3883, 1836, 1837, 1838, 1839, 1840, 3888, 1843, 1847, 1848, 1849, 1850, 3933, 3934, 1889, 1890, 3937, 3938, 3939, 3940, 3941, 3942, 1897, 1898, 1899, 1900, 3943, 3944, 3945, 1904, 1905, 1906, 3946, 3989, 3990, 3991, 3992, 3993, 3994, 3995, 3996, 1950, 1951, 1952, 1953, 1954, 4001, 1957, 1958, 1959, 1964, 1965, 1966, 4045, 4046, 4047, 4050, 4051, 4053, 4054, 2007, 4055, 4056, 4057, 4058, 2436, 2437, 2046, 2047]:
                    # if len(data_memory) - 1 in [3587, 3588, 3589, 2056, 2059, 2060, 2061, 2062, 4113, 4114, 4115, 3094, 1049, 1050, 1051, 4654, 4655, 4656, 4657, 2112, 2113, 2114, 3648, 3649, 3650, 2634, 2635, 1612, 1613, 1614, 2636, 4177, 82, 1107, 1108, 1109, 1110, 83, 84, 85, 86, 4178, 4179, 4180, 4713, 4714, 4715, 2174, 2175, 2176, 3710, 2179, 1674, 139, 140, 141, 142, 143, 1675, 1676, 1677, 1678, 3997, 3998, 4239, 4240, 3999, 4241, 4242, 4243, 4000, 3769, 3770, 3771, 3772, 2238, 2239, 713, 714, 715, 1250, 1251, 1252, 1253, 2286, 2287, 2288, 2289, 1788, 1789, 4875, 4876, 4877, 4878, 4879, 4880, 273, 274, 275, 4881, 278, 279, 3353, 1313, 1314, 1315, 1316, 1317, 2338, 2339, 2340, 810, 811, 812, 813, 814, 3884, 3885, 3886, 3887, 1844, 1845, 1846, 4413, 4414, 4415, 4416, 4417, 4418, 4419, 336, 337, 338, 339, 340, 341, 3409, 3410, 3411, 3412, 1376, 1377, 1378, 2914, 2915, 2916, 2917, 871, 872, 873, 874, 875, 1901, 1902, 1903, 4472, 4473, 4474, 4475, 4476, 2432, 2433, 3465, 3466, 3467, 3470, 399, 2973, 2974, 2975, 2976, 929, 930, 931, 932, 933, 934, 935, 1960, 1961, 1962, 1963, 4538, 4539, 4540, 4541, 3528, 3529, 1488, 1489, 1490, 4052, 3033, 3034, 987, 988, 989, 990, 991, 992, 993, 2524, 2525, 2526, 2527, 488, 490, 491, 492, 4593, 4594, 4596]:
                    # if len(data_memory) - 1 in [4355, 3594, 403, 1556, 280, 1052, 3356, 3358, 1057, 1322, 818, 1851,
                    #                             721, 87, 344, 343, 1495, 347, 1497, 3416, 3417, 608, 609, 2920, 1258,
                    #                             1259, 2922, 878, 495, 496, 1263, 1907, 1909, 1910]:
                if non_move_masking == True:
                    if True in avail_action_blue[i][1:]:
                        avail_action_blue[i][0] = False
                action_blue, prob, mask, a_index, graph_embedding, graph_feature, output = agent.sample_action_visualize(ship_feature, missile_node_feature,heterogeneous_edges, avail_action_blue[i],action_feature,
                                                                                                                        random =True)
                actions_blue.append(action_blue)


            action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)
            reward, win_tag, done, leakers = env.step(actions_blue, action_yellow)
            visual_memory.append(visual_memory_temp)
            visual_memory_temp = deepcopy(visual_memory_dummy)
            duration_memory.append(k)
            # vs = np.stack(visual_memory)
            # if (e == 20) and (k == 10):
            #     import seaborn as sns
            #     import matplotlib.pyplot as plt
            #     import pandas as pd
            #     #print(vs.shape)
            #     vs = vs.mean(axis=0)
            #     visual_memory_temp = np.zeros([70, cfg.discr_n])
            #
            #
            #     # vs 데이터를 DataFrame으로 변환
            #     df = pd.DataFrame(vs, columns=[i for i in range(cfg.discr_n)])
            #
            #     # 플롯 크기 설정
            #     plt.figure(figsize=(12, 8))
            #
            #     # 데이터 준비
            #     x = []
            #     y = []
            #     colors = []
            #     for i in range(len(df)):
            #         for j in range(len(df.columns)):
            #             x.append(j)
            #             y.append(i)
            #             colors.append(df.iloc[i, j])
            #
            #     # 산점도 생성
            #     scatter = plt.scatter(x, y, c=colors, cmap="turbo", s=100, vmin=0, vmax=0.05)
            #
            #     # 컬러바 추가
            #     cbar = plt.colorbar(scatter)
            #     cbar.set_label('미사일 분포', rotation=270, labelpad=15)
            #
            #     # 차트 꾸미기
            #     plt.title('Non Random', fontsize=16)
            #     plt.xlabel('구간', fontsize=12)
            #     plt.ylabel('시간', fontsize=12)
            #
            #     # x축, y축 눈금 설정
            #     plt.xticks(range(len(df.columns)), df.columns)
            #     plt.yticks(range(len(df)), [f't{i + 1}' for i in range(len(df))])
            #
            #     # 격자 추가
            #     plt.grid(True, linestyle='--', alpha=0.7)
            #
            #     plt.tight_layout()
            #     plt.show()
            #


            episode_reward += reward
        else:
            pass_transition = True
            actions_blue = list()
            for i in range(len(env.friendlies_fixed_list)):
                actions_blue.append([0, 0, 0, 0, 0, 0, 0, 0])

            env.step(action_blue=actions_blue, action_yellow=enemy_action_for_transition,pass_transition=pass_transition)

    return episode_reward, win_tag




if __name__ == "__main__":
    cfg = get_cfg()
    vessl_on = cfg.vessl
    if vessl_on == True:
        import vessl

        vessl.init()
        output_dir = "/output/"
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        print("시작")
        from torch.utils.tensorboard import SummaryWriter

        output_dir = "../output_susceptibility/"
        writer = SummaryWriter('./logs2')
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    import time

    """

    환경 시스템 관련 변수들

    """
    visualize =False  # 가시화 기능 사용 여부 / True : 가시화 적용, False : 가시화 미적용
    size = [600, 600]  # 화면 size / 600, 600 pixel
    tick = 2  # 가시화 기능 사용 시 빠르기
    n_step = cfg.n_step
    simtime_per_frame = cfg.simtime_per_frame
    decision_timestep = cfg.decision_timestep
    detection_by_height = False  # 고도에 의한
    mode = 'excel'  # 전처리 모듈 / 'excel' : input_data.xlsx 파일 적용, 'txt' "Data\ship.txt", "Data\patrol_aircraft.txt", "Data\SAM.txt", "Data\SSM.txt"를 적용
    rule = 'rule2'  # rule1 : 랜덤 정책 / rule2 : 거리를 기반 합리성에 기반한 정책(softmax policy)
    temperature = [10,
                   20]  # rule = 'rule2'인 경우만 적용 / 의사결정의 flexibility / 첫번째 index : 공중 위험이 낮은 상태, 두번째 index : 공중 위험이 높은 상태
    ciws_threshold = 1
    polar_chart_visualize = False
    non_move_masking = True
    scenarios = ['scenario1', 'scenario2', 'scenario3']
    lose_ratio = list()
    remains_ratio = list()
    polar_chart_scenario1 = [33, 29, 25, 33, 30, 30, 55, 27, 27, 35, 25, 30, 40]  # RCS의 polarchart 적용
    print(cfg)
    polar_chart = [polar_chart_scenario1]
    df_dict = {}
    # scenario = np.random.choice(scenarios)
    episode_polar_chart = polar_chart[0]
    records = list()
    datasets = [i for i in range(1, 2)]
    non_lose_ratio_list = []
    raw_data = list()
    for dataset in datasets:

        print("====dataset{}=====".format(dataset))
        fitness_history = []
        data = preprocessing(dataset)
        t = 0
        env = modeler(data,
                      visualize=visualize,
                      size=size,
                      detection_by_height=detection_by_height,
                      tick=tick,
                      simtime_per_framerate=simtime_per_frame,
                      ciws_threshold=ciws_threshold,
                      action_history_step=cfg.action_history_step)

        agent = Agent(action_size=env.get_env_info()["n_actions"],
                      feature_size_ship=env.get_env_info()["ship_feature_shape"],
                      feature_size_missile=env.get_env_info()["missile_feature_shape"],
                      n_node_feature_missile=env.friendlies_fixed_list[0].air_tracking_limit +
                                             env.friendlies_fixed_list[0].air_engagement_limit +
                                             env.friendlies_fixed_list[0].num_m_sam +
                                             1,
                     node_embedding_layers_ship=list(eval(cfg.ship_layers)),
                     node_embedding_layers_missile=list(eval(cfg.missile_layers)),
                     n_representation_ship = cfg.n_representation_ship,
                     n_representation_missile = cfg.n_representation_missile,
                     n_representation_action = cfg.n_representation_action,

                     learning_rate=cfg.lr,
                     learning_rate_critic=cfg.lr_critic,
                     gamma=cfg.gamma,
                     lmbda=cfg.lmbda,
                     eps_clip = cfg.eps_clip,
                     K_epoch = cfg.K_epoch,
                     layers=list(eval(cfg.ppo_layers))
                     )

        load_file = "episode33100"
        agent.load_network(load_file+'.pt') # 2900, 1600
        reward_list = list()

        non_lose_ratio = 0
        non_lose_records = []
        seed = cfg.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        for e in range(100):
            env = modeler(data,
                          visualize=visualize,
                          size=size,
                          detection_by_height=detection_by_height,
                          tick=tick,
                          simtime_per_framerate=simtime_per_frame,
                          ciws_threshold=ciws_threshold,
                          action_history_step=cfg.action_history_step)
            episode_reward, win_tag = evaluation(agent, env, e)

            if win_tag != 'lose':
                non_lose_ratio += 1 / cfg.n_test
                non_lose_records.append(1)
                raw_data.append([str(env.random_recording), 1])
            else:
                non_lose_records.append(0)
                raw_data.append([str(env.random_recording), 0])

            print(e, win_tag, np.mean(non_lose_records), np.mean(duration_memory))

        with open("visual feature final random2.json", "w", encoding='utf-8') as json_file:
            json.dump(visual_memory, json_file, ensure_ascii=False)

        # import matplotlib.pyplot as plt
        # from sklearn.manifold import TSNE
        #
        # X = np.array([item[0] for item in data_memory])
        # y = [item[1] for item in data_memory]
        # tsne = TSNE(n_components=1, random_state=0)
        # X_2d = tsne.fit_transform(X)
        #
        # # 시각화
        # plt.figure(figsize=(8, 6))
        # scatter = plt.scatter(X_2d[:, 0], y)
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title('t-SNE visualization')
        # plt.show()
        #
        #
        #


