# -- coding: utf-8 --
# @Time : 2023/2/10 15:06
# @Author : Ma Yunyi
# @Software: PyCharm
import pandas as pd

jamInfo = pd.read_csv('../data_sumovs/jamInfoAll.csv')
jam_time = jamInfo[['timeOneDay', 'day_index', 'period', 'jam_influ_segment', 'jam_influ_path']].values.tolist()

jam_time_all = []
jam_segment_info_dict = {}
jam_path_info_dict = {}
for j_t in jam_time:
    timeOneDay, day_index, period, jam_influ_segments, jam_influ_path = j_t[0], j_t[1], j_t[2], j_t[3], j_t[4]
    t_interval_start = 720 * (day_index - 1) + timeOneDay
    t_interval_end = t_interval_start + 30 * period
    jam_time_1 = [i for i in range(int(t_interval_start), int(t_interval_end))]
    jam_time_all.extend(jam_time_1)
    for t in jam_time_1:
        jam_segment_info_dict[t] = jam_influ_segments
        jam_path_info_dict[t] = jam_influ_path



with open('../data_sumovs/jam_time.txt', 'w') as f:
    f.write(str(jam_time_all))

with open('../data_sumovs/jam_segment_info_dict.txt', 'w') as f:
    f.write(str(jam_segment_info_dict))

with open('../data_sumovs/jam_path_info_dict.txt', 'w') as f:
    f.write(str(jam_path_info_dict))
