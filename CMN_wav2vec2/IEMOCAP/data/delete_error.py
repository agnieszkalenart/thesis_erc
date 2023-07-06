import pickle



transcripts, labels, own_historyID, other_historyID, own_historyID_rank, other_historyID_rank = pickle.load(open("CMN_wav2vec2/IEMOCAP/data/dataset.pkl",'rb'), encoding="latin1")

# print(len(own_historyID['Ses03F_impro06_F034']))
# print(own_historyID['Ses03F_impro06_F034'][15])
# print(len(own_historyID_rank['Ses03F_impro06_F034']))

# own_historyID['Ses03F_impro06_F034'] = ['Ses03F_impro06_F000', 'Ses03F_impro06_F001', 'Ses03F_impro06_F002', 'Ses03F_impro06_F003', 'Ses03F_impro06_F004', 'Ses03F_impro06_F005', 'Ses03F_impro06_F006', 'Ses03F_impro06_F007', 'Ses03F_impro06_F008', 'Ses03F_impro06_F009', 'Ses03F_impro06_F010', 'Ses03F_impro06_F011', 'Ses03F_impro06_F012', 'Ses03F_impro06_F013', 'Ses03F_impro06_F014', 'Ses03F_impro06_F015', 'Ses03F_impro06_F016', 'Ses03F_impro06_F017', 'Ses03F_impro06_F018', 'Ses03F_impro06_F019', 'Ses03F_impro06_F020', 'Ses03F_impro06_F021', 'Ses03F_impro06_F022', 'Ses03F_impro06_F023', 'Ses03F_impro06_F024', 'Ses03F_impro06_F025', 'Ses03F_impro06_F026', 'Ses03F_impro06_F027', 'Ses03F_impro06_F028', 'Ses03F_impro06_F029', 'Ses03F_impro06_F030', 'Ses03F_impro06_F031', 'Ses03F_impro06_F032', 'Ses03F_impro06_F033']
# own_historyID_rank['Ses03F_impro06_F034'] = [2, 3, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 24, 25, 26, 27, 29, 30, 31, 32, 34, 35, 36, 37, 39, 41, 43, 44, 47, 49, 51]
# other_historyID['Ses03F_impro06_F034'] = ['Ses03F_impro06_M000', 'Ses03F_impro06_M001', 'Ses03F_impro06_M002', 'Ses03F_impro06_M003', 'Ses03F_impro06_M004', 'Ses03F_impro06_M005', 'Ses03F_impro06_M006', 'Ses03F_impro06_M007', 'Ses03F_impro06_M008', 'Ses03F_impro06_M009', 'Ses03F_impro06_M010', 'Ses03F_impro06_M011', 'Ses03F_impro06_M012', 'Ses03F_impro06_M013', 'Ses03F_impro06_M014']
# other_historyID_rank['Ses03F_impro06_F034'] = [1, 4, 11, 19, 21, 23, 28, 33, 38, 42, 45, 46, 48, 50, 52]

# own_historyID['Ses03M_impro08b_M011'] = ['Ses03M_impro08b_M000', 'Ses03M_impro08b_M001', 'Ses03M_impro08b_M002', 'Ses03M_impro08b_M003', 'Ses03M_impro08b_M004',  'Ses03M_impro08b_M005', 'Ses03M_impro08b_M006', 'Ses03M_impro08b_M007', 'Ses03M_impro08b_M008', 'Ses03M_impro08b_M009', 'Ses03M_impro08b_M010']
# own_historyID_rank['Ses03M_impro08b_M011'] = [2, 4, 6, 8, 10, 15, 17, 19, 21, 22, 24]

# own_historyID['Ses03M_impro04_M038'] = ['Ses03M_impro04_M000', 'Ses03M_impro04_M001', 'Ses03M_impro04_M002', 'Ses03M_impro04_M003', 'Ses03M_impro04_M004', 'Ses03M_impro04_M005', 'Ses03M_impro04_M006', 'Ses03M_impro04_M007', 'Ses03M_impro04_M008', 'Ses03M_impro04_M009', 'Ses03M_impro04_M010', 'Ses03M_impro04_M011', 'Ses03M_impro04_M012', 'Ses03M_impro04_M013', 'Ses03M_impro04_M014', 'Ses03M_impro04_M015', 'Ses03M_impro04_M016', 'Ses03M_impro04_M017', 'Ses03M_impro04_M018', 'Ses03M_impro04_M019', 'Ses03M_impro04_M020', 'Ses03M_impro04_M021', 'Ses03M_impro04_M022', 'Ses03M_impro04_M023', 'Ses03M_impro04_M024', 'Ses03M_impro04_M025', 'Ses03M_impro04_M026', 'Ses03M_impro04_M027', 'Ses03M_impro04_M028', 'Ses03M_impro04_M029',  'Ses03M_impro04_M030', 'Ses03M_impro04_M031', 'Ses03M_impro04_M032', 'Ses03M_impro04_M033', 'Ses03M_impro04_M034', 'Ses03M_impro04_M035', 'Ses03M_impro04_M036', 'Ses03M_impro04_M037']
# own_historyID_rank['Ses03M_impro04_M038'] = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 27, 29, 31, 33, 34, 35, 37, 39, 41, 43, 45, 47, 49, 50, 51, 53, 54, 58, 60, 61, 63, 64, 66, 67, 70]

for id in own_historyID:
    values_to_remove_own_historyID = []
    values_to_remove_own_historyID_rank = []
    for i in range(len(own_historyID[id])):
        if own_historyID[id][i][-3:-1] == 'XX':
            values_to_remove_own_historyID.append(own_historyID[id][i])
            values_to_remove_own_historyID_rank.append(own_historyID_rank[id][i])
    for value in values_to_remove_own_historyID:
        own_historyID[id].remove(value)
    for value in values_to_remove_own_historyID_rank:
        own_historyID_rank[id].remove(value)


for id in other_historyID:
    values_to_remove_other_historyID = []
    values_to_remove_other_historyID_rank = []
    for i in range(len(other_historyID[id])):
        if other_historyID[id][i][-3:-1] == 'XX':
            values_to_remove_other_historyID.append(other_historyID[id][i])
            values_to_remove_other_historyID_rank.append(other_historyID_rank[id][i])
    for value in values_to_remove_other_historyID:
        other_historyID[id].remove(value)
    for value in values_to_remove_other_historyID_rank:
        other_historyID_rank[id].remove(value)

# Save the dictionary as a pickle file
with open("CMN_wav2vec2/IEMOCAP/data/dataset_2.pkl", 'wb') as file:
    pickle.dump([transcripts, labels, own_historyID, other_historyID, own_historyID_rank, other_historyID_rank ], file)

transcripts, labels, own_historyID, other_historyID, own_historyID_rank, other_historyID_rank = pickle.load(open("CMN_wav2vec2/IEMOCAP/data/dataset_2.pkl",'rb'), encoding="latin1")

print(own_historyID['Ses03F_impro06_F034'])
print(other_historyID['Ses03F_impro06_F034'])

