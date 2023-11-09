import numpy as np

def trans17to18(pose_results, i):
    pose_18 = [np.zeros((18, 2))]
    traj_ = [(pose_results[i]['bbox'][0] + pose_results[i]['bbox'][2]) / 2,
             (pose_results[i]['bbox'][1] + pose_results[i]['bbox'][3]) / 2]

    pose_ = pose_results[i]['keypoints'][:, 0:2]

    pose_18[0][0][0] = pose_[0][0]
    pose_18[0][0][1] = pose_[0][1]

    pose_18[0][2][0], pose_18[0][2][1] = pose_[6][0], pose_[6][1]

    pose_18[0][3][0], pose_18[0][3][1] = pose_[8][0], pose_[8][1]
    pose_18[0][4][0], pose_18[0][4][1] = pose_[10][0], pose_[10][1]
    pose_18[0][5][0], pose_18[0][5][1] = pose_[5][0], pose_[5][1]
    pose_18[0][6][0], pose_18[0][6][1] = pose_[7][0], pose_[7][1]
    pose_18[0][7][0], pose_18[0][7][1] = pose_[9][0], pose_[9][1]
    pose_18[0][8][0], pose_18[0][8][1] = pose_[12][0], pose_[12][1]
    pose_18[0][9][0], pose_18[0][9][1] = pose_[14][0], pose_[14][1]
    pose_18[0][10][0], pose_18[0][10][1] = pose_[16][0], pose_[16][1]
    pose_18[0][11][0], pose_18[0][11][1] = pose_[11][0], pose_[11][1]
    pose_18[0][12][0], pose_18[0][12][1] = pose_[13][0], pose_[13][1]
    pose_18[0][13][0], pose_18[0][13][1] = pose_[15][0], pose_[15][1]
    pose_18[0][14][0], pose_18[0][14][1] = pose_[2][0], pose_[2][1]
    pose_18[0][15][0], pose_18[0][15][1] = pose_[1][0], pose_[1][1]
    pose_18[0][16][0], pose_18[0][16][1] = pose_[4][0], pose_[4][1]
    pose_18[0][17][0], pose_18[0][17][1] = pose_[3][0], pose_[3][1]

    pose_18[0][1][0], pose_18[0][1][1] = (pose_18[0][5][0] + pose_18[0][6][0]) / 2, (
                pose_18[0][5][1] + pose_18[0][6][1]) / 2
    pose18 = pose_18[0]

    return traj_, pose18