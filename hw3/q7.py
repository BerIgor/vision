import random
from scipy import spatial

from hw3 import utils
from hw3 import q4


def calc_transform_ransac(ref_points, seq_points):
    if len(ref_points) != len(seq_points):
        raise Exception('Bad lengths')
    indices = get_best_inlier_group_indices(ref_points, seq_points)
    ref_points_best = [ref_points[i] for i in indices]
    seq_points_best = [seq_points[i] for i in indices]

    ref_points_best = utils.invert_points(ref_points_best)
    seq_points_best = utils.invert_points(seq_points_best)
    print(str(ref_points_best) + " " + str(seq_points_best))

    return q4.get_transformation(ref_points_best, seq_points_best)


def get_best_inlier_group_indices(ref_points, seq_points):
    if len(ref_points) != len(seq_points):
        raise Exception('Bad lengths')
    iteration = 1
    inlier_groups = list()
    while len(inlier_groups) == 0:
        inlier_groups = get_inlier_groups_indices(ref_points, seq_points, 100, max_dist=45*iteration)
        iteration += 1

    largest_group = max(inlier_groups, key=len)
    return largest_group


def get_inlier_groups_indices(ref_points, seq_points, repeats, max_dist=45):

    groups = list()
    if len(ref_points) != len(seq_points):
        raise Exception('Bad lengths')

    for i in range(repeats):
        rand_indices = random.sample(range(len(ref_points)), min(len(seq_points)-1, 3))
        ref_points_sample = [ref_points[i] for i in rand_indices]
        seq_points_sample = [seq_points[i] for i in rand_indices]
        a, b = q4.get_transformation(ref_points_sample, seq_points_sample)
        group = list()
        for j in range(len(ref_points_sample)):
            estimated_source_point = utils.transform(seq_points_sample[j], a, b)
            dist = spatial.distance.euclidean(estimated_source_point, ref_points_sample[j])
            if dist < max_dist:
                group.append(rand_indices[j])
        if len(group) > 0:
            groups.append(group)
    return groups


def test():
    frames_seq_points = utils.get_frames_points()
    a, b = calc_transform_ransac(frames_seq_points[0], frames_seq_points[1])
    print(a)
    print(b)
