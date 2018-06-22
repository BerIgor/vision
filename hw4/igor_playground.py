import numpy as np
import cv2
from hw4 import utils


if __name__ == "__main__":
    full_frame_list = utils.get_all_frames(utils.get_pwd() + '/our_data/ariel.mp4')

    full_frame_mask_list = utils.get_all_frames(utils.get_pwd() + '/our_data/mask.avi')

    rows = full_frame_list[0].shape[1]
    cols = full_frame_list[0].shape[0]

    video_format = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(utils.get_pwd() + '/our_data/ariel_upright.avi', video_format, 30, (cols, rows))
    video_format = cv2.VideoWriter_fourcc(*"XVID")
    video_writer_mask = cv2.VideoWriter(utils.get_pwd() + '/our_data/mask2.avi', video_format, 30, (cols, rows))

    # flipped_frames = list()
    for i in range(len(full_frame_mask_list)):
        print(i)
        frame = full_frame_list[i]
        frame = np.transpose(frame, (1, 0, 2))
        frame = np.flip(frame, 1)

        # flipped_frames.append(frame)
        video_writer.write(np.uint8(frame))
        mask = np.flip(full_frame_mask_list[i], 1)
        video_writer_mask.write(np.uint8(mask))

        # print(np.shape(frame))
        # print(np.shape(mask))

    video_writer.release()
    video_writer_mask.release()
