from typing import List

import numpy as np


def frame2time(n: int, frame_size: float = 0.032, frame_shift: float = 0.008
               ) -> float:
    return n * frame_shift + frame_size / 2


def prediction_to_vad_label(
        prediction: List[float],
        frame_size: float = 0.032,
        frame_shift: float = 0.009,
) -> List[List[float]]:
    speech_frames = []
    prev = 0

    start, end = 0, 0
    for i, pred in enumerate(prediction):

        if prev != pred:
            if prev == 0:
                # 0 -> 1
                start = i
            else:
                # 1 -> 0
                end = i
                speech_frames.append(
                    [frame2time(start, frame_size, frame_shift), frame2time(end, frame_size, frame_shift)])
        elif i == len(prediction) - 1:
            end = i
            speech_frames.append(
                [frame2time(start, frame_size, frame_shift), frame2time(end, frame_size, frame_shift)])
        prev = pred

    return speech_frames


# def prediction_to_vad_label(
#         prediction: List[float],
#         frame_size: float = 0.032,
#         frame_shift: float = 0.008,
# ):
#     """Convert model prediction to VAD labels.
#
#     Args:
#         prediction (List[float]): predicted speech activity of each **frame** in one sample
#             e.g. [0.01, 0.03, 0.48, 0.66, 0.89, 0.87, ..., 0.72, 0.55, 0.20, 0.18, 0.07]
#         frame_size (float): frame size (in seconds) that is used when
#                             extracting spectral features
#         frame_shift (float): frame shift / hop length (in seconds) that
#                             is used when extracting spectral features
#         threshold (float): prediction values that are higher than `threshold` are set to 1,
#                             and those lower than or equal to `threshold` are set to 0
#     Returns:
#         vad_label (str): converted VAD label
#             e.g. "0.31,2.56 2.6,3.89 4.62,7.99 8.85,11.06"
#
#     NOTE: Each frame is converted to the timestamp according to its center time point.
#     Thus the converted labels may not exactly coincide with the original VAD label, depending
#     on the specified `frame_size` and `frame_shift`.
#     See the following example for more detailed explanation.
#
#     Examples:
#         >>> label = parse_vad_label("0.31,0.52 0.75,0.92")
#         >>> prediction_to_vad_label(label)
#         '0.31,0.53 0.75,0.92'
#     """
#     frame2time = lambda n: n * frame_shift + frame_size / 2
#     speech_frames = {"speech_times": []}
#
#     prev_state = False
#     start, end = 0, 0
#     end_prediction = len(prediction) - 1
#     for i, pred in enumerate(prediction):
#         state = pred > threshold
#         if not prev_state and state:
#             # 0 -> 1
#             start = i
#         elif not state and prev_state:
#             # 1 -> 0
#             end = i
#             speech_frames["speech_times"].append({"start_time": frame2time(start), "end_time": frame2time(end)})
#             # speech_frames.append(
#             #     "{:.3f},{:.3f}".format(frame2time(start), frame2time(end))
#             # )
#         elif i == end_prediction and state:
#             # 1 -> 1 (end)
#             end = i
#             speech_frames["speech_times"].append({"start_time": frame2time(start), "end_time": frame2time(end)})
#             # speech_frames.append(
#             #     "{:.3f},{:.3f}".format(frame2time(start), frame2time(end))
#             # )
#         prev_state = state
#     return speech_frames


def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder

    Reason is:
    1. This does not belong to a class necessarily
    2. Import DecisionEncoder requires sndfile over some other imports..which causes some problems on clusters

    """

    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


if __name__ == '__main__':
    preds = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    res = prediction_to_vad_label(preds)
    print(res)
