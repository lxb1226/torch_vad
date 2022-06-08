
import librosa
import numpy as np


def convert_times_to_labels(times):
    pass


def parse_vad_label(speechs, frame_size=0.01, frame_shift=0.005):
    frame2time = lambda n: n * frame_shift + frame_size / 2  # 除以2是判断中心点是不是在某段里
    frames = []
    frame_n = 0
    for speech in speechs:
        start, end = speech
        assert end >= start, (start, end)
        if end == start:
            pass
        while frame2time(frame_n) < start:
            frames.append(0)
            frame_n += 1
        while frame2time(frame_n) <= end:
            frames.append(1)
            frame_n += 1
    return frames


def extract_feature(audio, mel_args):
    # MEL_ARGS = {
    #     'n_mels': ARGS.n_mels,
    #     'n_fft': 2048,
    #     'hop_length': int(ARGS.sr * ARGS.hoplen / 1000),
    #     'win_length': int(ARGS.sr * ARGS.winlen / 1000)
    # }

    EPS = np.spacing(1)

    lms_feature = np.log(librosa.feature.melspectrogram(audio, **mel_args) + EPS).T

    return lms_feature
