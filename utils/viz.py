import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from .vad_utils import prediction_to_vad_label


def show_result(data, labels, sr):
    speech_times = prediction_to_vad_label(labels)
    plot_wave_and_label(data, speech_times, sr)


def plot_wave_and_label(data, speech_times, sr):
    plt.figure(figsize=(15, 10))
    sns.set()
    sns.lineplot(x=[i / sr for i in range(len(data))], y=data)

    start, end = 0, 0
    for time in speech_times:
        plt.axvspan(end, time[0], alpha=0.5, color="r")
        start, end = time[0], time[1]
        plt.axvspan(start, end, alpha=0.5, color="g")
    plt.axvspan(end, (len(data) - 1) / sr, alpha=0.5, color="r")

    plt.title(f"Sample number  with speech in green", size=20)
    plt.xlabel("Time (s)", size=20)
    plt.ylabel("Amplitude", size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()


if __name__ == '__main__':
    audio_path = r'F:\workspace\GHT\projects\vad\data\dataset\val\6411-58876-0056.wav'
    speech_times = [[0.14, 3.76], [3.98, 5.80], [6.36, 11.06], [11.72, 12.82], [12.96, 15.36]]
    wav_data, sr = librosa.load(audio_path, sr=8000)
    plot_wave_and_label(wav_data, speech_times, sr)
