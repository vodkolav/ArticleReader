
import torch

from datetime import datetime, timedelta
from ArticleReader.NodesVisitor import Extractor

class Narrator:
    def __init__(self, profiling=0):
        self.loadModels()
        self.hop_len = 256
        if profiling:
            from Benchmarking import MemoryMonitor
            self.profilers = {}

            profiler1 = MemoryMonitor(stage="tts")
            self.tacotron2.encode_batch = profiler1.monitor_memory_decorator(self.tacotron2.encode_batch)
            self.profilers["tacotron"] = profiler1

            profiler2 = MemoryMonitor(stage="vocoder")
            self.hifi_gan.decode_batch = profiler2.monitor_memory_decorator(self.hifi_gan.decode_batch)
            self.profilers["vocoder"] = profiler2

    def loadModels(self):
        # Load SpeechBrain models
        self.tacotron2 = Tacotron2.from_hparams(
            source="speechbrain/tts-tacotron2-ljspeech",
            savedir="checkpoints/tacotron2",
            overrides={"max_decoder_steps": 2000},
        )
        self.hifi_gan = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech", savedir="checkpoints/hifigan"
        )

    def orderUnorder(self, chunks, itemFn=len, reverse=True):
        """sort a list while preserving  info for restoring original order

        Args:
            chunks (list): list to be sorted
            itemFn (function, optional): function to apply to each item before the sort. Defaults to len.
            reverse (bool, optional): sort descending. Defaults to True.

        Returns:
            (order, unorder): index that sorts the list, index that restores original order
        """

        order = sorted(
            range(len(chunks)), key=lambda k: itemFn(chunks[k]), reverse=reverse
        )

        unorder = [(i, order[i]) for i in range(len(order))]
        unorder = sorted(unorder, key=lambda k: k[1], reverse=not reverse)
        unorder = [i[0] for i in unorder]
        return order, unorder

    def seq_len(self, item):
        return self.tacotron2.text_to_seq(item)[1]

    def add_pauses(self, ordered, mel_lengths, pause_dur=40):
        # add pause between paragraphs
        mml = torch.ones_like(mel_lengths) * max(mel_lengths)

        pause = torch.tensor([pause_dur if "\n\n" in c else 0 for c in ordered])

        mel_lengths += pause
        mel_lengths = torch.min(mel_lengths, mml)
        return mel_lengths
        # [min(mml,ml + p) for ml,p in zip(mel_lengths, pause)]

    def text_to_speech(self, chunks):

        #  must  sort chunks by length descending

        order, unorder = self.orderUnorder(chunks, self.seq_len)

        ordered = [chunks[i] for i in order]

        # print(ordered)
        # return ordered, ordered, ordered
        print("run tacotron")
        mel_outputs, mel_lengths, alignments = self.tacotron2.encode_batch(ordered)

        if self.tacotron2.hparams.max_decoder_steps in mel_lengths:
            Warning("We probably have truncated chunks")

        print("run vocoder")
        hop_len = 256
        waveforms = self.hifi_gan.decode_batch(
            mel_outputs, mel_lengths, hop_len
        ).squeeze(1)

        print("adding pauses")
        mel_lengths = self.add_pauses(ordered, mel_lengths, pause_dur=40)

        print("recombine")
        # turn into array
        arr = torch.tensor_split(waveforms, len(order), dim=0)

        # cut padding
        arr = [a[:, :l] for a, l in zip(arr, mel_lengths * hop_len)]

        # restore order
        arr = [arr[i] for i in unorder]

        mel_lengths = mel_lengths.detach().numpy()
        mel_lengths = [mel_lengths[i] for i in unorder]

        return arr, mel_lengths, hop_len

    def infer(self, batch):
        # incoming: batch of chunks (~sentences)
        print("run tacotron")
        mel_outputs, mel_lengths, alignments = self.tacotron2.encode_batch(batch)

        if self.tacotron2.hparams.max_decoder_steps in mel_lengths:
            Warning("We probably have truncated chunks")

        print("run vocoder")

        waveforms = self.hifi_gan.decode_batch(
            mel_outputs, mel_lengths, self.hop_len
        )  # .squeeze(1)
        # out: batch of waveforms, mel_lengths
        return waveforms, mel_lengths

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx : min(ndx + n, l)]

    def text_to_speech_batched(self, chunks):

        #  must  sort chunks by length descending

        order, unorder = self.orderUnorder(chunks, self.seq_len)

        ordered = [chunks[i] for i in order]

        ordered_wf = []  # torch.tensor([])

        ordered_mel_lens = torch.tensor([])

        batchSize = 50

        for b in self.batch(ordered, batchSize):

            waveforms, mel_lengths = self.infer(b)
            ordered_wf += waveforms
            print("adding pauses")
            mel_lengths = self.add_pauses(b, mel_lengths, pause_dur=40)
            ordered_mel_lens = torch.cat((ordered_mel_lens, mel_lengths))

        print("recombine")
        # turn into array
        # ordered_wf = torch.tensor_split(ordered_wf, len(order), dim=0)

        # cut padding
        ordered_wf = [
            a[:, : l.int()] for a, l in zip(ordered_wf, ordered_mel_lens * self.hop_len)
        ]

        # restore order
        unordered_wf = [ordered_wf[i] for i in unorder]

        ordered_mel_lens = ordered_mel_lens.detach().numpy()
        ordered_durations = ordered_mel_lens * self.hop_len

        unordered_durations = ordered_durations[unorder]

        return (
            unordered_wf,
            unordered_durations,
        )

    def save_audio(self, output_wav, waveform):
        torchaudio.save(output_wav, waveform, 22050, format="wav")
        print(f"Audio saved to {output_wav}")

    def save_video(self, output_file):
        # ffmpeg -loop 1 -i image.png -i 24.12.08-17.wav -c:v libx264 -tune stillimage -c:a aac -b:a 192k -pix_fmt yuv420p -shortest output.mp4

        import os

        print("\n saving static video")
        command = f"""
  ffmpeg -y -loop 1 -i output/image.png -i {output_file}.wav 
  -c:v libx264 -tune stillimage -c:a aac -b:a 
  192k -pix_fmt yuv420p -shortest {output_file}.mp4
  """
        command = command.replace("\n", "")
        print(command)
        os.system(command)