
import torch
import torchaudio
from speechbrain.inference import Tacotron2, HIFIGAN
from datetime import timedelta
import pandas as pd

class Narrator:
    def __init__(self, tts_model = None, vocoder_model = None):
        self.loadModels(tts_model, vocoder_model)
        self.hop_len = 256 # this should be coming from model hparams


    def loadModels(self, tts_model, vocoder_model):
        # Load SpeechBrain models
        provider = "speechbrain"
        if tts_model is None: #load defalt mopdel
            model_name = "tts-tacotron2-ljspeech"
            self.tts = Tacotron2.from_hparams(
                source=f"{provider}/{model_name}",
                savedir=f"checkpoints/{model_name}",
                overrides={"max_decoder_steps": 2000},
            )
        else:
            self.tts = tts_model

        if vocoder_model is None: #load defalt mopdel    
            model_name = "tts-hifigan-ljspeech"
            self.vocoder = HIFIGAN.from_hparams(
                source=f"{provider}/{model_name}",
                savedir=f"checkpoints/{model_name}"
            )
        else: 
            self.vocoder = vocoder_model


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
        return self.tts.text_to_seq(item)[1]

    def add_pauses(self, ordered, mel_lengths, pause_dur=40):
        # add pause between paragraphs
        # TODO: implement different pauses and other prosody for different cases 
        mml = torch.ones_like(mel_lengths) * max(mel_lengths)

        pause = torch.tensor([pause_dur if "\n\n" in c else 0 for c in ordered])

        mel_lengths += pause
        mel_lengths = torch.min(mel_lengths, mml)
        return mel_lengths
        # [min(mml,ml + p) for ml,p in zip(mel_lengths, pause)]

    def text_to_speech_df(self, batch_df):# (self, batch_df: pd.DataFrame):

        # ensure sentences are sorted by seq_len
        batch_df.loc[:,"seq_len"] = batch_df.sentence.map(self.seq_len)
        batch_df.sort_values("seq_len", ascending=False, inplace=True)
        
        # TODO: ensure batch_df has column index for restoration of order later 
        #       ensure batch_df is a proper DF, not a view of another DF

        waveforms, mel_lengths = self.infer(batch_df.sentence)
        
        # defining pauses between paragraphs                
        mel_lengths = self.add_pauses(batch_df.sentence, mel_lengths, pause_dur=40)        

        # turning tensor into regular array
        arr = torch.tensor_split(waveforms.squeeze(1), len(waveforms), dim=0)

        # TODO: something fishy is going on here. the [b,1,smaples] tensor is cut into
        # array of 0-dim tensors. might affect performance, need to check that

        # cut padding
        arr = [a[:, :l] for a, l in zip(arr, mel_lengths * self.hop_len)]
        
        mel_lengths = mel_lengths.detach().numpy()
        batch_df["waveform"] = arr
        batch_df["mel_lengths"] = mel_lengths
        batch_df["durations_sec"] = mel_lengths / 22050.0
        return batch_df

    def text_to_speech(self, batch):

        #  must  sort chunks by length descending

        order, unorder = self.orderUnorder(batch, self.seq_len)

        ordered = [batch[i] for i in order]

        waveforms, mel_lengths = self.infer(batch)

        print("adding pauses")
        mel_lengths = self.add_pauses(ordered, mel_lengths, pause_dur=40)

        print("recombine")
        # turn into array
        arr = torch.tensor_split(waveforms, len(order), dim=0)

        # cut padding
        arr = [a[:, :l] for a, l in zip(arr, mel_lengths * self.hop_len)]

        # restore order
        arr = [arr[i] for i in unorder]

        mel_lengths = mel_lengths.detach().numpy()
        mel_lengths = [mel_lengths[i] for i in unorder]

        return arr, mel_lengths, self.hop_len

    def infer(self, batch):
        # incoming: batch of chunks (~sentences)
        print("     running TTS model")
        print("     batch size: ", len(batch))
        output = self.tts.encode_batch(batch)
        
        if output is not None: 
            mel_outputs, mel_lengths, alignments = output
            print("     TTS model finished")
            if self.tts.hparams.max_decoder_steps in mel_lengths:
                Warning("       We probably have truncated chunks")

            print("     running vocoder model")
            waveforms = self.vocoder.decode_batch(
                mel_outputs, mel_lengths, self.hop_len          
            )  # .squeeze(1)                  

            if waveforms is not None:
                print("     vocoder model finished")   
                # out: batch of waveforms, mel_lengths
                return waveforms, mel_lengths    
            else:                       
                print("     vocoder failed. returning silence")
                # return zeros tensor of expected size
                #waveforms = torch.zeros(batch.shape[0],1,max(mel_lengths) * self.hop_len)
        else: 
            print("     tts model failed. skipping vocoder stage, returning silence")
            mel_lengths = torch.ones(len(batch)) * 256 # just arbitrary number
            # return zeros tensor of expected size        
        #TODO: maybe handle output of failed runs without garbage data (the zeros tensor)
        waveforms = torch.zeros(batch.shape[0],1,int(max(mel_lengths)) * self.hop_len)        
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



    def generate_srt(self, sentences, durations, output_file="output.srt"):
        """
        Generates an SRT file from a list of sentences and durations.

        Args:
            sentences (list of str): The sentences to be shown as subtitles.
            durations (list of int): Durations (in seconds) for each sentence in chronological order.
            output_file (str): Path to save the SRT file.

        Returns:
            None
        """

        def format_timestamp(seconds):
            """Formats seconds into SRT timestamp format."""
            td = timedelta(seconds=seconds)
            total_seconds = int(td.total_seconds())
            milliseconds = int(td.microseconds / 1000)
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

        with open(output_file, "w") as file:
            current_time = 0  # Start time in seconds
            for i, (sentence, duration) in enumerate(zip(sentences, durations), start=1):
                start_time = format_timestamp(current_time)
                end_time = format_timestamp(current_time + duration)
                current_time += duration  # Increment time by the duration of the sentence

                # Write SRT entry
                file.write(f"{i}\n")
                file.write(f"{start_time} --> {end_time}\n")
                file.write(f"{sentence.strip()}\n\n")

        print(f"SRT file saved to {output_file}")


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