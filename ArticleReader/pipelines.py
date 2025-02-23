

from ArticleReader.Chunker import Chunker 
from ArticleReader.Narrator import Narrator
from ArticleReader.LatexToSpeech import LatexParser
import torch
from datetime import datetime


def basicPipeline(test_batch_size = 10):
    input_file = "data/arXiv-2106.04624v1/main.tex"
    output_file = "output/" + datetime.now().strftime(r"%y.%m.%d-%H")

    parser = LatexParser()
    content = parser.read_latex(input_file)
    processed = parser.custom_latex_to_text(content)
    parser.save_text(processed, "dbg/spec_my.txt")

    tables = parser.get_tables()
    parser.save_text(tables, "dbg/tables.tex")

    chunker = Chunker(max_len=200)
    chunker.split_text_into_chunks(processed)
    chunks = chunker.get_test_batch(test_batch_size, 0)
    # chunks = chunker.chunks
    chunker.save_chunks_as_text(output_file + ".md", chunks)
    print("text chunks:", [len(ch) for ch in chunks])

    narrator = Narrator()
    waveforms, durations = narrator.text_to_speech_batched(chunks)
    durations_sec = durations / 22050.0

    print("durations: ", durations_sec)

    waveform = torch.cat(waveforms, dim=1)

    print("saving audio")
    narrator.save_audio(output_file + ".wav", waveform)

    narrator.save_video(output_file)

    narrator.generate_srt(chunks, durations_sec, output_file + ".srt")


def main():
    basicPipeline()

if __name__ == "__main__":
    main()
