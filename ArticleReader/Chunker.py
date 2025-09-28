import numpy as np
import re
import pandas as pd

class Chunker:

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_value):
        if new_value < 0:
            raise ValueError("Value cannot be negative.")
        self._batch_size = new_value


    # Split simple text into chunks
    def __init__(self, max_len=100):
        self.max_len = max_len
        self._batch_size = 1 # default
        self.limit = slice(None)

    def breakByWords(self, text):
        ss = re.split(r"[ \n]+", text)
        # ss = text.split(" ")
        for i in range(len(ss) - 1):
            ss[i] = ss[i] + " "
        res = []  # ws[0]]
        # print(len(ws))
        for i in range(len(ss)):
            if res and len(res[-1]) + 1 + len(ss[i]) < self.max_len:
                res[-1] = res[-1] + ss[i]
            else:
                res += [ss[i]]
        return res

    def breakByWordsEqual(self, text):
        """break a sentence into chunks of several words roughly equally, in terms of character length

        Raises:
            ValueError: maxlen must be longer than words

        Returns:
            list: text broken down to several texts
        """
        ss = re.split(r"[ \n]+", text)
        lens = np.array([len(s) + 1 for s in ss])  # lenghts of each word

        if max(lens) > self.max_len:
            raise ValueError("maxlen must be larger than lengths of words in text.")

        n = int(np.ceil(np.sum(lens) / self.max_len))  # decide how many chunks needed
        cs = int(np.floor(np.sum(lens) / n))  # decide avg size of chunk
        csum = np.cumsum(lens)

        # find cutoffs by which to distribute words into chunks
        cutoffs = [0]
        for i in range(1, n):
            x = [i * cs] * len(lens)
            a = np.abs(x - csum)
            cutoffs.append(np.argmin(a))  ##(y[y>=a]))
        cutoffs += [len(ss)]

        # distribute words into chunks
        chunks = [""] * (len(cutoffs) - 1)
        for i in range(len(cutoffs) - 1):
            f = cutoffs[i]
            t = cutoffs[i + 1]
            chunks[i] = " ".join(ss[f:t]) + " "
        return chunks

    def breakByCommas(self, text):
        ss = re.split(r"\,[ \n]?", text)
        # ss = text.split(", ")
        for i in range(len(ss) - 1):
            ss[i] = ss[i] + ", "

        # ss = [s+ r"," for s in ss]
        # print(ss)
        res = []  # [ss[0]]
        for i in range(len(ss)):
            if len(ss[i]) > self.max_len:
                res += self.breakByWordsEqual(ss[i])
                continue
            if res and len(res[-1]) + 2 + len(ss[i]) < self.max_len:
                res[-1] = res[-1] + ss[i]
            else:
                res += [ss[i]]
        return res

    def breakByPeriods(self, text):

        ss = re.split(r"\.[ \n]", text)
        # ss = ss[:-1]
        ss = [s + r". " for s in ss if len(s) > 3]
        # print(text)
        res = []  # ss[0]]
        for i in range(len(ss)):
            if len(ss[i]) > self.max_len:
                res += self.breakByCommas(ss[i])
                # res += [ss[i]]
                continue
            if res and len(res[-1]) + 2 + len(ss[i]) < self.max_len:
                res[-1] = res[-1] + ss[i]  # + r". "
            else:
                res += [ss[i]]
        return res

    def breakByParagraphs(self, text):
        # ss = re.split(r'(?:\.? +)?\n+', text)

        ss = re.split(r"(?: +)?\n{2,}", text)
        # ss = [s+ "\n" for s in ss]
        # print(ss)
        res = []  # ss[0]]
        for i in range(len(ss)):
            if ss and len(ss[i]) > self.max_len:
                par = self.breakByPeriods(ss[i])
                # res += [ss[i]]
                # continue
            # if res and len(res[-1]) + 3 + len(ss[i]) < max_len:
            #   res[-1] = res[-1] + r"  \n" + ss[i]
            else:
                par = [ss[i]]
            par[-1] += "  \n"
            res += par
        return [r for r in res if len(r) > 3]

    def split_text_into_chunks(self, text):
        ## temporary hack for citation shit
        text = re.sub(" ([\\.,])", r"\g<1>", text)

        # deal with i.e an e.g. that break sentence separation
        # text = text.replace("i.e.,", r"that is")
        # text = text.replace("e.g.,", r"for example")
        text = text.replace("cf.", r"confer")

        # print(text)

        self.chunks = self.breakByParagraphs(text)
        
        # self.chunks = pd.DataFrame(rawchunks, columns=["sentence"]).reset_index()
        # self.chunks["text_len"] = self.chunks.sentence.str.len()

    # in BEnchmark: 
    # fr = self.chunker.find_chunk("Figure four shows the evolution")
    def find_chunk(self, text):
        # find number of chunk which contains specific text
        # may be useful if you want to test on specific portion of file
        for i, ch in enumerate(self.chunks):
            if text in ch:
                return i
        return -1


    def feed_df_batches(self, ):
        l = len(self.chunks_df)
        n = self.batch_size
        for ndx in range(0, l, n):
            # TODO: use iloc? 
            fr, to = ndx , min(ndx + n, l)
            print(f"feeding chunks {fr} to {to} out of {l}")

            yield self.chunks_df.iloc[fr : to].copy()

    def sort_by_text_len(self):
        self.chunks_df.sort_values("text_len", ascending=False, inplace=True)  

    def as_pandas(self):
        chunks = pd.DataFrame(self.chunks, columns=["sentence"]).reset_index()
        chunks["text_len"] = chunks.sentence.str.len()
        return chunks

    def init_df(self):
        self.chunks_df = self.as_pandas()[self.limit]
        
    def get_chunks_sorted(self, n_chunks = 3, start=0):
        df = self.as_pandas().sort_values("text_len", ascending=False)        
        return df.iloc[start : start +n_chunks]
        
    def get_chunks_chronological(self, n_chunks = 3, start=0):   
        df = self.as_pandas()                 
        return df.iloc[start : start + n_chunks]

    def get_dbg_subset(self, chunks=20, start=0):
        return self.chunks[start : start + chunks]
    
    def get_all_chronological(self):
        return self.as_pandas()
    
    def get_chunks(self):
        return self.chunks

    def save_chunks_as_text(self, filename):
        with open(filename, "w+") as f:
            f.write("\n|".join(self.chunks))

    def testChunking(self, test, chunks):
        with open("test.txt", "w+") as f:
            f.write(test)
        with open("test_res.txt", "w+") as f:
            f.write(r"".join(chunks))