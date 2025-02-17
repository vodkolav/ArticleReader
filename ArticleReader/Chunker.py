import numpy as np
import re
import pandas as pd

class Chunker:
    # Split simple text into chunks
    def __init__(self, max_len=100):
        self.max_len = max_len

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
                res += self.breakByPeriods(ss[i])
                # res += [ss[i]]
                # continue
            # if res and len(res[-1]) + 3 + len(ss[i]) < max_len:
            #   res[-1] = res[-1] + r"  \n" + ss[i]
            else:
                res += [ss[i]]
            res += ["  \n\n"]
        return [r for r in res if len(r) > 3]

    def split_text_into_chunks(self, text):
        ## temporary hack for citation shit
        text = re.sub(" ([\\.,])", r"\g<1>", text)

        # deal with i.e an e.g. that break sentence separation
        # text = text.replace("i.e.,", r"that is")
        # text = text.replace("e.g.,", r"for example")
        text = text.replace("cf.", r"confer")

        # print(text)

        rawchunks = self.breakByParagraphs(text)
        self.chunks = pd.DataFrame(rawchunks, columns=["sentence"]).reset_index()
        self.chunks["text_len"] = self.chunks.sentence.str.len()


    def get_batch_sorted(self, batch_size = 3, start=0):
        self.chunks.sort_values("text_len", ascending=False, inplace=True)        
        return self.chunks.iloc[start : start +batch_size].copy()
        
    def get_batch_chronological(self, batch_size = 3, start=0):                    
        return self.chunks.iloc[start : start +batch_size].copy()

    def get_test_batch(self, chunks=20, start=0):
        return self.chunks[start : start + chunks]

    def save_chunks_as_text(self, filename, chunks):
        with open(filename, "w+") as f:
            f.write("\n|".join(chunks))

    def testChunking(self, test, chunks):
        with open("test.txt", "w+") as f:
            f.write(test)
        with open("test_res.txt", "w+") as f:
            f.write(r"".join(chunks))