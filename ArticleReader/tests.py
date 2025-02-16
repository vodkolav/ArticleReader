from ArticleReader.Chunker import Chunker 
import numpy as np 

test = 'It also provides training recipes pretrained models and inference scripts for popular speech datasets as well \
as tutorials which allow anyone with basic Python proficiency to familiarize themselves with speech technologies'

def BBWtst(text, maxlen = 100):
  print("\nmaxlen: ", maxlen)
  C = Chunker()
  res = C.breakByWordsEqual(text ,maxlen )
  [print(r) for r in res]
  res = " ".join(res)
  # print(text)
  # print(res)
  assert text == res

smp = np.random.normal(100, 200, 30).astype(int)
smp = smp[smp > 20]
smp = smp[:20]


for i in smp:
  BBWtst(test,i)

print("tests passed!")