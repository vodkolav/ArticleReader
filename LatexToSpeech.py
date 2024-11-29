
from pylatexenc import latexwalker, latex2text, macrospec

import torch
import torchaudio
from speechbrain.inference import Tacotron2, HIFIGAN

import re





class LatexParser:
  #
  # Define macros, environments, specials for the *parser*
  #
  def __init__(self):     
    self.ignore = [("PassOptionsToPackage","{{"),
              ("bibliographystyle","{"),
              ("newfloatcommand","{{[["),
              ("setlist","[{"),
              ("newcolumntype","{[{"),
              ("floatstyle","{"),
              ("restylefloat","{"),
              ("affil","[{"),
              ("author","[{"),
              ("lstdefinestyle","{{"),
              ("lstset","{"),
              ("lstdefinelanguage","{{"),
              ('cite' , "{"), # '<cit.>'
              ('citet', "{"),
              ('citep', "{"),
              ('CC', "{")
              #("def","[{")
    ]

    self.lw_context_db = latexwalker.get_default_latex_context_db()
    self.lw_context_db.add_context_category(
        'my-quotes',
        prepend=True,
        macros=[macrospec.MacroSpec(nam, pars) for (nam,pars) in self.ignore]
    )

    self.l2t_context_db = latex2text.get_default_latex_context_db()
    self.l2t_context_db.add_context_category(
      'my-quotes',
      prepend=True,
      macros=[latex2text.MacroTextSpec(nam, simplify_repl=r'') for (nam,pars) in self.ignore] + 
              [latex2text.MacroTextSpec('maketitle', lambda n, l2tobj: 
                                        getattr(l2tobj, '_doc_title', r'[NO \title GIVEN]'))
              #,latex2text.MacroTextSpec('CC', 'C++')
              ]
  )

  def read_latex(self, filepath):
    with open(filepath)as f:
      content = f.read()
    return content

  #
  # Implement macros, environments, specials for the *conversion to text*
  #

  def _get_optional_arg(self, node, default, l2tobj):
      """Helper that returns the `node` converted to text, or `default`
      if the node is `None` (e.g. an optional argument that was not
      specified)"""
      if node is None:
          return default
      return l2tobj.nodelist_to_text([node])

  def put_in_quotes_macro_repl(self, n, l2tobj):
      """Get the text replacement for the macro
      \putinquotes[open-quote][close-quote]{text}"""
      if not n.nodeargd:
          # n.nodeargd can be empty if e.g. \putinquotes was a single
          # token passed as an argument to a macro,
          # e.g. \newcommand\putinquotes...
          return ''
      open_q_s = self._get_optional_arg(n.nodeargd.argnlist[0], '“', l2tobj)
      close_q_s = self._get_optional_arg(n.nodeargd.argnlist[1], '”', l2tobj)
      return (open_q_s + l2tobj.nodelist_to_text([n.nodeargd.argnlist[2]])
              + close_q_s)

  def in_quotes_env_repl(self, n, l2tobj):
      """Get the text replacement for the {inquotes} environment"""
      open_q_s = self._get_optional_arg(n.nodeargd.argnlist[0], '“', l2tobj)
      close_q_s = self._get_optional_arg(n.nodeargd.argnlist[1], '”', l2tobj)
      return open_q_s + l2tobj.nodelist_to_text(n.nodelist) + close_q_s


  def do_author(n, l2tobj):
      """Get the text replacement for the macro
      \putinquotes[open-quote][close-quote]{text}"""
      if not n.nodeargd:
          # n.nodeargd can be empty if e.g. \putinquotes was a single
          # token passed as an argument to a macro,
          # e.g. \newcommand\putinquotes...
          return ''
      #open_q_s = _get_optional_arg(n.nodeargd.argnlist[0], '“', l2tobj)
      
      return "" 


  #l2t_context_db.set_unknown_macro_spec()


  def custom_latex_to_text(self, input_latex ):
      # the latex parser instance with custom latex_context
      lw_obj = latexwalker.LatexWalker(input_latex,
                                      latex_context=self.lw_context_db)
      # parse to node list
      nodelist, pos, length = lw_obj.get_latex_nodes()
      # initialize the converter to text with custom latex_context
      l2t_obj = latex2text.LatexNodes2Text(latex_context=self.l2t_context_db)
      # convert to text
      processed = l2t_obj.nodelist_to_text( nodelist )
      return self.postprocess(processed)



  def postprocess(self, processed):
    # Temporary hacks: manually remove all that latex parser didn't catch  
    processed = processed.replace("C⁠[4].4ex++","").strip()
    processed = re.sub(r" {2,}", " ", processed)
    return processed
  
  def save_text(text):
    with open("processed.txt", "w+")as f:
      f.write(text)


  #Split simple text into chunks 

class Chunker:

  def breakByWords(self, text, max_len=100):
    ws = text.split(" ")
    for i in range(len(ws)-1):
      ws[i] = ws[i] + " " 
    res = [] #ws[0]]  
    #print(len(ws))
    for i in range(len(ws)):
      if res and len(res[-1]) + 1 + len(ws[i]) < max_len:
        res[-1] = res[-1] + ws[i]
      else:
        res += [ws[i]]
    return res

  def breakByCommas(self, text, max_len=100):  
    ss = text.split(", ")
    for i in range(len(ss)-1):
      ss[i] = ss[i] + ", " 

    #ss = [s+ r"," for s in ss]
    #print(ss)
    res = [] # [ss[0]]
    for i in range(len(ss)):
      if len(ss[i])>max_len:
        res += self.breakByWords(ss[i])
        continue
      if res and len(res[-1]) + 2 + len(ss[i]) < max_len:
        res[-1] = res[-1]+ ss[i]
      else:
        res += [ss[i]]
    return res

  def breakByPeriods(self, text, max_len=100):
    
    ss = re.split(r'\. ?', text)
    #ss = ss[:-1]
    ss = [s+ r". " for s in ss if len(s)>3]
    print(text)
    res = [] #ss[0]]
    for i in range(len(ss)):
      if len(ss[i])>max_len:
        res += self.breakByCommas(ss[i])
        #res += [ss[i]]
        continue
      if res and len(res[-1]) + 2 + len(ss[i]) < max_len:
        res[-1] = res[-1] + ss[i] #+ r". "
      else:
        res += [ss[i]]
    return res

  def breakByParagraphs(self, text, max_len=100):
    #ss = re.split(r'(?:\.? +)?\n+', text) 
    
    ss = re.split(r'(?: +)?\n+', text) 
    #ss = [s+ "\n" for s in ss]
    print(ss)
    res = [] #ss[0]]
    for i in range(len(ss)):
      if ss and len(ss[i])>max_len:
        res += self.breakByPeriods(ss[i])
        #res += [ss[i]]
        #continue
      # if res and len(res[-1]) + 3 + len(ss[i]) < max_len:
      #   res[-1] = res[-1] + r"  \n" + ss[i]
      else:
        res += [ss[i]]
      res +=  ["  \n\n"]
    return [r for r in res if len(r)>3]


  def split_text_into_chunks(self, text, max_length=100):
      ## temporary hack for citation shit
      text = re.sub(" ([\.,])", r"\g<1>", text)
      #print(text)

      self.chunks =  self.breakByParagraphs(text, max_len=max_length)

  def get_test_batch(self, chunks = 20):
    return self.chunks[:chunks]

  def save_chunks_as_text(self, filename, chunks):
    with open(filename, 'w+') as f: 
      f.write(r"".join(chunks))



  def testChunking(self, test, chunks):
    with open('test.txt', 'w+') as f: 
      f.write(test)
    with open('test_res.txt', 'w+') as f: 
      f.write(r"".join(chunks))




class Narrator:
  def __init__(self):
     self.loadModels()

  def loadModels(self):
    # Load SpeechBrain models
    self.tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
    self.hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")


  def orderUnorder(self,chunks, itemFn = len, reverse=True):
    """sort a list while preserving  info for restoring original order

    Args:
        chunks (list): list to be sorted
        itemFn (function, optional): function to apply to each item before the sort. Defaults to len.
        reverse (bool, optional): sort descending. Defaults to True.

    Returns:
        (order, unorder): index that sorts the list, index that restores original order
    """

    order = sorted(range(len(chunks)), key=lambda k: itemFn(chunks[k]), reverse=reverse)

    unorder = [(i,order[i]) for i in range(len(order))]
    unorder = sorted(unorder, key=lambda k: k[1], reverse=not reverse)
    unorder = [i[0] for i in unorder]
    return order, unorder



  def text_to_speech(self, chunks):      

      #  must  sort chunks by length descending

      order, unorder = self.orderUnorder(chunks)

      ordered = [chunks[i] for i in order]

      #print(ordered)
      #return ordered, ordered, ordered
      print("run tacotron")
      mel_outputs, mel_lengths, alignments = self.tacotron2.encode_batch(ordered)

      print(mel_lengths)
      

      print("run hifigan")
      hop_len = 256
      waveforms = self.hifi_gan.decode_batch(mel_outputs, mel_lengths, hop_len).squeeze(1)

      # add pause between paragraphs
      mml = torch.ones_like(mel_lengths) * max(mel_lengths)
      pause_dur = 40
      pause = torch.tensor([ pause_dur if "\n\n" in c else 0 for c in ordered])

      mel_lengths += pause
      mel_lengths =  torch.min(mel_lengths, mml)
      #[min(mml,ml + p) for ml,p in zip(mel_lengths, pause)] 


      # turn into array
      arr = torch.tensor_split(waveforms, len(order), dim=0)

      #cut padding
      arr = [a[:,:l]  for a,l in zip(arr, mel_lengths * hop_len) ]

      # restore order 
      arr = [arr[i] for i in unorder]

      waveform = torch.cat(arr, dim=1)

      return waveform, order, alignments


  def save_audio(self, output_wav, waveform):
    torchaudio.save(output_wav, waveform, 22050, format="wav")
    print(f"Audio saved to {output_wav}")



def main():

  input_file = "arXiv-2106.04624v1/main.tex"
  output_file = "output/28.11.6"

  parser = LatexParser()
  content = parser.read_latex(input_file)
  processed = parser.custom_latex_to_text(content)

  chunker = Chunker()
  chunker.split_text_into_chunks(processed)
  chunks = chunker.get_test_batch(50)
  chunker.save_chunks_as_text(output_file + ".md", chunks)


  narrator = Narrator()
  waveform, order, alignments = narrator.text_to_speech(chunks)  


  narrator.save_audio(output_file + ".wav", waveform)


if __name__ == "__main__":
  main()