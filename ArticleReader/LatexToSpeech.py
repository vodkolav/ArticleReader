
from pylatexenc import latexwalker, latex2text, macrospec

import torch
import torchaudio
from speechbrain.inference import Tacotron2, HIFIGAN
import numpy as np 
import re
from datetime import datetime as dt
from NodesVisitor import Extractor



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
              ('footnote', '{'),
              ('cite' , "{"), # '<cit.>'
              ('citet', "{"),
              ('citep', "{"),
              #("def","[{")
    ]

    self.lw_context_db = latexwalker.get_default_latex_context_db()
    self.lw_context_db.add_context_category(
        'my-quotes',
        prepend=True,
        macros=[macrospec.MacroSpec(nam, pars) for (nam,pars) in self.ignore]+
               [macrospec.MacroSpec("CC", "{")],
        environments= [macrospec.EnvironmentSpec('table', '['),
                       macrospec.EnvironmentSpec('figure', '[')]
    )

    self.l2t_context_db = latex2text.get_default_latex_context_db()
    self.l2t_context_db.add_context_category(
      'my-quotes',
      prepend=True,
      macros=[latex2text.MacroTextSpec(nam, simplify_repl=r'') for (nam,pars) in self.ignore] + 
             [latex2text.MacroTextSpec('url', '%s'),
              latex2text.MacroTextSpec('maketitle', self.maketitle),
              latex2text.MacroTextSpec('ref', simplify_repl= self.fmt_ref_macro_node),
              latex2text.MacroTextSpec('section',self.fmt_subsections),
              latex2text.MacroTextSpec('subsection',self.fmt_subsections),
              latex2text.MacroTextSpec('subsubsection',self.fmt_subsections),
              latex2text.MacroTextSpec("CC", "si plus plus")
              #,latex2text.MacroTextSpec('CC', 'C++')
              ],
      environments= [latex2text.EnvironmentTextSpec("table", simplify_repl=self.fmt_table_environment_node, discard=False),
                     latex2text.EnvironmentTextSpec("figure", simplify_repl=self.fmt_figure_environment_node, discard=False),
                     # TODO: math mode
                     #latex2text.EnvironmentTextSpec("tabular", simplify_repl=self.fmt_table_environment_node)
                     ]
                     )
    self.tables = []
    self.figures = []

  def maketitle(self, node, l2tobj):
    return getattr(l2tobj, '_doc_title', r'[NO \title GIVEN]') + ".  "



  def read_latex(self, filepath, size = -1):
    with open(filepath)as f:
      content = f.read(size)
    return content

  def fmt_table_environment_node(self, node, l2tobj):

    #if node.environmentname == 'table':
    nv = Extractor()
    res = nv.start(node)
    if res: 
      table_name = res['visited_results_body'][0]['label']
      #table_name = table_name.replace(":","_")
      #print(res)
      found = { "label": f"{table_name}", "content": node.latex_verbatim()}
      self.tables.append(found)


      # content  =  #l2tobj.nodelist_to_text(node.nodelist)
      # self.save_text(content, dest)
      return "" #"LatexTable: " + table_name
    return "" #"LatexTable: ERROR" 

  def fmt_figure_environment_node(self, node, l2tobj):

    #if node.environmentname == 'table':
    nv = Extractor()
    res = nv.start(node)
    if res: 
      figure_name = res['visited_results_body'][0]['label']
      #table_name = table_name.replace(":","_")
      #print(res)
      found = { "label": f"{figure_name}", "content": node.latex_verbatim()}
      self.figures.append(found)


      # content  =  #l2tobj.nodelist_to_text(node.nodelist)
      # self.save_text(content, dest)
      return "" #"LatexFigure: " + figure_name
    return "" #"LatexFigure: ERROR" 

  def fmt_ref_macro_node(self, node, l2tobj):
    res = node.nodeargs[0].nodelist.nodelist[0].chars
    
    if res: 
      # self.save_text(content, dest)
      return "one" #"LatexRef: " + res
    return "oops" #"LatexRef: ERROR" 


  def fmt_subsections(self, node, l2tobj): 
    res = l2tobj.node_arg_to_text(node, 2)
    if res: 
      return node.macroname.capitalize() + ": " + res + ".\n"
    return "" #"LatexSubsection: ERROR" 

  def get_tables(self):
    #TODO: convert to images:
    # https://tex.stackexchange.com/questions/11866/compile-a-latex-document-into-a-png-image-thats-as-short-as-possible
    tables = [ f" % {t['label']} \n {t['content']} " for t in self.tables]
    return "\n".join(tables)
  
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
    # Temporary hacks: 
    # manually remove all that latex parser didn't catch  
    processed = processed.replace("C⁠[4].4ex++","").strip()
    #processed = processed.replace("si plus plus","").strip()
    
    # shorten long spaces to just one
    processed = re.sub(r" {2,}", " ", processed)
    return processed
  
  def save_text(self, text, filename):
    with open(filename, "w+")as f:
      f.write(text)


  #Split simple text into chunks 

class Chunker:

  def __init__(self, max_len=100):
    self.max_len = max_len

  def breakByWords(self, text):
    ss = re.split(r'[ \n]+', text)
    #ss = text.split(" ")
    for i in range(len(ss)-1):
      ss[i] = ss[i] + " " 
    res = [] #ws[0]]  
    #print(len(ws))
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
    ss = re.split(r'[ \n]+', text)  
    lens = np.array([len(s)+1 for s in ss]) # lenghts of each word

    if (max(lens)> self.max_len):
      raise ValueError("maxlen must be larger than lengths of words in text.")
    
    n = int(np.ceil(np.sum(lens) / self.max_len))  # decide how many chunks needed
    cs = int(np.floor(np.sum(lens) / n)) # decide avg size of chunk
    csum = np.cumsum(lens)

    # find cutoffs by which to distribute words into chunks 
    cutoffs = [0]
    for i in range(1,n):
      x = [i*cs] * len(lens)
      a = np.abs(x - csum)
      cutoffs.append(np.argmin(a)) ##(y[y>=a]))
    cutoffs += [len(ss)]

    # distribute words into chunks
    chunks = [''] * (len(cutoffs) - 1)
    for i in range(len(cutoffs)-1):
      f = cutoffs[i]
      t = cutoffs[i+1]
      chunks[i] = " ".join(ss[f:t]) + " "
    return chunks


  def breakByCommas(self, text):  
    ss = re.split(r'\,[ \n]?', text)
    #ss = text.split(", ")
    for i in range(len(ss)-1):
      ss[i] = ss[i] + ", " 

    #ss = [s+ r"," for s in ss]
    #print(ss)
    res = [] # [ss[0]]
    for i in range(len(ss)):
      if len(ss[i])>self.max_len:
        res += self.breakByWordsEqual(ss[i])
        continue
      if res and len(res[-1]) + 2 + len(ss[i]) < self.max_len:
        res[-1] = res[-1]+ ss[i]
      else:
        res += [ss[i]]
    return res

  def breakByPeriods(self, text):
    
    ss = re.split(r'\.[ \n]', text)
    #ss = ss[:-1]
    ss = [s+ r". " for s in ss if len(s)>3]
    #print(text)
    res = [] #ss[0]]
    for i in range(len(ss)):
      if len(ss[i])>self.max_len:
        res += self.breakByCommas(ss[i])
        #res += [ss[i]]
        continue
      if res and len(res[-1]) + 2 + len(ss[i]) < self.max_len:
        res[-1] = res[-1] + ss[i] #+ r". "
      else:
        res += [ss[i]]
    return res

  def breakByParagraphs(self, text):
    #ss = re.split(r'(?:\.? +)?\n+', text) 
    
    ss = re.split(r'(?: +)?\n{2,}', text) 
    #ss = [s+ "\n" for s in ss]
    #print(ss)
    res = [] #ss[0]]
    for i in range(len(ss)):
      if ss and len(ss[i])>self.max_len:
        res += self.breakByPeriods(ss[i])
        #res += [ss[i]]
        #continue
      # if res and len(res[-1]) + 3 + len(ss[i]) < max_len:
      #   res[-1] = res[-1] + r"  \n" + ss[i]
      else:
        res += [ss[i]]
      res +=  ["  \n\n"]
    return [r for r in res if len(r)>3]


  def split_text_into_chunks(self, text):
      ## temporary hack for citation shit
      text = re.sub(" ([\.,])", r"\g<1>", text)

      # deal with i.e an e.g. that break sentence separation 
      #text = text.replace("i.e.,", r"that is")
      #text = text.replace("e.g.,", r"for example")
      text = text.replace("cf.",   r"confer")
      
      #print(text)

      self.chunks =  self.breakByParagraphs(text)

  def get_test_batch(self, chunks = 20, start = 0):
    return self.chunks[start:start+chunks]

  def save_chunks_as_text(self, filename, chunks):
    with open(filename, 'w+') as f: 
      f.write("\n|".join(chunks))



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
    self.tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", 
                                            savedir="checkpoints/tacotron2",
                                            overrides={"max_decoder_steps": 2000})
    self.hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", 
                                         savedir="checkpoints/hifigan")


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


  def seq_len(self, item):
    return self.tacotron2.text_to_seq(item)[1] 


  def add_pauses(self, ordered, mel_lengths, pause_dur = 40):
    # add pause between paragraphs
    mml = torch.ones_like(mel_lengths) * max(mel_lengths)
    
    pause = torch.tensor([ pause_dur if "\n\n" in c else 0 for c in ordered])

    mel_lengths += pause
    mel_lengths =  torch.min(mel_lengths, mml)
    return mel_lengths
    #[min(mml,ml + p) for ml,p in zip(mel_lengths, pause)] 

  def text_to_speech(self, chunks):      

      #  must  sort chunks by length descending

      order, unorder = self.orderUnorder(chunks, self.seq_len)

      ordered = [chunks[i] for i in order]

      #print(ordered)
      #return ordered, ordered, ordered
      print("run tacotron")
      mel_outputs, mel_lengths, alignments = self.tacotron2.encode_batch(ordered)
           
      if self.tacotron2.hparams.max_decoder_steps in mel_lengths:
        Warning("We probably have truncated chunks")

      print("run vocoder")
      hop_len = 256
      waveforms = self.hifi_gan.decode_batch(mel_outputs, mel_lengths, hop_len).squeeze(1)

      print("adding pauses")
      mel_lengths = self.add_pauses(ordered, mel_lengths, pause_dur = 40)

      print("recombine")
      # turn into array
      arr = torch.tensor_split(waveforms, len(order), dim=0)

      #cut padding
      arr = [a[:,:l]  for a,l in zip(arr, mel_lengths * hop_len) ]

      # restore order 
      arr = [arr[i] for i in unorder]

      mel_lengths = mel_lengths.detach().numpy()
      mel_lengths = [mel_lengths[i] for i in unorder]

      return arr, mel_lengths, hop_len


  def save_audio(self, output_wav, waveform):
    torchaudio.save(output_wav, waveform, 22050, format="wav")
    print(f"Audio saved to {output_wav}")



def main():

  input_file = "data/arXiv-2106.04624v1/main.tex"
  output_file = "output/" + dt.now().strftime(r"%y.%m.%d-%H")

  parser = LatexParser()
  content = parser.read_latex(input_file)
  processed = parser.custom_latex_to_text(content)
  parser.save_text(processed, "dbg/spec_my.txt")

  tables = parser.get_tables()
  parser.save_text(tables, "dbg/tables.tex")


  chunker = Chunker(max_len=300)
  chunker.split_text_into_chunks(processed)
  chunks = chunker.get_test_batch(50,0)
  #chunks = chunker.chunks
  chunker.save_chunks_as_text(output_file + ".md", chunks)
  print("text chunks:", [len(ch) for ch in chunks])
  
  narrator = Narrator()
  waveforms, mel_lengths, hop_len = narrator.text_to_speech(chunks)  

  
  print ("mel_lengths: ", mel_lengths, hop_len)

  waveform = torch.cat(waveforms, dim=1)

  print("saving audio")
  narrator.save_audio(output_file + ".wav", waveform)


if __name__ == "__main__":
  main()