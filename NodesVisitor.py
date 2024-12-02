from pylatexenc.latexnodes.nodes import LatexNodesVisitor

class Extractor(LatexNodesVisitor):
  def visit_macro_node(self, node, **kwargs):
    if node.macroname == 'label':
      label = node.nodeargd.argnlist[0].nodelist[0].chars
#node.nodelist.nodelist[1].nodelist.nodelist[3].nodelist.nodelist[4].nodeargd.argnlist[0].nodelist.nodelist[0].chars
      return {'label': label}

  def visit_environment_node(self, node, **kwargs):
    res = cleanup(kwargs)
    return res

# 
# node.nodelist[8].nodeargd.argnlist[0].nodelist[0].chars
def cleanup(res):
  if res is None:
    return
  if type(res) == dict:
    tmp = { k:cleanup(v) for k,v in res.items() if cleanup(v) is not None}
  else:
    tmp = [i for i in res if i != None]
  return tmp if tmp else None