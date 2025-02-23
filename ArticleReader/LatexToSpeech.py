
from pylatexenc import latexwalker, latex2text, macrospec
import re
from ArticleReader.NodesVisitor import Extractor

class LatexParser:
    #
    # Define macros, environments, specials for the *parser*
    #
    def __init__(self):
        self.ignore = [
            ("PassOptionsToPackage", "{{"),
            ("bibliographystyle", "{"),
            ("newfloatcommand", "{{[["),
            ("setlist", "[{"),
            ("newcolumntype", "{[{"),
            ("floatstyle", "{"),
            ("restylefloat", "{"),
            ("affil", "[{"),
            ("author", "[{"),
            ("lstdefinestyle", "{{"),
            ("lstset", "{"),
            ("lstdefinelanguage", "{{"),
            ("footnote", "{"),
            ("cite", "{"),  # '<cit.>'
            ("citet", "{"),
            ("citep", "{"),
            # ("def","[{")
        ]

        self.lw_context_db = latexwalker.get_default_latex_context_db()
        self.lw_context_db.add_context_category(
            "my-quotes",
            prepend=True,
            macros=[macrospec.MacroSpec(nam, pars) for (nam, pars) in self.ignore]
            + [macrospec.MacroSpec("CC", "{")],
            environments=[
                macrospec.EnvironmentSpec("table", "["),
                macrospec.EnvironmentSpec("figure", "["),
            ],
        )

        self.l2t_context_db = latex2text.get_default_latex_context_db()
        self.l2t_context_db.add_context_category(
            "my-quotes",
            prepend=True,
            macros=[
                latex2text.MacroTextSpec(nam, simplify_repl=r"")
                for (nam, pars) in self.ignore
            ]
            + [
                latex2text.MacroTextSpec("url", "%s"),
                latex2text.MacroTextSpec("maketitle", self.maketitle),
                latex2text.MacroTextSpec("ref", simplify_repl=self.fmt_ref_macro_node),
                latex2text.MacroTextSpec("section", self.fmt_subsections),
                latex2text.MacroTextSpec("subsection", self.fmt_subsections),
                latex2text.MacroTextSpec("subsubsection", self.fmt_subsections),
                latex2text.MacroTextSpec("CC", "si plus plus"),
                # ,latex2text.MacroTextSpec('CC', 'C++')
            ],
            environments=[
                latex2text.EnvironmentTextSpec(
                    "table",
                    simplify_repl=self.fmt_table_environment_node,
                    discard=False,
                ),
                latex2text.EnvironmentTextSpec(
                    "figure",
                    simplify_repl=self.fmt_figure_environment_node,
                    discard=False,
                ),
                # TODO: math mode
                # latex2text.EnvironmentTextSpec("tabular", simplify_repl=self.fmt_table_environment_node)
            ],
        )
        self.tables = []
        self.figures = []

    def maketitle(self, node, l2tobj):
        return getattr(l2tobj, "_doc_title", r"[NO \title GIVEN]") + ".  "

    def read_latex(self, filepath, size=-1):
        with open(filepath) as f:
            content = f.read(size)
        return content

    def fmt_table_environment_node(self, node, l2tobj):

        # if node.environmentname == 'table':
        nv = Extractor()
        res = nv.start(node)
        if res:
            table_name = res["visited_results_body"][0]["label"]
            # table_name = table_name.replace(":","_")
            # print(res)
            found = {"label": f"{table_name}", "content": node.latex_verbatim()}
            self.tables.append(found)

            # content  =  #l2tobj.nodelist_to_text(node.nodelist)
            # self.save_text(content, dest)
            return ""  # "LatexTable: " + table_name
        return ""  # "LatexTable: ERROR"

    def fmt_figure_environment_node(self, node, l2tobj):

        # if node.environmentname == 'table':
        nv = Extractor()
        res = nv.start(node)
        if res:
            figure_name = res["visited_results_body"][0]["label"]
            # table_name = table_name.replace(":","_")
            # print(res)
            found = {"label": f"{figure_name}", "content": node.latex_verbatim()}
            self.figures.append(found)

            # content  =  #l2tobj.nodelist_to_text(node.nodelist)
            # self.save_text(content, dest)
            return ""  # "LatexFigure: " + figure_name
        return ""  # "LatexFigure: ERROR"

    def fmt_ref_macro_node(self, node, l2tobj):
        res = node.nodeargs[0].nodelist.nodelist[0].chars

        if res:
            # self.save_text(content, dest)
            return "one"  # "LatexRef: " + res
        return "oops"  # "LatexRef: ERROR"

    def fmt_subsections(self, node, l2tobj):
        res = l2tobj.node_arg_to_text(node, 2)
        if res:
            return node.macroname.capitalize() + ": " + res + ".\n"
        return ""  # "LatexSubsection: ERROR"

    def get_tables(self):
        # TODO: convert to images:
        # https://tex.stackexchange.com/questions/11866/compile-a-latex-document-into-a-png-image-thats-as-short-as-possible
        tables = [f" % {t['label']} \n {t['content']} " for t in self.tables]
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
        \\putinquotes[open-quote][close-quote]{text}"""
        if not n.nodeargd:
            # n.nodeargd can be empty if e.g. \putinquotes was a single
            # token passed as an argument to a macro,
            # e.g. \newcommand\putinquotes...
            return ""
        open_q_s = self._get_optional_arg(n.nodeargd.argnlist[0], "“", l2tobj)
        close_q_s = self._get_optional_arg(n.nodeargd.argnlist[1], "”", l2tobj)
        return open_q_s + l2tobj.nodelist_to_text([n.nodeargd.argnlist[2]]) + close_q_s

    def in_quotes_env_repl(self, n, l2tobj):
        """Get the text replacement for the {inquotes} environment"""
        open_q_s = self._get_optional_arg(n.nodeargd.argnlist[0], "“", l2tobj)
        close_q_s = self._get_optional_arg(n.nodeargd.argnlist[1], "”", l2tobj)
        return open_q_s + l2tobj.nodelist_to_text(n.nodelist) + close_q_s

    def do_author(n, l2tobj):
        """Get the text replacement for the macro
        \\putinquotes[open-quote][close-quote]{text}"""
        if not n.nodeargd:
            # n.nodeargd can be empty if e.g. \putinquotes was a single
            # token passed as an argument to a macro,
            # e.g. \newcommand\putinquotes...
            return ""
        # open_q_s = _get_optional_arg(n.nodeargd.argnlist[0], '“', l2tobj)

        return ""

    # l2t_context_db.set_unknown_macro_spec()

    def custom_latex_to_text(self, input_latex):
        # the latex parser instance with custom latex_context
        lw_obj = latexwalker.LatexWalker(input_latex, latex_context=self.lw_context_db)
        # parse to node list
        nodelist, pos, length = lw_obj.get_latex_nodes()
        # initialize the converter to text with custom latex_context
        l2t_obj = latex2text.LatexNodes2Text(latex_context=self.l2t_context_db)
        # convert to text
        processed = l2t_obj.nodelist_to_text(nodelist)
        return self.postprocess(processed)

    def postprocess(self, processed):
        # Temporary hacks:
        # manually remove all that latex parser didn't catch
        processed = processed.replace("C⁠[4].4ex++", "").strip()
        # processed = processed.replace("si plus plus","").strip()

        # shorten long spaces to just one
        processed = re.sub(r" {2,}", " ", processed)
        return processed

    def save_text(self, text, filename):
        with open(filename, "w+") as f:
            f.write(text)


