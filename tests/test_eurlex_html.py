import pytest

from ai_act_qa.corpus import build_chunks
from ai_act_qa.eurlex_html import html_to_text

HTML = """
<html><body>
<p class="title-doc-first">REGULATION OF THE EUROPEAN PARLIAMENT</p>
<div class="eli-subdivision" id="enc_1">
  <div id="cpt_I">
    <p class="title-division-1">CHAPTER I</p>
    <p class="title-division-2"><span class="boldface">GENERAL PROVISIONS</span></p>
    <p class="modref"><a href="x">&#9660;M1</a><a class="anchorarrow" href="y">v</a></p>
    <div class="eli-subdivision" id="art_1">
      <p class="title-article-norm">Article&nbsp;1</p>
      <div class="eli-title"><p class="stitle-article-norm">Subject matter'</p></div>
      <div class="norm"><span class="no-parag">1.&nbsp;&nbsp;</span>
        <div class="norm inline-element">The purpose is to improve the market<a href="#E1"><span class="superscript">1</span></a> (<a href="#E1"><span class="superscript">2</span></a>).</div>
      </div>
      <div class="grid-container grid-list">
        <div class="list grid-list-column-1"><span>(a)&nbsp;</span></div>
        <div class="grid-list-column-2"><p class="norm">first item;</p></div>
      </div>
      <p class="footnote">(1) Some footnote text.</p>
    </div>
    <div class="eli-subdivision" id="art_4a">
      <p class="title-article-norm">Article&nbsp;4a</p>
      <div class="eli-title"><p class="stitle-article-norm">Bias detection</p></div>
      <div class="norm">Providers may process data.</div>
    </div>
  </div>
</div>
<div id="anx_I">
  <p class="title-annex-1">ANNEX I</p>
  <p class="title-annex-2">List of legislation</p>
  <table><tr><td><p class="tbl-norm">Code</p></td><td><p class="tbl-norm">Meaning</p></td></tr></table>
</div>
</body></html>
"""


def test_html_to_text_removes_markers_footnotes_and_joins_list_items():
    text = html_to_text(HTML)
    assert "\u25bc" not in text and "Some footnote" not in text
    assert "1. The purpose is to improve the market." in text
    assert "(a) first item;" in text
    assert "Code | Meaning" in text
    assert "\u00a0" not in text


def test_html_to_text_feeds_the_parser():
    chunks = {c.ref: c for c in build_chunks(html_to_text(HTML))}
    assert list(chunks) == ["Article 1", "Article 4a", "Annex I"]
    assert chunks["Article 1"].title == "Subject matter"
    assert chunks["Article 1"].chapter == "Chapter I - GENERAL PROVISIONS"
    assert chunks["Annex I"].title == "List of legislation"


def test_unrecognised_page_raises():
    with pytest.raises(ValueError):
        html_to_text("<html><body><p>niente</p></body></html>")