from sec_research_agent.filings import html_to_text, split_sections

BODY = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 30

HTML = f"""
<html><head><title>10-K</title></head><body>
<div style="display:none">hidden xbrl header</div>
<table>
  <tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
  <tr><td>Item 1A.</td><td>Risk Factors</td><td>9</td></tr>
  <tr><td>Item 7.</td><td>Management's Discussion and Analysis</td><td>30</td></tr>
</table>
<p><span>Item 1.</span><span> Business</span></p>
<p>Company Background. {BODY}</p>
<p><b>Item 1A. Risk Factors</b></p>
<p>Supply chain risk. {BODY} See Part I, Item 7 of this Form 10-K for more on this topic and related matters.</p>
<p>Item 1B. Unresolved Staff Comments</p><p>None.</p>
<p>Item 7. Management's Discussion and Analysis of Financial Condition</p>
<p>Fiscal year highlights. {BODY}</p>
<table><tr><td>Revenue</td><td>$</td><td>391,035</td><td>$</td><td>383,285</td></tr></table>
<p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p><p>Interest rates.</p>
</body></html>
"""


def test_html_to_text_drops_hidden_and_numeric_tables():
    text = html_to_text(HTML)
    assert "hidden xbrl header" not in text
    assert "391,035" not in text
    assert "Item 1. Business" in text


def test_split_sections_skips_table_of_contents():
    sections = split_sections(html_to_text(HTML), "10-K")
    assert set(sections) == {"business", "risk_factors", "mdna"}
    assert sections["business"].startswith("Item 1. Business")
    assert "Company Background" in sections["business"]
    assert "Supply chain risk" in sections["risk_factors"]
    assert "Unresolved Staff" not in sections["risk_factors"]
    assert "Fiscal year highlights" in sections["mdna"]
    assert "Interest rates" not in sections["mdna"]


def test_unknown_form_returns_nothing():
    assert split_sections(html_to_text(HTML), "8-K") == {}
