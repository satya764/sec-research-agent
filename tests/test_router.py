from sec_research_agent.models import ParsedQuestion
from sec_research_agent.router import route


def test_route():
    cheap, strong = "cheap", "strong"
    assert route(ParsedQuestion(companies=["AAPL"], question_type="lookup"), cheap, strong)[0] == cheap
    assert route(ParsedQuestion(companies=["AAPL"], question_type="narrative"), cheap, strong)[0] == cheap
    assert route(ParsedQuestion(companies=["AAPL", "MSFT"], question_type="lookup"), cheap, strong)[0] == strong
    assert route(ParsedQuestion(companies=["AAPL"], fiscal_years=[2023, 2024]), cheap, strong)[0] == strong
    assert route(ParsedQuestion(companies=["AAPL"], question_type="calculation"), cheap, strong)[0] == strong
