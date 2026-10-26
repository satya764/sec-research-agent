from sec_research_agent.chunking import chunk_text


def test_chunks_respect_max_size():
    text = "\n".join(f"Paragraph {i}. " + "word " * 80 for i in range(40))
    chunks = chunk_text(text, max_chars=800, overlap=100)
    assert len(chunks) > 5
    assert all(len(c) <= 800 for c in chunks)


def test_long_paragraph_is_split_by_sentence_then_hard_cut():
    text = "Short sentence. " * 200 + "x" * 3000
    chunks = chunk_text(text, max_chars=500, overlap=50)
    assert all(len(c) <= 500 for c in chunks)
    assert "".join(chunks).count("x") >= 3000


def test_overlap_carries_context():
    text = "\n".join(f"Sentence number {i} about supply chain risk." for i in range(100))
    chunks = chunk_text(text, max_chars=400, overlap=120)
    last_words = chunks[0].split()[-3:]
    assert " ".join(last_words) in chunks[1]


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.") == ["Hello world."]
