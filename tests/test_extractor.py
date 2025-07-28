import os
import pytest
from outline_extractor.extractor import extract_outline
from outline_extractor.utils import validate_outline_schema

def test_extract_outline():
    sample_pdf = os.path.join(os.path.dirname(__file__), 'sample.pdf')
    if not os.path.exists(sample_pdf):
        pytest.skip('Sample PDF not found')
    outline = extract_outline(sample_pdf)
    assert validate_outline_schema(outline)
    assert 'title' in outline
    assert isinstance(outline['headings'], list)

