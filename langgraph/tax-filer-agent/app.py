#!/usr/bin/env python3
import base64
import io
import json
import os
import time
import traceback
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from llama_cloud import LlamaCloud
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import FreeText
from pypdf.generic import ArrayObject, NameObject
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&family=Source+Sans+3:wght@300;400;500;600;700&display=swap');

:root {
    --brand-navy:       #1E3A8A;
    --brand-blue:       #2563EB;
    --brand-blue-lt:    #3B82F6;
    --brand-orange:     #F97316;
    --brand-orange-dk:  #EA6A0A;

    --n-0:   #FFFFFF;
    --n-50:  #F8FAFC;
    --n-100: #F1F5F9;
    --n-200: #E2E8F0;
    --n-300: #CBD5E1;
    --n-400: #94A3B8;
    --n-500: #64748B;
    --n-600: #475569;
    --n-700: #334155;
    --n-800: #1E293B;
    --n-900: #0F172A;

    --ok:    #059669;
    --ok-bg: #ECFDF5;
    --ok-br: #A7F3D0;
    --warn:  #D97706;
    --warn-bg:#FFFBEB;
    --warn-br:#FDE68A;
    --err:   #DC2626;
    --err-bg:#FEF2F2;
    --err-br:#FECACA;

    --font-h: 'Lexend', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-b: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif;

    --r-sm:  8px;
    --r-md:  12px;
    --r-lg:  16px;
    --r-xl:  20px;
    --r-2xl: 24px;
    --r-full:9999px;

    --s1: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    --s2: 0 4px 12px rgba(0,0,0,.06), 0 2px 4px rgba(0,0,0,.04);
    --s3: 0 8px 24px rgba(0,0,0,.07), 0 4px 8px rgba(0,0,0,.04);
    --s-blue:   0 4px 14px rgba(37,99,235,.22);
    --s-orange: 0 4px 14px rgba(249,115,22,.30);
}

#MainMenu, footer, header { visibility: hidden; }

.stApp { background: var(--n-50); font-family: var(--font-b); }

.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 4rem !important;
    max-width: 1100px;
}

section[data-testid="stSidebar"] {
    background: var(--n-0) !important;
    border-right: 1px solid var(--n-200) !important;
}
section[data-testid="stSidebar"] .block-container { padding-top: 24px; }

.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 24px 0 28px;
    border-bottom: 1px solid var(--n-200);
    margin-bottom: 56px;
}
.app-logo {
    display: flex;
    align-items: center;
    gap: 12px;
}
.app-logo-mark {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--brand-navy) 0%, var(--brand-blue) 100%);
    border-radius: var(--r-md);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.app-logo-text {
    font-family: var(--font-h);
    font-size: 1.1875rem;
    font-weight: 600;
    color: var(--n-900);
    letter-spacing: -0.02em;
}
.app-status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.model-tag {
    font-family: var(--font-b);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--n-400);
    letter-spacing: 0.04em;
    padding: 4px 10px;
    background: var(--n-100);
    border-radius: var(--r-full);
    margin-left: 4px;
}

.hero {
    text-align: center;
    padding: 0 0 60px;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border: 1px solid #BFDBFE;
    border-radius: var(--r-full);
    padding: 5px 16px;
    font-family: var(--font-h);
    font-size: 0.6875rem;
    font-weight: 600;
    color: var(--brand-blue);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 22px;
}
.hero-headline {
    font-family: var(--font-h);
    font-size: 3.25rem;
    font-weight: 700;
    color: var(--n-900);
    letter-spacing: -0.035em;
    line-height: 1.08;
    margin: 0 0 20px;
}
.hero-headline em {
    font-style: normal;
    background: linear-gradient(125deg, var(--brand-blue) 0%, var(--brand-navy) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: var(--font-b);
    font-size: 1.125rem;
    color: var(--n-500);
    line-height: 1.65;
    max-width: 520px;
    margin: 0 auto;
}

.section-divider {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 0 0 32px;
}
.section-divider-line { flex: 1; height: 1px; background: var(--n-200); }
.section-label {
    font-family: var(--font-h);
    font-size: 0.6875rem;
    font-weight: 700;
    color: var(--n-400);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.upload-card {
    background: var(--n-0);
    border: 1.5px solid var(--n-200);
    border-radius: var(--r-xl);
    padding: 28px 24px 22px;
    box-shadow: var(--s1);
    transition: border-color .2s ease, box-shadow .2s ease;
}
.upload-card:hover {
    border-color: #93C5FD;
    box-shadow: var(--s2);
}

.step-label {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.step-num {
    width: 26px; height: 26px;
    border-radius: 50%;
    background: var(--brand-blue);
    color: #fff;
    font-family: var(--font-h);
    font-size: 0.75rem;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.step-name {
    font-family: var(--font-h);
    font-size: 0.8125rem;
    font-weight: 700;
    color: var(--n-700);
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.card-desc {
    font-family: var(--font-b);
    font-size: 0.9375rem;
    color: var(--n-500);
    line-height: 1.55;
    margin: 6px 0 18px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: var(--r-full);
    font-family: var(--font-b);
    font-size: 0.8125rem;
    font-weight: 600;
    white-space: nowrap;
}
.badge-ok   { background: var(--ok-bg);   border: 1px solid var(--ok-br);   color: var(--ok);   }
.badge-err  { background: var(--err-bg);  border: 1px solid var(--err-br);  color: var(--err);  }
.badge-warn { background: var(--warn-bg); border: 1px solid var(--warn-br); color: var(--warn); }
.badge-dot  { width: 7px; height: 7px; border-radius: 50%; background: currentColor; flex-shrink: 0; }

.file-chip {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 14px 16px;
    border-radius: var(--r-md);
    margin-top: 14px;
}
.file-chip-ok   { background: var(--ok-bg);   border: 1px solid var(--ok-br);   }
.file-chip-warn { background: var(--warn-bg); border: 1px solid var(--warn-br); }
.file-chip-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }
.file-chip-name {
    font-family: var(--font-h);
    font-size: 0.9375rem;
    font-weight: 600;
    color: var(--n-800);
    margin-bottom: 2px;
    word-break: break-all;
}
.file-chip-meta { font-family: var(--font-b); font-size: 0.8125rem; color: var(--n-500); }
.file-list-item {
    font-family: var(--font-b);
    font-size: 0.8125rem;
    color: var(--n-600);
    padding: 3px 0 3px 14px;
}

.waiting-card {
    background: #EFF6FF;
    border: 1.5px dashed #93C5FD;
    border-radius: var(--r-xl);
    padding: 52px 32px;
    text-align: center;
    margin: 28px 0;
}
.waiting-title {
    font-family: var(--font-h);
    font-size: 1.125rem;
    font-weight: 600;
    color: var(--brand-blue);
    margin-bottom: 8px;
}
.waiting-sub { font-family: var(--font-b); font-size: 0.9375rem; color: var(--n-400); }

.stButton > button {
    font-family: var(--font-h) !important;
    font-weight: 600 !important;
    border-radius: var(--r-md) !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
    min-height: 54px !important;
    font-size: 1rem !important;
    letter-spacing: -0.01em !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(160deg, var(--brand-orange) 0%, var(--brand-orange-dk) 100%) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: var(--s-orange) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(249,115,22,.40) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

.config-err {
    background: var(--err-bg);
    border: 1.5px solid var(--err-br);
    border-radius: var(--r-lg);
    padding: 24px 28px;
    margin: 32px 0;
}
.config-err-title {
    font-family: var(--font-h);
    font-weight: 700;
    color: var(--err);
    font-size: 1rem;
    margin-bottom: 6px;
}
.config-err-body { font-family: var(--font-b); font-size: 0.9375rem; color: var(--n-700); }

.results-banner {
    background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 60%, #FFFFFF 100%);
    border: 1.5px solid var(--ok-br);
    border-radius: var(--r-xl);
    padding: 36px 32px;
    text-align: center;
    margin-bottom: 36px;
}
.results-check {
    width: 60px; height: 60px;
    background: var(--ok);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 18px;
    box-shadow: 0 4px 16px rgba(5,150,105,.25);
}
.results-headline {
    font-family: var(--font-h);
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--n-900);
    letter-spacing: -0.025em;
    margin-bottom: 6px;
}
.results-sub { font-family: var(--font-b); font-size: 0.9375rem; color: var(--n-500); }

.metric-card {
    background: var(--n-0);
    border: 1.5px solid var(--n-200);
    border-radius: var(--r-lg);
    padding: 24px 20px;
    text-align: center;
    box-shadow: var(--s1);
}
.metric-value {
    font-family: var(--font-h);
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--brand-navy);
    line-height: 1;
    letter-spacing: -0.04em;
}
.metric-label {
    font-family: var(--font-b);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--n-400);
    margin-top: 8px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.stDownloadButton > button {
    font-family: var(--font-h) !important;
    font-weight: 700 !important;
    border-radius: var(--r-md) !important;
    min-height: 58px !important;
    font-size: 1.0625rem !important;
    letter-spacing: -0.01em !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
}
.stDownloadButton > button[kind="primary"] {
    background: linear-gradient(160deg, var(--brand-blue) 0%, var(--brand-navy) 100%) !important;
    border: none !important;
    box-shadow: var(--s-blue) !important;
}
.stDownloadButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(37,99,235,.35) !important;
}

[data-testid="stFileUploader"] {
    border: 2px dashed var(--n-300) !important;
    border-radius: var(--r-lg) !important;
    background: var(--n-50) !important;
    transition: border-color .18s ease, background .18s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #93C5FD !important;
    background: #EFF6FF !important;
}

.streamlit-expanderHeader {
    font-family: var(--font-h) !important;
    font-weight: 600;
    color: var(--n-700);
}

.sidebar-section-title {
    font-family: var(--font-h);
    font-size: 0.6875rem;
    font-weight: 700;
    color: var(--n-400);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.sidebar-file {
    font-family: monospace;
    font-size: 0.8125rem;
    color: var(--n-600);
    padding: 3px 0;
}

.unmapped-warn {
    margin-top: 16px;
    padding: 14px 16px;
    background: var(--warn-bg);
    border: 1px solid var(--warn-br);
    border-radius: var(--r-md);
    font-size: 0.875rem;
}
.unmapped-warn strong { color: var(--warn); }
.unmapped-warn div { color: var(--n-600); margin-top: 4px; }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}

[data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] li {
    color: var(--n-800) !important;
}
</style>
"""
load_dotenv()
LLAMA_CLOUD_API_KEY: str = os.getenv('LLAMA_CLOUD_API_KEY', '')
ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')
MODEL_NAME: str = 'claude-sonnet-4-5'
WORKSPACE = Path('workspace')
WORKSPACE.mkdir(exist_ok=True)

def workspace_file(name: str) -> Path:
    """Return a workspace path, recreating the directory for tool-thread writes."""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    return WORKSPACE / name

def run_ocr(pdf_bytes: bytes, filename: str='document.pdf') -> str:
    client = LlamaCloud(api_key=LLAMA_CLOUD_API_KEY)
    file = client.files.create(file=(filename, pdf_bytes, 'application/pdf'), purpose='parse')
    result = client.parsing.parse(file_id=file.id, tier='agentic', version='latest', expand=['markdown'])
    if not result.markdown or not result.markdown.pages:
        raise RuntimeError(f'LlamaCloud Parse returned no pages for {filename}.')
    markdown = '\n\n---\n\n'.join((page.markdown for page in result.markdown.pages))
    if not markdown.strip():
        raise RuntimeError(f'LlamaCloud Parse extracted empty text from {filename}.')
    return markdown

def run_ocr_multiple(files: list[tuple[bytes, str]]) -> str:
    all_markdown = []
    for pdf_bytes, filename in files:
        markdown = run_ocr(pdf_bytes, filename)
        all_markdown.append(f'# Source: {filename}\n\n{markdown}')
    combined = '\n\n' + '=' * 80 + '\n\n'.join(all_markdown)
    workspace_file('ocr_output.md').write_text(combined, encoding='utf-8')
    return combined

def extract_all_fields(markdown: str) -> dict:
    model = ChatAnthropic(model=MODEL_NAME, api_key=ANTHROPIC_API_KEY, temperature=0)
    prompt = 'Analyze this document text and extract ALL relevant data fields.\n\nYour task:\n1. Identify what type of document(s) this is (tax form, invoice, application, etc.)\n2. Extract EVERY piece of meaningful data you find\n3. Use clear, descriptive field names (e.g., "employer_name", "total_wages", "social_security_number")\n4. Preserve exact values - don\'t modify or calculate anything\n\nRules:\n- Extract ALL data, not just common fields\n- Use snake_case for field names\n- For monetary values, extract as numbers (no $ or commas)\n- For dates, use ISO format (YYYY-MM-DD) when possible\n- Include document metadata (tax_year, form_type, etc.) if present\n- If a field appears multiple times (e.g., multiple employers), use numbered keys\n\nReturn ONLY valid JSON in this format:\n{\n  "document_type": "detected type",\n  "field_name_1": "value1",\n  "field_name_2": 12345.67,\n  ...\n}\n\nDocument text:\n' + markdown
    response = model.invoke([HumanMessage(content=prompt)])
    content = response.content
    if '```json' in content:
        content = content.split('```json')[1].split('```')[0]
    elif '```' in content:
        content = content.split('```')[1].split('```')[0]
    fields = json.loads(content.strip())
    workspace_file('extracted_fields.json').write_text(json.dumps(fields, indent=2), encoding='utf-8')
    return fields

def get_acroform_fields(pdf_bytes: bytes) -> dict[str, str]:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        fields = reader.get_fields() or {}
        return {name: field.get('/V', '') or '' for name, field in fields.items()}
    except Exception:
        return {}

def get_field_annotation_id(annotation) -> str | None:
    components = []
    while annotation:
        field_name = annotation.get('/T')
        if field_name:
            components.append(str(field_name))
        annotation = annotation.get('/Parent')
    return '.'.join(reversed(components)) if components else None

def extract_form_field_info(pdf_bytes: bytes) -> list[dict]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    fields = reader.get_fields() or {}
    field_info_by_id = {}
    possible_radio_names = set()
    for field_id, field in fields.items():
        ft = field.get('/FT')
        if field.get('/Kids'):
            if ft == '/Btn':
                possible_radio_names.add(field_id)
            continue
        field_dict = {'field_id': field_id}
        if ft == '/Tx':
            field_dict['type'] = 'text'
        elif ft == '/Btn':
            field_dict['type'] = 'checkbox'
            states = field.get('/_States_', [])
            if len(states) == 2:
                if '/Off' in states:
                    field_dict['checked_value'] = states[0] if states[0] != '/Off' else states[1]
                    field_dict['unchecked_value'] = '/Off'
                else:
                    field_dict['checked_value'] = states[0]
                    field_dict['unchecked_value'] = states[1]
        elif ft == '/Ch':
            field_dict['type'] = 'choice'
            opts = field.get('/Opt', [])
            field_dict['choice_options'] = [str(opt) for opt in opts] if opts else []
        else:
            field_dict['type'] = 'unknown'
        field_info_by_id[field_id] = field_dict
    radio_fields_by_id = {}
    for page_idx, page in enumerate(reader.pages):
        annotations = page.get('/Annots', [])
        if not annotations:
            continue
        for ann in annotations:
            try:
                ann_obj = ann.get_object() if hasattr(ann, 'get_object') else ann
            except Exception:
                continue
            field_id = get_field_annotation_id(ann_obj)
            rect = ann_obj.get('/Rect')
            if field_id in field_info_by_id:
                field_info_by_id[field_id]['page'] = page_idx + 1
                if rect:
                    field_info_by_id[field_id]['rect'] = [float(r) for r in rect]
            elif field_id in possible_radio_names:
                try:
                    ap = ann_obj.get('/AP', {})
                    n = ap.get('/N', {})
                    on_values = [v for v in n.keys() if v != '/Off']
                except Exception:
                    continue
                if on_values:
                    if field_id not in radio_fields_by_id:
                        radio_fields_by_id[field_id] = {'field_id': field_id, 'type': 'radio_group', 'page': page_idx + 1, 'radio_options': []}
                    radio_fields_by_id[field_id]['radio_options'].append({'value': on_values[0], 'rect': [float(r) for r in rect] if rect else None})
    all_fields = []
    for field_info in field_info_by_id.values():
        if 'page' in field_info:
            all_fields.append(field_info)
    all_fields.extend(radio_fields_by_id.values())

    def sort_key(f):
        page = f.get('page', 0)
        rect = f.get('rect') or (f.get('radio_options', [{}])[0].get('rect') if f.get('radio_options') else None)
        if rect:
            return (page, -rect[1], rect[0])
        return (page, 0, 0)
    all_fields.sort(key=sort_key)
    return all_fields

def convert_form_to_images(pdf_bytes: bytes, max_dim: int=1200) -> list[tuple[bytes, int, int]]:
    if not PDF2IMAGE_AVAILABLE:
        return []
    try:
        images = convert_from_bytes(pdf_bytes, dpi=150)
        result = []
        for img in images:
            width, height = img.size
            if width > max_dim or height > max_dim:
                scale = min(max_dim / width, max_dim / height)
                new_size = (int(width * scale), int(height * scale))
                img = img.resize(new_size)
                width, height = new_size
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            result.append((buf.getvalue(), width, height))
        return result
    except Exception:
        return []

def map_w9_fields(extracted_data: dict, form_fields: list[str]) -> dict[str, str]:
    field_lookup = {}
    for field_id in form_fields:
        simple = field_id.split('.')[-1].replace('[0]', '')
        field_lookup[simple] = field_id
    w9_indicators = ['f1_01', 'f1_02', 'f1_07', 'f1_09', 'c1_1']
    if not any((ind in field_lookup for ind in w9_indicators)):
        return {}
    mapping = {}
    name = extracted_data.get('full_legal_name') or extracted_data.get('name') or ''
    if name and 'f1_01' in field_lookup:
        mapping[field_lookup['f1_01']] = name
    business = extracted_data.get('business_name_dba') or extracted_data.get('business_name') or ''
    if business and 'f1_02' in field_lookup:
        mapping[field_lookup['f1_02']] = business
    address = extracted_data.get('street_address') or extracted_data.get('address') or ''
    if address and 'f1_07' in field_lookup:
        mapping[field_lookup['f1_07']] = address
    city = extracted_data.get('city', '')
    state = extracted_data.get('state', '')
    zip_code = extracted_data.get('zip_code', '')
    city_state_zip = f'{city}, {state} {zip_code}'.strip(', ')
    if city_state_zip and 'f1_09' in field_lookup:
        mapping[field_lookup['f1_09']] = city_state_zip
    ssn = extracted_data.get('social_security_number', '').replace('-', '').replace(' ', '')
    if len(ssn) >= 9:
        if 'f1_11' in field_lookup:
            mapping[field_lookup['f1_11']] = ssn[:3]
        if 'f1_12' in field_lookup:
            mapping[field_lookup['f1_12']] = ssn[3:5]
        if 'f1_13' in field_lookup:
            mapping[field_lookup['f1_13']] = ssn[5:9]
    tax_class = extracted_data.get('tax_classification', '').lower()
    if 'individual' in tax_class or 'sole proprietor' in tax_class:
        if 'c1_1' in field_lookup:
            mapping[field_lookup['c1_1']] = '/1'
    return mapping

GENERAL_FORM_FILLING_GUIDANCE = """General form-filling rules:
- Treat the uploaded PDF as an arbitrary form, not a W-9-only template.
- Use visual labels, field positions, field IDs, and extracted source data together.
- Fill any form field with the most specific matching source value when confidence is high.
- If a visual field is a row of separate boxes for digits or characters, map the full value to the grouped PDF field; the filler will split it per visible box.
- For taxpayer IDs, SSNs, EINs, account numbers, dates, ZIP codes, and similar boxed values, preserve exact characters and digit order.
- For checkboxes/radio fields, choose the option that matches the source classification or facts; do not write the literal backend value as text.
- Do not invent missing values. Leave low-confidence fields unmapped."""

def smart_map_fields_with_vision(extracted_data: dict, field_info: list[dict], form_images: list[tuple[bytes, int, int]]) -> dict[str, str]:
    if not field_info:
        return {}
    form_fields = [f['field_id'] for f in field_info]
    w9_mapping = map_w9_fields(extracted_data, form_fields)
    if w9_mapping:
        return w9_mapping
    model = ChatAnthropic(model=MODEL_NAME, api_key=ANTHROPIC_API_KEY, temperature=0, max_tokens=4096)
    text_fields = [f for f in field_info if f.get('type') == 'text']
    checkbox_fields = [f for f in field_info if f.get('type') == 'checkbox']
    field_summary = 'TEXT FIELDS (with positions on page):\n'
    for f in text_fields:
        rect = f.get('rect', [])
        pos = f"page {f.get('page', '?')}, position ({rect[0]:.0f}, {rect[1]:.0f})" if rect else 'unknown position'
        field_summary += f"  - {f['field_id']}: {pos}\n"
    if checkbox_fields:
        field_summary += '\nCHECKBOX FIELDS:\n'
        for f in checkbox_fields:
            rect = f.get('rect', [])
            pos = f"page {f.get('page', '?')}, position ({rect[0]:.0f}, {rect[1]:.0f})" if rect else 'unknown position'
            field_summary += f"  - {f['field_id']}: {pos}, check with '{f.get('checked_value', '/Yes')}'\n"
    prompt_text = f"""{GENERAL_FORM_FILLING_GUIDANCE}\n\nYou are a form filling expert. Analyze the form image(s) and map the extracted data to the correct form fields.\n\nEXTRACTED DATA (from source documents):\n{json.dumps(extracted_data, indent=2)}\n\nFORM FIELDS IN THE PDF:\n{field_summary}\n\nINSTRUCTIONS:\n1. Look at the form image(s) to understand the visual layout\n2. Use the field positions to identify what label/purpose each field has\n3. Match extracted data to the appropriate form fields based on:\n   - Visual proximity of fields to their labels in the image\n   - Field position coordinates (PDF coordinates: y=0 is bottom, increases upward)\n   - Common form conventions (name, address, taxpayer ID, dates, signatures, checkboxes, etc.)\n\n4. If the form is a W-9 or similar tax form:\n   - f1_01 is typically "Name" (line 1)\n   - f1_02 is typically "Business name/DBA" (line 2)\n   - f1_07 is typically "Address" (line 5)\n   - f1_08 or f1_09 may be "City, state, ZIP"\n   - Checkbox fields (c1_*) are often tax classification choices\n   - f1_11, f1_12, f1_13 are commonly grouped SSN digit boxes\n\n5. CRITICAL: For digit/character box groups:\n   - If several visible boxes represent one logical value, map the full value to the grouped PDF field\n   - Example: SSN "123-45-6789" may map as "123" to the 3-box SSN field, "45" to the 2-box field, and "6789" to the 4-box field\n   - Do not squeeze grouped digits into a single visual box; the PDF filler splits grouped values per visible box\n\n6. Only map fields where you have HIGH CONFIDENCE in the match\n7. Convert values to strings suitable for each field type\n\nReturn ONLY valid JSON mapping field IDs to values:\n{{\n  "topmostSubform[0].Page1[0].f1_01[0]": "John Smith",\n  "topmostSubform[0].Page1[0].f1_02[0]": "Smith Consulting",\n  ...\n}}\n\nReturn an empty object {{}} if you cannot confidently map any fields."""
    message_content = []
    if form_images:
        for i, (img_bytes, width, height) in enumerate(form_images):
            b64_image = base64.standard_b64encode(img_bytes).decode('utf-8')
            message_content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64_image}})
            message_content.append({'type': 'text', 'text': f'[Form page {i + 1} - {width}x{height}px]'})
    message_content.append({'type': 'text', 'text': prompt_text})
    response = model.invoke([HumanMessage(content=message_content)])
    content = response.content
    if '```json' in content:
        content = content.split('```json')[1].split('```')[0]
    elif '```' in content:
        content = content.split('```')[1].split('```')[0]
    try:
        mapping = json.loads(content.strip())
        valid_ids = {f['field_id'] for f in field_info}
        return {k: str(v) for k, v in mapping.items() if k in valid_ids}
    except json.JSONDecodeError:
        return {}

def smart_map_fields(extracted_data: dict, form_fields: list[str]) -> dict[str, str]:
    if not form_fields:
        return {}
    w9_mapping = map_w9_fields(extracted_data, form_fields)
    if w9_mapping:
        return w9_mapping
    model = ChatAnthropic(model=MODEL_NAME, api_key=ANTHROPIC_API_KEY, temperature=0)
    prompt = f"""{GENERAL_FORM_FILLING_GUIDANCE}\n\nYou are mapping extracted document data to PDF form fields. The form may be any fillable PDF, including but not limited to US tax forms.\n\nEXTRACTED DATA (from source documents):\n{json.dumps(extracted_data, indent=2)}\n\nFORM FIELDS (in the blank PDF form to fill):\n{json.dumps(form_fields, indent=2)}\n\nCOMMON FIELD PATTERNS:\n- Sequential text fields often represent name, business, address, dates, IDs, or signature areas\n- Checkbox/radio fields often represent classifications, yes/no answers, certifications, or status selections\n- Groups of adjacent fields or boxed areas may represent SSN/EIN, dates, phone numbers, ZIP codes, account numbers, or other per-character values\n- Address fields may use names like Address, street, city, state, zip, or generic IDs like f1_07/f1_09\n\nYour task:\n1. Match each form field to the most appropriate extracted data value\n2. Use exact source values and preserve digit order\n3. For W-9 forms specifically:\n   - f1_01 = Name (line 1)\n   - f1_02 = Business name (line 2)\n   - f1_07 = Street address\n   - f1_08 or f1_09 = City, State, ZIP\n   - f1_11/f1_12/f1_13 = grouped SSN boxes; map "123", "45", "6789" style chunks when appropriate\n4. Be conservative - only map fields where you're confident\n5. Convert values to strings suitable for form filling\n\nReturn ONLY valid JSON mapping form field names to values:\n{{\n  "form_field_name_1": "value to fill",\n  ...\n}}\n\nOnly include fields that have a clear match."""
    response = model.invoke([HumanMessage(content=prompt)])
    content = response.content
    if '```json' in content:
        content = content.split('```json')[1].split('```')[0]
    elif '```' in content:
        content = content.split('```')[1].split('```')[0]
    try:
        mapping = json.loads(content.strip())
        return {k: str(v) for k, v in mapping.items() if k in form_fields}
    except json.JSONDecodeError:
        return {}

def is_xfa_form(pdf_bytes: bytes) -> bool:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        root = reader.trailer.get('/Root', {})
        acro_form = root.get('/AcroForm', {})
        if hasattr(acro_form, 'get_object'):
            acro_form = acro_form.get_object()
        return bool(acro_form.get('/XFA')) if acro_form else False
    except Exception:
        return False

def _write_pdf_bytes(writer: PdfWriter) -> bytes:
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()

def _rects_overlap(first: list[float], second: list[float]) -> bool:
    return not (
        first[2] < second[0]
        or second[2] < first[0]
        or first[3] < second[1]
        or second[3] < first[1]
    )

def _annotation_display_value(field: dict, value: str) -> str:
    if field.get('type') in {'checkbox', 'radio_group'}:
        if str(value).strip().lower() in {'', '/off', 'off', 'false', 'none'}:
            return ''
        return 'X'
    return str(value)

def _is_w9_digit_group_field(field: dict, value: str) -> bool:
    """Detect W-9 SSN fields that visually contain one box per digit."""
    field_id = field.get('field_id', '')
    simple_field_id = field_id.split('.')[-1].split('[')[0]
    digits = str(value).replace('-', '').replace(' ', '')
    return (
        simple_field_id in {'f1_11', 'f1_12', 'f1_13'}
        and digits.isdigit()
        and len(digits) > 1
    )

def _add_digit_group_annotations(
    writer: PdfWriter,
    page_num: int,
    rect: list[float],
    digits: str,
    font_size: int,
) -> None:
    left, bottom, right, top = rect
    box_width = (right - left) / len(digits)
    for index, digit in enumerate(digits):
        digit_left = left + (box_width * index)
        digit_right = digit_left + box_width
        annotation = FreeText(
            text=digit,
            rect=(digit_left, bottom, digit_right, top),
            font='Helvetica',
            font_size=f'{font_size}pt',
            font_color='000000',
            border_color=None,
            background_color=None,
        )
        writer.add_annotation(page_number=page_num - 1, annotation=annotation)

def _remove_previous_generated_annotations(writer: PdfWriter, field_info: list[dict]) -> None:
    """Remove generated FreeText inside target fields before refilling."""
    target_rects_by_page: dict[int, list[list[float]]] = {}
    for field in field_info:
        rect = field.get('rect')
        page_num = field.get('page', 1)
        if rect:
            target_rects_by_page.setdefault(page_num - 1, []).append([float(r) for r in rect])
    for page_idx, page in enumerate(writer.pages):
        target_rects = target_rects_by_page.get(page_idx)
        annots = page.get('/Annots')
        if not target_rects or not annots:
            continue
        kept = ArrayObject()
        for ann_ref in annots:
            ann = ann_ref.get_object() if hasattr(ann_ref, 'get_object') else ann_ref
            rect = ann.get('/Rect')
            is_generated = ann.get('/Subtype') == '/FreeText' and rect and any(
                _rects_overlap([float(r) for r in rect], target_rect)
                for target_rect in target_rects
            )
            if not is_generated:
                kept.append(ann_ref)
        page[NameObject('/Annots')] = kept

def fill_form_with_annotations(form_pdf_bytes: bytes, field_mapping: dict[str, str], field_info: list[dict]) -> bytes:
    reader = PdfReader(io.BytesIO(form_pdf_bytes))
    writer = PdfWriter()
    writer.append(reader)
    field_lookup = {f['field_id']: f for f in field_info}
    _remove_previous_generated_annotations(writer, field_info)
    page_heights = {}
    for i, page in enumerate(reader.pages):
        mediabox = page.mediabox
        page_heights[i + 1] = float(mediabox.height)
    for field_id, value in field_mapping.items():
        if not value or field_id not in field_lookup:
            continue
        field = field_lookup[field_id]
        rect = field.get('rect')
        page_num = field.get('page', 1)
        if not rect:
            continue
        left, bottom, right, top = rect
        field_height = top - bottom
        font_size = min(max(int(field_height * 0.7), 8), 12)
        display_value = _annotation_display_value(field, value)
        if not display_value:
            continue
        try:
            if _is_w9_digit_group_field(field, display_value):
                _add_digit_group_annotations(
                    writer,
                    page_num,
                    [left, bottom, right, top],
                    display_value,
                    font_size,
                )
                continue
            annotation = FreeText(text=display_value, rect=(left, bottom, right, top), font='Helvetica', font_size=f'{font_size}pt', font_color='000000', border_color=None, background_color=None)
            writer.add_annotation(page_number=page_num - 1, annotation=annotation)
        except Exception:
            continue
    return _write_pdf_bytes(writer)

def fill_form(form_pdf_bytes: bytes, field_mapping: dict[str, str], field_info: list[dict] | None=None) -> tuple[bytes, list[str], list[str]]:
    reader = PdfReader(io.BytesIO(form_pdf_bytes))
    all_fields = list((reader.get_fields() or {}).keys())
    filled = [f for f in field_mapping.keys() if f in all_fields]
    not_filled = [f for f in all_fields if f not in field_mapping]
    use_annotations = is_xfa_form(form_pdf_bytes)
    if use_annotations and field_info:
        pdf_bytes = fill_form_with_annotations(form_pdf_bytes, field_mapping, field_info)
    else:
        writer = PdfWriter(clone_from=reader)
        if field_mapping:
            if field_info:
                fields_by_page: dict[int, dict] = {}
                field_lookup = {f['field_id']: f for f in field_info}
                for field_id, value in field_mapping.items():
                    if field_id in field_lookup:
                        page = field_lookup[field_id].get('page', 1)
                        if page not in fields_by_page:
                            fields_by_page[page] = {}
                        fields_by_page[page][field_id] = value
                for page_num, page_fields in fields_by_page.items():
                    writer.update_page_form_field_values(writer.pages[page_num - 1], page_fields, auto_regenerate=False)
            else:
                writer.update_page_form_field_values(None, field_mapping, auto_regenerate=False)
        writer.set_need_appearances_writer(True)
        pdf_bytes = _write_pdf_bytes(writer)
    workspace_file('filled_form.pdf').write_bytes(pdf_bytes)
    return (pdf_bytes, filled, not_filled)

def create_summary_pdf(extracted_data: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    doc_type = extracted_data.get('document_type', 'Document')
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 12, f'{doc_type} -- Extracted Data', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 6, f"Generated: {time.strftime('%Y-%m-%d %H:%M')} | Model: {MODEL_NAME}", new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.ln(4)
    pdf.set_fill_color(220, 220, 220)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(70, 8, 'Field', border=1, fill=True)
    pdf.cell(0, 8, 'Value', border=1, fill=True, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 10)
    for i, (key, value) in enumerate(sorted(extracted_data.items())):
        if key == 'document_type':
            continue
        fill_color = (248, 248, 248) if i % 2 else (255, 255, 255)
        pdf.set_fill_color(*fill_color)
        display_value = str(value)
        if len(display_value) > 60:
            display_value = display_value[:57] + '...'
        pdf.cell(70, 8, key[:30], border=1, fill=True)
        pdf.cell(0, 8, display_value, border=1, fill=True, new_x='LMARGIN', new_y='NEXT')
    pdf.ln(6)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.multi_cell(0, 5, 'Note: The uploaded form had no interactive (AcroForm) fields.\nThis summary PDF lists all extracted values.\nTo fill an actual form, upload a fillable PDF with AcroForm fields.')
    buf = io.BytesIO()
    pdf.output(buf)
    pdf_bytes = buf.getvalue()
    workspace_file('filled_form.pdf').write_bytes(pdf_bytes)
    return pdf_bytes

class TaxFilerState(TypedDict, total=False):
    source_files: list[tuple[bytes, str]]
    form_bytes: bytes
    combined_markdown: str
    extracted_data: dict
    field_info: list[dict]
    has_form_fields: bool
    field_mapping: dict[str, str]
    filled_pdf: bytes
    filled_list: list[str]
    not_filled_list: list[str]

class TaxFilerReactState(TypedDict):
    messages: Annotated[list[Any], add_messages]

TAX_FILER_REACT_PROMPT = """You are a US tax form filing agent.

Goal: produce a filled PDF from uploaded source documents and a blank form.
The blank PDF can be any fillable form, not just a W-9. Infer the form's
purpose from labels, field IDs, field positions, and source data.

You decide which tool to call next. Use observations from previous tools before
choosing the next action. Do not claim the form is ready until
fill_or_summarize_pdf reports success.

Important filling behavior:
- For visual groups of boxes (SSN, EIN, ZIP, dates, phone, account numbers, or
  any per-character entry), preserve exact characters and digit order.
- Map the correct chunk/value to the PDF field. The filler can split grouped
  W-9 SSN values like 123, 45, and 6789 into individual visible boxes.
- For checkboxes/radio fields, select the matching option instead of treating
  backend values as literal text.
- If a field is low confidence, leave it unmapped rather than inventing data.

Useful strategy:
1. read_source_documents
2. extract_document_fields
3. inspect_blank_form
4. map_fields_to_form if fillable fields exist
5. fill_or_summarize_pdf

If a tool says a prerequisite is missing, call the missing prerequisite tool.
"""

def create_tax_filer_tools(
    artifacts: TaxFilerState,
    progress_callback: Callable[[str], None] | None = None,
) -> list[Any]:
    """Create stateful tools while keeping binary PDFs out of chat messages."""

    def progress(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    @tool
    def read_source_documents() -> str:
        """OCR all uploaded source documents and save combined markdown."""
        progress("Tool: read_source_documents")
        combined_markdown = run_ocr_multiple(artifacts["source_files"])
        artifacts["combined_markdown"] = combined_markdown
        return f"Read {len(artifacts['source_files'])} source document(s). Markdown length: {len(combined_markdown)}."

    @tool
    def extract_document_fields() -> str:
        """Extract all meaningful fields from the OCR markdown."""
        progress("Tool: extract_document_fields")
        markdown = artifacts.get("combined_markdown")
        if not markdown:
            return "Missing OCR markdown. Call read_source_documents first."
        extracted_data = extract_all_fields(markdown)
        artifacts["extracted_data"] = extracted_data
        field_count = len([key for key in extracted_data if key != "document_type"])
        return json.dumps({
            "document_type": extracted_data.get("document_type", "unknown"),
            "field_count": field_count,
            "field_names": list(extracted_data.keys()),
        })

    @tool
    def inspect_blank_form() -> str:
        """Inspect the blank PDF for fillable fields and positions."""
        progress("Tool: inspect_blank_form")
        field_info = extract_form_field_info(artifacts["form_bytes"])
        artifacts["field_info"] = field_info
        artifacts["has_form_fields"] = bool(field_info)
        if not field_info:
            return "No fillable AcroForm fields found. Use fill_or_summarize_pdf to create a summary PDF."
        return json.dumps({
            "field_count": len(field_info),
            "fields": [
                {
                    "field_id": field.get("field_id"),
                    "type": field.get("type"),
                    "page": field.get("page"),
                }
                for field in field_info[:30]
            ],
        })

    @tool
    def map_fields_to_form() -> str:
        """Map extracted fields to any inspected PDF form, including grouped digit boxes."""
        progress("Tool: map_fields_to_form")
        extracted_data = artifacts.get("extracted_data")
        field_info = artifacts.get("field_info")
        if not extracted_data:
            return "Missing extracted data. Call extract_document_fields first."
        if field_info is None:
            return "Missing blank form inspection. Call inspect_blank_form first."
        if not field_info:
            artifacts["field_mapping"] = {}
            return "No fillable fields exist, so no field mapping is needed."

        if PDF2IMAGE_AVAILABLE:
            form_images = convert_form_to_images(artifacts["form_bytes"])
            if form_images:
                field_mapping = smart_map_fields_with_vision(extracted_data, field_info, form_images)
            else:
                form_fields = [field["field_id"] for field in field_info]
                field_mapping = smart_map_fields(extracted_data, form_fields)
        else:
            form_fields = [field["field_id"] for field in field_info]
            field_mapping = smart_map_fields(extracted_data, form_fields)

        artifacts["field_mapping"] = field_mapping
        return json.dumps({
            "mapped_field_count": len(field_mapping),
            "mapped_fields": list(field_mapping.keys()),
        })

    @tool
    def fill_or_summarize_pdf() -> str:
        """Fill the blank PDF or create a summary PDF when no form fields exist."""
        progress("Tool: fill_or_summarize_pdf")
        extracted_data = artifacts.get("extracted_data")
        if not extracted_data:
            return "Missing extracted data. Call extract_document_fields first."

        has_form_fields = artifacts.get("has_form_fields", False)
        if has_form_fields:
            field_info = artifacts.get("field_info") or []
            field_mapping = artifacts.get("field_mapping")
            if field_mapping is None:
                return "Missing field mapping. Call map_fields_to_form first."
            filled_pdf, filled_list, not_filled_list = fill_form(
                artifacts["form_bytes"], field_mapping, field_info
            )
        else:
            filled_pdf = create_summary_pdf(extracted_data)
            filled_list = list(extracted_data.keys())
            not_filled_list = []
            artifacts["field_mapping"] = {}

        artifacts["filled_pdf"] = filled_pdf
        artifacts["filled_list"] = filled_list
        artifacts["not_filled_list"] = not_filled_list
        return json.dumps({
            "status": "filled_pdf_ready",
            "filled_count": len(filled_list),
            "unfilled_count": len(not_filled_list),
            "output_size_bytes": len(filled_pdf),
        })

    return [
        read_source_documents,
        extract_document_fields,
        inspect_blank_form,
        map_fields_to_form,
        fill_or_summarize_pdf,
    ]

def build_tax_filer_react_agent(tools: list[Any], model: Any | None = None):
    """Build the explicit StateGraph + ToolNode ReAct loop."""
    agent_model = model or ChatAnthropic(
        model=MODEL_NAME,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,
        max_tokens=4096,
    )
    bound_model = agent_model.bind_tools(tools)

    def call_model(state: TaxFilerReactState) -> dict:
        response = bound_model.invoke([
            SystemMessage(content=TAX_FILER_REACT_PROMPT),
            *state["messages"],
        ])
        return {"messages": [response]}

    def route_after_model(state: TaxFilerReactState) -> str:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    builder = StateGraph(TaxFilerReactState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_model, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()

def run_tax_filer_react_agent(
    source_files: list[tuple[bytes, str]],
    form_bytes: bytes,
    model: Any | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> TaxFilerState:
    """Run the ReAct agent and return the filled PDF artifacts."""
    artifacts: TaxFilerState = {"source_files": source_files, "form_bytes": form_bytes}
    tools = create_tax_filer_tools(artifacts, progress_callback=progress_callback)
    agent = build_tax_filer_react_agent(tools, model=model)
    agent.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "Fill the uploaded tax form. Decide which tools are needed, "
                    "call them in the right order, and stop only after the PDF is ready."
                )
            )
        ]
    })
    if "filled_pdf" not in artifacts:
        raise RuntimeError("Tax filer ReAct agent ended before creating a filled PDF.")
    return artifacts

def render_status_badge(connected: bool, label: str) -> str:
    cls = "badge-ok" if connected else "badge-err"
    return f'<div class="badge {cls}"><span class="badge-dot"></span>{label}</div>'

def render_metric_card(value: str, label: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>'
    )

def main() -> None:
    st.set_page_config(
        page_title='Tax Filer — AI Form Automation',
        page_icon='📄',
        layout='wide',
        initial_sidebar_state='collapsed',
    )
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

    has_llama = bool(LLAMA_CLOUD_API_KEY)
    has_anthropic = bool(ANTHROPIC_API_KEY)

    with st.sidebar:
        st.markdown('<p class="sidebar-section-title">System</p>', unsafe_allow_html=True)
        st.markdown(render_status_badge(has_llama, 'LlamaCloud'), unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(render_status_badge(has_anthropic, 'Anthropic'), unsafe_allow_html=True)
        if not (has_llama and has_anthropic):
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.code('LLAMA_CLOUD_API_KEY=...\nANTHROPIC_API_KEY=...', language='bash')
        st.markdown("<hr style='margin:24px 0; opacity:.4;'>", unsafe_allow_html=True)
        st.markdown(
            f'<p class="sidebar-section-title">Model</p>'
            f'<div class="sidebar-file">{MODEL_NAME}</div>',
            unsafe_allow_html=True,
        )
        if WORKSPACE.exists():
            files = list(WORKSPACE.iterdir())
            if files:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                st.markdown('<p class="sidebar-section-title">Workspace</p>', unsafe_allow_html=True)
                for f in sorted(files):
                    st.markdown(f'<div class="sidebar-file">{f.name}</div>', unsafe_allow_html=True)

    doc_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" '
        'viewBox="0 0 24 24" stroke-width="1.8" stroke="white">'
        '<path stroke-linecap="round" stroke-linejoin="round" '
        'd="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 '
        '13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125'
        '.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 '
        '1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"/></svg>'
    )
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-logo">
                <div class="app-logo-mark">{doc_svg}</div>
                <span class="app-logo-text">Tax Filer</span>
            </div>
            <div class="app-status-row">
                {render_status_badge(has_llama, 'LlamaCloud')}
                {render_status_badge(has_anthropic, 'Anthropic')}
                <span class="model-tag">{MODEL_NAME}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">✦ AI-Powered &nbsp;·&nbsp; Zero Manual Entry</div>
            <h1 class="hero-headline">Fill any tax form<br><em>in seconds.</em></h1>
            <p class="hero-sub">
                Upload your source documents and a blank IRS or state form.
                The AI agent reads every field and fills the form automatically —
                no copy-paste, no errors.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_anthropic or not has_llama:
        st.markdown(
            """
            <div class="config-err">
                <div class="config-err-title">Configuration Required</div>
                <div class="config-err-body">
                    Add your API keys to the <code>.env</code> file to continue.
                    Both <strong>LLAMA_CLOUD_API_KEY</strong> and
                    <strong>ANTHROPIC_API_KEY</strong> are required.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    st.markdown(
        """
        <div class="section-divider">
            <div class="section-divider-line"></div>
            <span class="section-label">Upload Files</span>
            <div class="section-divider-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap='large')

    with col1:
        st.markdown(
            """
            <div class="upload-card">
                <div class="step-label">
                    <div class="step-num">1</div>
                    <span class="step-name">Source Documents</span>
                </div>
                <p class="card-desc">
                    W-2, 1099, invoices, or any PDF with data to extract.
                    Multiple files supported.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        source_files = st.file_uploader(
            'Source documents',
            type=['pdf'],
            accept_multiple_files=True,
            help='One or more PDFs containing the data to extract.',
            label_visibility='collapsed',
            key='source_uploader',
        )
        if source_files:
            count = len(source_files)
            plural = 's' if count > 1 else ''
            names_html = ''.join(
                f'<div class="file-list-item">→ {f.name}</div>' for f in source_files
            )
            st.markdown(
                f"""
                <div class="file-chip file-chip-ok">
                    <span class="file-chip-icon">✓</span>
                    <div>
                        <div class="file-chip-name">{count} document{plural} ready</div>
                        {names_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            """
            <div class="upload-card">
                <div class="step-label">
                    <div class="step-num">2</div>
                    <span class="step-name">Blank Form</span>
                </div>
                <p class="card-desc">
                    The fillable IRS or state tax form PDF.
                    Works best with AcroForm-enabled PDFs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        form_file = st.file_uploader(
            'Blank form',
            type=['pdf'],
            help='Upload the blank PDF form to fill.',
            label_visibility='collapsed',
            key='form_uploader',
        )
        if form_file:
            form_bytes_preview = form_file.read()
            form_file.seek(0)
            preview_fields = get_acroform_fields(form_bytes_preview)
            if preview_fields:
                st.markdown(
                    f"""
                    <div class="file-chip file-chip-ok">
                        <span class="file-chip-icon">✓</span>
                        <div>
                            <div class="file-chip-name">{form_file.name}</div>
                            <div class="file-chip-meta">
                                {len(preview_fields)} fillable fields detected
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="file-chip file-chip-warn">
                        <span class="file-chip-icon">⚠</span>
                        <div>
                            <div class="file-chip-name">{form_file.name}</div>
                            <div class="file-chip-meta">
                                No fillable fields detected — will generate a summary PDF
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    can_run = bool(source_files and form_file)

    if not can_run:
        st.markdown(
            """
            <div class="waiting-card">
                <div class="waiting-title">Ready when you are</div>
                <div class="waiting-sub">Upload both files above to unlock processing</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        run_clicked = st.button('Process Documents  →', type='primary', use_container_width=True)

    if not run_clicked:
        st.stop()

    form_bytes = form_file.read()
    overall_start = time.time()
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    files_to_ocr = [(f.read(), f.name) for f in source_files]

    progress_messages: list[str] = []

    def safe_progress(msg: str) -> None:
        """Collect agent step messages without touching Streamlit context from threads."""
        progress_messages.append(msg)
        print(f'[AGENT] {msg}')

    with st.status('AI agent working…', expanded=True) as status:
        try:
            agent_state = run_tax_filer_react_agent(
                source_files=files_to_ocr,
                form_bytes=form_bytes,
                progress_callback=safe_progress,
            )
            elapsed = time.time() - overall_start
            status.update(
                label=f'Completed in {elapsed:.1f}s',
                state='complete',
                expanded=False,
            )
            if progress_messages:
                with st.expander('Agent Step Log', expanded=False):
                    for msg in progress_messages:
                        st.write(msg)
        except Exception as e:
            status.update(label='Agent encountered an error', state='error')
            st.error(f'Processing failed: {e}')
            st.code(traceback.format_exc(), language='python')
            st.stop()

    extracted_data = agent_state['extracted_data']
    has_form_fields = agent_state['has_form_fields']
    field_mapping = agent_state['field_mapping']
    filled_pdf = agent_state['filled_pdf']
    filled_list = agent_state['filled_list']
    not_filled_list = agent_state['not_filled_list']
    total_elapsed = time.time() - overall_start

    check_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" fill="none" '
        'viewBox="0 0 24 24" stroke-width="2.5" stroke="white">'
        '<path stroke-linecap="round" stroke-linejoin="round" d="m4.5 12.75 6 6 9-13.5"/>'
        '</svg>'
    )
    st.markdown(
        f"""
        <div class="results-banner">
            <div class="results-check">{check_svg}</div>
            <div class="results-headline">Form Filled Successfully</div>
            <div class="results-sub">Finished in {total_elapsed:.1f} seconds</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            render_metric_card(str(len(extracted_data) - 1), 'Fields Extracted'),
            unsafe_allow_html=True,
        )
    with m2:
        filled_val = str(len(filled_list)) if has_form_fields else '—'
        st.markdown(
            render_metric_card(filled_val, 'Fields Filled'),
            unsafe_allow_html=True,
        )
    with m3:
        size_kb = len(filled_pdf) / 1024
        st.markdown(
            render_metric_card(f'{size_kb:.1f} KB', 'Output Size'),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    _, dl_col, _ = st.columns([1, 2, 1])
    with dl_col:
        st.download_button(
            label='Download Filled Form  ↓',
            data=filled_pdf,
            file_name='filled_form.pdf',
            mime='application/pdf',
            type='primary',
            use_container_width=True,
        )

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    with st.expander('View Extracted Data', expanded=False):
        st.json(extracted_data)

    if has_form_fields and field_mapping:
        with st.expander('View Field Mapping', expanded=False):
            st.json(field_mapping)
            if not_filled_list:
                overflow = f' (+{len(not_filled_list) - 10} more)' if len(not_filled_list) > 10 else ''
                st.markdown(
                    f"""
                    <div class="unmapped-warn">
                        <strong>Unmapped Fields</strong>
                        <div>{', '.join(not_filled_list[:10])}{overflow}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

if __name__ == '__main__':
    main()
